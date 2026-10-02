"""Evaluate cat-face boxes on an image-level annotation manifest."""
import argparse, hashlib, json, statistics, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0., x2-x1) * max(0., y2-y1)
    aa = max(0., a[2]-a[0]) * max(0., a[3]-a[1]); ab = max(0., b[2]-b[0]) * max(0., b[3]-b[1])
    return inter / (aa + ab - inter) if aa + ab - inter else 0.


def map_boxes(boxes, sx, sy):
    return [[round(x1/sx), round(y1/sy), round(x2/sx), round(y2/sy)] for x1,y1,x2,y2 in boxes]


def head_context(box, width, height):
    x1,y1,x2,y2 = map(float, box); w=x2-x1; h=y2-y1
    cx=(x1+x2)/2; cy=(y1+y2)/2 - .15*h
    nw,nh=1.5*w,1.5*h
    return [max(0, round(cx-nw/2)), max(0, round(cy-nh/2)),
            min(width, round(cx+nw/2)), min(height, round(cy+nh/2))]


def match_boxes(predictions, truth, threshold=.5):
    pairs = sorted(((iou(p, g), pi, gi) for pi,p in enumerate(predictions) for gi,g in enumerate(truth) if iou(p,g) >= threshold), reverse=True)
    used_p, used_g, matched = set(), set(), []
    for score, pi, gi in pairs:
        if pi not in used_p and gi not in used_g:
            used_p.add(pi); used_g.add(gi); matched.append((pi, gi, score))
    return matched


def _rows(path, split):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = [r for r in rows if r.get("split") == split]
    if not selected: raise ValueError(f"manifest split is empty: {split}")
    for row in selected:
        if row.get("annotation_scope") not in {"single_target", "exhaustive"}:
            raise ValueError(f"unknown annotation_scope for {row.get('sample_id')}")
    if len({r.get("sample_id") for r in selected}) != len(selected): raise ValueError("duplicate manifest sample_id")
    return selected


def _resize(image, max_side):
    image = ImageOps.exif_transpose(image).convert("RGB")
    scale = min(1., max_side / max(image.size))
    size = (max(1, round(image.width*scale)), max(1, round(image.height*scale)))
    return image.resize(size, Image.Resampling.BILINEAR) if scale < 1 else image, (size[0]/image.width, size[1]/image.height)


def haar_detector(cascade_name, max_side, threads):
    import cv2
    cv2.setNumThreads(threads)
    filename = "haarcascade_frontalcatface.xml" if cascade_name == "basic" else "haarcascade_frontalcatface_extended.xml"
    cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / filename))
    if cascade.empty(): raise RuntimeError(f"Haar cascade unavailable: {filename}")
    def detect(image):
        resized, (sx, sy) = _resize(image, max_side)
        gray = cv2.cvtColor(np.asarray(resized), cv2.COLOR_RGB2GRAY)
        found = cascade.detectMultiScale(gray, 1.1, 3, minSize=(32,32), flags=cv2.CASCADE_SCALE_IMAGE)
        return map_boxes([[x,y,x+w,y+h] for x,y,w,h in found], sx, sy)
    return detect


def _size_group(box, width, height):
    fraction = max(0., box[2]-box[0]) * max(0., box[3]-box[1]) / max(1., width*height)
    return "small" if fraction < .01 else "medium" if fraction < .05 else "large"


def evaluate_rows(rows, predictions, image_sizes=None, timings=None, decode_ms=None):
    per_image=[]; size_stats={k:{"matched":0,"total":0} for k in ("small","medium","large")}
    tp=fp=fn=0
    for row in rows:
        pred = predictions.get(row["sample_id"], [])
        truth = row.get("boxes", []); matches=match_boxes(pred, truth)
        width,height=(image_sizes or {}).get(row["sample_id"], (0,0));
        for box in truth:
            group=_size_group(box,width,height); size_stats[group]["total"]+=1
        for _,gi,_ in matches:
            size_stats[_size_group(truth[gi],width,height)]["matched"]+=1
        tp += len(matches)
        scope=row.get("annotation_scope", "single_target")
        if scope == "exhaustive": fp += len(pred)-len(matches); fn += len(truth)-len(matches)
        per_image.append({"sample_id":row["sample_id"],"predicted_boxes":pred,"ground_truth_boxes":truth,"matched":len(matches),"total_targets":len(truth),"annotation_scope":scope,"decode_ms":(decode_ms or {}).get(row["sample_id"]),"detect_ms":(timings or {}).get(row["sample_id"])})
    exhaustive=all(r.get("annotation_scope","single_target")=="exhaustive" for r in rows)
    result={"n_images":len(rows),"matched":tp,"total_targets":sum(len(r.get("boxes",[])) for r in rows),"target_coverage":tp/max(1,sum(len(r.get("boxes",[])) for r in rows)),"empty_prediction_images":sum(not predictions.get(r["sample_id"],[]) for r in rows),"multiple_prediction_images":sum(len(predictions.get(r["sample_id"],[]))>1 for r in rows),"size_groups":size_stats,"per_image":per_image}
    if exhaustive:
        result.update({"tp":tp,"fp":fp,"fn":fn,"precision":tp/max(1,tp+fp),"recall":tp/max(1,tp+fn)})
    if timings:
        values=sorted(timings.values()); result["latencyP50_ms"]=statistics.median(values); result["latencyP95_ms"]=values[max(0,round(len(values)*.95)-1)]
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--data-root",type=Path,default=Path("data")); p.add_argument("--split",default="development"); p.add_argument("--out",type=Path,required=True); p.add_argument("--predictions",type=Path); p.add_argument("--backend",choices=("haar",)); p.add_argument("--cascade",choices=("basic","extended"),default="extended"); p.add_argument("--max-side",type=int,choices=(320,640),default=640); p.add_argument("--threads",type=int,default=2); p.add_argument("--head-context",action="store_true")
    a=p.parse_args(argv)
    if a.threads<1: p.error("threads must be positive")
    if bool(a.predictions)==bool(a.backend): p.error("choose exactly one of --predictions or --backend haar")
    rows=_rows(a.manifest,a.split); predictions={}; sizes={}; timings={}; decode_ms={}
    for row in rows:
        if "width" in row and "height" in row: sizes[row["sample_id"]]=(int(row["width"]),int(row["height"]))
    selected_ids={r["sample_id"] for r in rows}
    if a.predictions:
        seen=set()
        for line in a.predictions.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r=json.loads(line); sid=r["sample_id"]
                if sid in seen: raise ValueError(f"duplicate prediction sample_id: {sid}")
                seen.add(sid)
                if sid in selected_ids: predictions[sid]=r.get("boxes", r.get("predicted_boxes", [])); timings[sid]=r.get("detect_ms")
        missing=selected_ids-seen
        if missing: raise ValueError(f"missing predictions for selected samples: {len(missing)}")
        timings={k:v for k,v in timings.items() if v is not None}
    else:
        detect=haar_detector(a.cascade,a.max_side,a.threads)
        for i,row in enumerate(rows,1):
            path=Path(row["path"]); path=path if path.is_absolute() else a.data_root/path
            t=time.perf_counter()
            with Image.open(path) as source:
                image=ImageOps.exif_transpose(source).convert("RGB"); sizes[row["sample_id"]]=image.size; d=time.perf_counter(); boxes=detect(image)
            decode_ms[row["sample_id"]]=(d-t)*1000; timings[row["sample_id"]]=(time.perf_counter()-d)*1000; predictions[row["sample_id"]]=boxes
    if a.head_context:
        predictions={sid:[head_context(box,*sizes[sid]) for box in boxes] for sid,boxes in predictions.items()}
    result=evaluate_rows(rows,predictions,sizes,timings,decode_ms); result.update({"split":a.split,"backend":"predictions" if a.predictions else "haar","timing_scope":"external detect_ms only; no Haar timing" if a.predictions else "decode separate; detect includes resize, grayscale and Haar scan"})
    if not a.predictions:
        import cv2
        result["metadata"]={"manifest_sha256":hashlib.sha256(a.manifest.read_bytes()).hexdigest(),"cascade":a.cascade,"max_side":a.max_side,"threads":a.threads,"opencv_version":cv2.__version__,"detector_parameters":{"scaleFactor":1.1,"minNeighbors":3,"minSize":[32,32]},"head_context":"width,height x1.5; center_y up 0.15*h; clipped to manifest image bounds" if a.head_context else None,"limitations":"coverage on annotated targets; not AP; single_target does not support precision/FP"}
    else:
        result["metadata"]={"manifest_sha256":hashlib.sha256(a.manifest.read_bytes()).hexdigest(),"head_context":"width,height x1.5; center_y up 0.15*h; clipped to manifest image bounds" if a.head_context else None,"limitations":"external predictions review; not AP; timing uses optional predictions detect_ms"}
    a.out.mkdir(parents=True,exist_ok=True); (a.out/"per-image.jsonl").write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in result.pop("per_image"))+"\n",encoding="utf-8"); (a.out/"summary.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__ == "__main__": main()
