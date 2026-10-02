"""Small CPU benchmark for one detector or detector+identity pipeline."""
import argparse, json, platform, time
from pathlib import Path
import psutil
from PIL import Image, ImageOps

from .evaluate import haar_detector, head_context


def _percentile(values, p):
    return float(__import__("numpy").percentile(values, p))


def _rss():
    proc = psutil.Process(); info = proc.memory_info()
    peak = getattr(info, "peak_wset", None)
    if peak is None:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() != "Darwin":
            peak *= 1024
    return peak / 2**20


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True); p.add_argument("--data-root",type=Path,default=Path("data")); p.add_argument("--split",default="development")
    p.add_argument("--backend",choices=("haar","onnx","pipeline"),required=True); p.add_argument("--model",type=Path); p.add_argument("--identity-model",type=Path); p.add_argument("--gallery",type=Path)
    p.add_argument("--cascade",choices=("basic","extended"),default="extended"); p.add_argument("--max-side",type=int,choices=(320,640),default=320); p.add_argument("--head-context",action="store_true")
    p.add_argument("--threads",type=int,default=2); p.add_argument("--warmup",type=int,default=20); p.add_argument("--runs",type=int,default=100)
    p.add_argument("--out",type=Path)
    a=p.parse_args(argv)
    if a.threads<1 or a.warmup<0 or a.runs<1: p.error("threads/runs must be positive and warmup non-negative")
    if a.backend in {"onnx","pipeline"} and not a.model: p.error("--model required")
    if a.backend=="pipeline" and (not a.identity_model or not a.gallery): p.error("pipeline requires --identity-model and --gallery")
    if a.head_context and a.backend != "haar": p.error("--head-context is only the Haar box-definition comparison")
    rows=[json.loads(x) for x in a.manifest.read_text(encoding="utf-8").splitlines() if x.strip() and json.loads(x).get("split")==a.split][:20]
    if not rows: raise ValueError("selected manifest split is empty")
    images=[]; source_ids=[]
    for row in rows:
        path=Path(row["path"]); path=path if path.is_absolute() else a.data_root/path
        with Image.open(path) as source: images.append(ImageOps.exif_transpose(source).convert("RGB")); source_ids.append(row["sample_id"])
    loaded=time.perf_counter(); detector=None; encoder=None; gallery=None
    if a.backend=="haar": detector=haar_detector(a.cascade,a.max_side,a.threads)
    else:
        from .onnx import Detector
        detector=Detector(a.model,a.threads)
        if a.backend=="pipeline":
            from experiments.identity.deploy import read_gallery
            from experiments.identity.runtime import Encoder
            encoder=Encoder(a.identity_model,a.threads); gallery=read_gallery(a.gallery,encoder)
    load_ms=(time.perf_counter()-loaded)*1000
    def run_one(image):
        if a.backend=="pipeline":
            from .replay import replay
            return replay(image,detector,encoder,gallery)
        started=time.perf_counter(); boxes=detector(image) if a.backend=="haar" else detector.detect(image)[0]
        if hasattr(boxes, "tolist"): boxes=boxes.tolist()
        if a.head_context: boxes=[head_context(box,image.width,image.height) for box in boxes]
        return {"face_boxes":boxes,"detect_ms":(time.perf_counter()-started)*1000}
    for i in range(a.warmup): run_one(images[i%len(images)])
    detector_ms=[]; frame_ms=[]; identity_ms=[]; status={}; identity_count=0
    for i in range(a.runs):
        started=time.perf_counter(); result=run_one(images[i%len(images)]); frame_ms.append((time.perf_counter()-started)*1000)
        detector_ms.append(float(result["detect_ms"]));
        if a.backend=="pipeline":
            status[result["status"]]=status.get(result["status"],0)+1
            if result["status"] in {"known","unknown"}: identity_count+=1; identity_ms.append(float(result["identity_and_match_ms"]))
    report={"backend":a.backend,"split":a.split,"source_ids":source_ids,"distinct_images":len(images),"warmup":a.warmup,"runs":a.runs,"threads":a.threads,"model_load_ms":load_ms,"rss_peak_mib":_rss(),"detector_ms":{"p50":_percentile(detector_ms,50),"p95":_percentile(detector_ms,95)},"frame_ms":{"p50":_percentile(frame_ms,50),"p95":_percentile(frame_ms,95)},"timing_scope":"decode and image cache excluded; detector includes resize/scan or ORT; frame includes all branches"}
    if a.backend=="pipeline": report.update({"identity_executions":identity_count,"status_counts":status,"identity_and_match_ms_conditional":{"p50":_percentile(identity_ms,50) if identity_ms else None,"p95":_percentile(identity_ms,95) if identity_ms else None},"timing_scope":"decode/cache excluded; identity timing is conditional on exactly one valid detection"})
    try:
        import cv2; report["opencv_version"]=cv2.__version__
    except ImportError: pass
    try:
        import onnxruntime as ort; report["onnxruntime_version"]=ort.__version__
    except ImportError: pass
    from .onnx import _sha256
    report.update({"platform":platform.platform(),"processor":platform.processor(),"machine":platform.machine(),
                   "python":platform.python_version(),"head_context":a.head_context,
                   "haar": {"cascade":a.cascade,"max_side":a.max_side} if a.backend=="haar" else None,
                   "model_sha256":_sha256(a.model) if a.model else None,
                   "identity_model_sha256":_sha256(a.identity_model) if a.identity_model else None,
                   "scope":"CPU research benchmark; no extrapolation to another device; process peak includes decoded cache and model load"})
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2,ensure_ascii=False))

if __name__ == "__main__": main()
