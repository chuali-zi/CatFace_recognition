"""Build an identity gallery from full images using exactly one detector crop."""
import argparse, hashlib, json
from pathlib import Path
from PIL import Image, ImageOps

from .onnx import Detector, single_crop, crop_policy
from experiments.identity.enroll import build_calibrated_gallery
from experiments.identity.runtime import Encoder, model_digest, preprocess


def validate_rows(rows, root=Path(".")):
    if not rows: raise ValueError("manifest is empty")
    seen_paths, ids, sample_ids = set(), set(), set()
    normalized=[]
    for row in rows:
        if row.get("split") not in {"gallery", "calibration_unknown"}:
            raise ValueError("split must be gallery or calibration_unknown")
        if not isinstance(row.get("cat_id"), str) or not row["cat_id"].strip(): raise ValueError("cat_id must be nonempty")
        path = Path(row.get("path", ""))
        if not path.name: raise ValueError("image path is required")
        resolved = (path if path.is_absolute() else root / path).resolve()
        key = str(resolved)
        if key in seen_paths: raise ValueError("sample paths must be unique")
        if not resolved.is_file(): raise ValueError(f"image path does not exist: {path}")
        seen_paths.add(key)
        row=dict(row); row["path"] = key; row["sample_id"] = str(row.get("sample_id") or key)
        if row["sample_id"] in sample_ids: raise ValueError("sample_id must be unique")
        sample_ids.add(row["sample_id"])
        if row["split"] == "gallery": ids.add(row["cat_id"])
        normalized.append(row)
    if ids & {r["cat_id"] for r in normalized if r["split"] == "calibration_unknown"}:
        raise ValueError("gallery and calibration_unknown identities must be disjoint")
    if not ids or not any(r["split"] == "calibration_unknown" for r in normalized):
        raise ValueError("manifest needs gallery and calibration_unknown")
    return normalized


def _pixel_digest(image):
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def _write_failure(out, rows, states, error):
    path = Path(str(out) + ".failures.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"error": str(error), "states": states, "n_rows": len(rows)}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--detector",type=Path,required=True); p.add_argument("--model",type=Path,required=True); p.add_argument("--out",type=Path,required=True); p.add_argument("--threads",type=int,default=2); p.add_argument("--target-fpir",type=float,default=.01); p.add_argument("--crop-mode",choices=("head","face_core"),default="head")
    a=p.parse_args(argv)
    if a.threads<1 or not 0<=a.target_fpir<1: p.error("threads positive and target-fpir in [0,1)")
    if a.out.exists(): raise ValueError("output already exists; refusing overwrite")
    rows=[json.loads(x) for x in a.manifest.read_text(encoding="utf-8").splitlines() if x.strip()]
    try: rows=validate_rows(rows, a.manifest.parent)
    except Exception as exc:
        _write_failure(a.out, rows, [], exc); raise
    seen_pixels=set(); states=[]; embeddings={}; valid_cal=0
    try:
        for row in rows:
            with Image.open(row["path"]) as opened:
                image=ImageOps.exif_transpose(opened).convert("RGB")
            digest=_pixel_digest(image)
            if digest in seen_pixels: raise ValueError("duplicate image pixels")
            seen_pixels.add(digest)
    except Exception as exc:
        failure=_write_failure(a.out, rows, [], exc); raise ValueError(f"manifest validation failed; see {failure}") from exc
    detector=Detector(a.detector,a.threads); encoder=Encoder(a.model,a.threads)
    for row in rows:
        state={"sample_id":row.get("sample_id",str(row.get("path"))),"cat_id":row["cat_id"],"split":row["split"],"path":row["path"],"status":"failed"}
        try:
            with Image.open(row["path"]) as opened: image=ImageOps.exif_transpose(opened).convert("RGB")
            state["status"]="unique"
            boxes,scores=detector.detect(image); status,box,crop=single_crop(image,boxes,a.crop_mode)
            if status != "ok": raise ValueError(f"detector status: {status}")
            sample_id=state["sample_id"]; embeddings[sample_id]=encoder.encode_array(preprocess(crop,encoder.size)); state.update(status="ok",crop_box=box,score=float(scores[0]) if len(scores) else None)
            if row["split"] == "calibration_unknown": valid_cal += 1
        except Exception as exc:
            state.update(status="failed",error=str(exc))
            if row["split"] == "gallery":
                failure=_write_failure(a.out,rows,states+[state],"gallery enrollment failed"); raise ValueError(f"gallery enrollment failed; see {failure}") from exc
        states.append(state)
    if valid_cal == 0:
        failure=_write_failure(a.out,rows,states,"no calibration_unknown samples succeeded"); raise ValueError(f"calibration failed; see {failure}")
    usable=[r for r in rows if r.get("sample_id",str(r.get("path"))) in embeddings]
    result=build_calibrated_gallery(usable,embeddings,{"model_sha256":model_digest(a.model),"input_size":encoder.size,"embedding_dim":int(encoder.metadata["embedding_dim"])},a.target_fpir)
    result.update({"crop_pipeline":{"mode":"detected","crop_mode":a.crop_mode,"detector_model_sha256":detector.model_sha256,"detector_metadata_sha256":detector.metadata_sha256,"crop_policy":crop_policy(a.crop_mode)},"calibration_plan":len([r for r in rows if r["split"]=="calibration_unknown"]),"calibration_success":valid_cal,"calibration_status":states,"diagnostic_only":True})
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")

if __name__ == "__main__": main()
