"""Read-only probe for the two suspicious SE activation QDQ points."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from .runtime import load_image


TARGETS = (
    "features.1/block/block.1/activation/Relu_output_0",
    "features.4/block/block.2/activation/Relu_output_0",
)
ACTIVATIONS = (
    "features.0/features.0.2/HardSwish_output_0",
    "features.1/block/block.0/block.0.2/Relu_output_0",
)


def options():
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    return so


def session_with_outputs(path: Path, needles):
    model = onnx.load(str(path))
    existing = {x.name for x in model.graph.output}
    by_name = {x.name: x for x in model.graph.value_info}
    by_name.update({x.name: x for x in model.graph.input})
    by_name.update({x.name: x for x in model.graph.output})
    names = []
    for node in model.graph.node:
        for output in node.output:
            # Probe the float ReLU output, or the post-QDQ float tensor.  Do
            # not expose the intermediate int8 QuantizeLinear output as float.
            if any(
                output.endswith(needle)
                or output.endswith(needle.replace("_output_0", "_output_0_DequantizeLinear_Output"))
                for needle in needles
            ):
                names.append(output)
    for name in names:
        if name not in existing:
            model.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
    return ort.InferenceSession(model.SerializeToString(), sess_options=options(), providers=["CPUExecutionProvider"]), names


def conv_output_names(path: Path, quantized: bool):
    model = onnx.load(str(path))
    names = []
    for node in model.graph.node:
        if node.op_type == "Conv":
            for output in node.output:
                names.append(output + "_DequantizeLinear_Output" if quantized else output)
    return names


def stats(values):
    arr = np.asarray(values)
    return {"min": float(arr.min()), "max": float(arr.max()),
            "mean": float(arr.mean()), "nonzero": int(np.count_nonzero(arr)),
            "size": int(arr.size)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("experiments/runs/edge160"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    root = args.root
    meta = json.loads((root / "baseline-int8/model.json").read_text(encoding="utf-8"))
    rows = meta["quantization_samples"][:3]
    fp, fp_names = session_with_outputs(root / "baseline-int8/model.preprocessed.onnx", TARGETS)
    qi, qi_names = session_with_outputs(root / "baseline-int8/model.onnx", TARGETS)
    fp_act, fp_act_names = session_with_outputs(root / "baseline-int8/model.preprocessed.onnx", ACTIVATIONS)
    qi_act, qi_act_names = session_with_outputs(root / "baseline-int8/model.onnx", ACTIVATIONS)
    # The first activation is explicitly QDQ-wrapped; the depthwise block's
    # ReLU is not quantized as an op, so its semantic counterpart is the
    # identically named float output after the quantized Conv.
    fp_act_names = [name for name in fp_act_names if name.endswith(ACTIVATIONS[0]) or name.endswith(ACTIVATIONS[1])]
    qi_act_names = []
    for target in ACTIVATIONS:
        candidates = [name for name in qi_act_names if name.endswith(target)]
        # (Names are populated below from the graph session output list.)
    qi_act_names = [name.name for name in qi_act.get_outputs() if name.name.endswith(ACTIVATIONS[1]) or name.name.endswith(ACTIVATIONS[0] + "_DequantizeLinear_Output")]
    result = {"samples": rows, "float": {}, "quant": {}, "float_tensor_names": fp_names,
              "quant_tensor_names": qi_names}
    result["activation_comparison"] = []
    # Isolate quant_pre_process from quantization itself.
    raw = ort.InferenceSession(str(root / "baseline/model.onnx"), sess_options=options(), providers=["CPUExecutionProvider"])
    prep = ort.InferenceSession(str(root / "baseline-int8/model.preprocessed.onnx"), sess_options=options(), providers=["CPUExecutionProvider"])
    result["raw_vs_preprocessed"] = []
    for row in rows:
        x = load_image(Path("data") / row, 160)[None]
        raw_out = raw.run(None, {"image": x})[0][0]
        prep_out = prep.run(None, {"image": x})[0][0]
        result["raw_vs_preprocessed"].append({
            "cosine": float(np.dot(raw_out, prep_out)),
            "max_abs": float(np.max(np.abs(raw_out - prep_out))),
        })
        for label, sess, names in (("float", fp, fp_names), ("quant", qi, qi_names)):
            out = sess.run(names, {"image": x})
            for name, value in zip(names, out):
                result[label].setdefault(name, []).append(stats(value))
        for fp_name, qi_name in zip(fp_act_names, qi_act_names):
            a = fp_act.run([fp_name], {"image": x})[0].astype(np.float32).ravel()
            b = qi_act.run([qi_name], {"image": x})[0].astype(np.float32).ravel()
            result["activation_comparison"].append({"tensor": fp_name,
                "cosine": float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)),
                "relative_l2": float(np.linalg.norm(a - b) / max(float(np.linalg.norm(a)), 1e-12)),
                "float_min": float(a.min()), "float_max": float(a.max()),
                "float_nonzero": int(np.count_nonzero(a))})
    # Compare every Conv's float output with the post-QDQ float output.  This
    # is intentionally limited to the same three already selected samples.
    fp_stage_names = conv_output_names(root / "baseline-int8/model.preprocessed.onnx", False)
    qi_stage_names = conv_output_names(root / "baseline-int8/model.onnx", True)
    fp_stage, _ = session_with_outputs(root / "baseline-int8/model.preprocessed.onnx", fp_stage_names)
    qi_stage, _ = session_with_outputs(root / "baseline-int8/model.onnx", qi_stage_names)
    raw_stage, _ = session_with_outputs(root / "baseline/model.onnx", fp_stage_names)
    raw_stage_rows = []
    stage_rows = []
    fp_stage_values = {name: [] for name in fp_stage_names}
    qi_stage_values = {name: [] for name in qi_stage_names}
    raw_stage_values = {name: [] for name in fp_stage_names}
    for row in rows:
        x = load_image(Path("data") / row, 160)[None]
        for name, value in zip(fp_stage_names, fp_stage.run(fp_stage_names, {"image": x})):
            fp_stage_values[name].append(value.astype(np.float32).ravel())
        for name, value in zip(qi_stage_names, qi_stage.run(qi_stage_names, {"image": x})):
            qi_stage_values[name].append(value.astype(np.float32).ravel())
        for name, value in zip(fp_stage_names, raw_stage.run(fp_stage_names, {"image": x})):
            raw_stage_values[name].append(value.astype(np.float32).ravel())
    for fp_name, qi_name in zip(fp_stage_names, qi_stage_names):
        errors = []
        for a, b in zip(fp_stage_values[fp_name], qi_stage_values[qi_name]):
            denom = max(float(np.linalg.norm(a)), 1e-12)
            errors.append({"cosine": float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)),
                           "relative_l2": float(np.linalg.norm(a - b) / denom),
                           "float_nonzero": int(np.count_nonzero(a)),
                           "float_max_abs": float(np.max(np.abs(a)))})
        stage_rows.append({"tensor": fp_name, "errors": errors,
                           "mean_relative_l2": float(np.mean([e["relative_l2"] for e in errors])),
                           "mean_cosine": float(np.mean([e["cosine"] for e in errors]))})
    result["conv_stage_comparison"] = sorted(stage_rows, key=lambda x: x["mean_relative_l2"], reverse=True)
    for name in fp_stage_names:
        errors = []
        for a, b in zip(raw_stage_values[name], fp_stage_values[name]):
            errors.append(float(np.linalg.norm(a - b) / max(float(np.linalg.norm(a)), 1e-12)))
        raw_stage_rows.append({"tensor": name, "mean_relative_l2": float(np.mean(errors))})
    result["raw_vs_preprocessed_conv"] = raw_stage_rows
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
