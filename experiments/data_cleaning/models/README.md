---
license: apache-2.0
pipeline_tag: image-feature-extraction
tags:
  - computer-vision
  - image-retrieval
  - animal-re-identification
  - cat-identification
  - pytorch
  - onnx
  - tensorrt
---

<p align="center">
  <img src="assets/logo.png" alt="MeowID" width="420">
</p>

<h1 align="center">MeowID: A Dual-Expert Retrieval System for Individual Cat Identification</h1>

<p align="center">
  <img alt="Version" src="https://img.shields.io/badge/version-0.3.0-11bfae">
  <img alt="Embedding" src="https://img.shields.io/badge/embedding-512D-0875c1">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-supported-ee4c2c">
  <img alt="ONNX" src="https://img.shields.io/badge/ONNX-supported-005ced">
  <img alt="TensorRT" src="https://img.shields.io/badge/TensorRT-supported-032b4a">
</p>

<p align="center">
  <strong>TL;DR:</strong> MeowID prioritizes fine-grained facial evidence, augments it with whole-cat context, and falls back to whole-cat retrieval when a usable face is unavailable.
</p>

<div align="center">
  Zhangchi Hu<sup>1,2,*,†</sup>,
  Yi Shang<sup>2,*</sup>,
  Haocheng Yang<sup>4,2,*</sup>,
  Qiwei Hu<sup>5,*</sup>,
  and Yuzheng Li<sup>3,*</sup>
</div>

<p></p>

<div align="center"><sub>
  <sup>1</sup> Department of Electronic Engineering and Information Science, University of Science and Technology of China<br>
  <sup>2</sup> School of Intelligent Software Engineering, Hefei University of Technology<br>
  <sup>3</sup> School of Software Engineering, Sun Yat-sen University<br>
  <sup>4</sup> School of Computer Science, Northwestern Polytechnical University<br>
  <sup>5</sup> College of Biological Sciences and Technology, Beijing Forestry University
</sub></div>

<p align="center">
  <sup>*</sup> Equal contribution &nbsp;&nbsp; <sup>†</sup> Project leader
</p>

## Model overview

MeowID is a face-priority, dual-expert retrieval system for identifying individual cats in unconstrained photographs. It combines separately parameterized face and whole-cat encoders while keeping their embedding galleries route-specific.

- When a usable aligned face is available, the face expert produces the primary representation and receives a gated whole-cat correction.
- When facial evidence is unavailable, the system falls back to the whole-cat expert.
- New identities can be enrolled through embedding extraction and gallery insertion without retraining the recognition models.
- All retrieval embeddings are L2-normalized, 512-dimensional vectors.

## Method pipeline

<p align="center">
  <img src="assets/meowid-pipeline.png" alt="MeowID method pipeline" width="100%">
</p>

The whole-cat expert produces an embedding for every image. A valid ECPose detection activates PetFace-style face alignment, the face expert, and validation-guided whole-cat hint fusion. Queries are compared only with the gallery associated with their selected route.

## Repository contents

| Path | Contents | Intended use |
| --- | --- | --- |
| `artifacts/MeowID-Base/` | MeowID-Base and ECPose weights in PyTorch, ONNX, and TensorRT formats | End-to-end identification and deployment |
| `artifacts/ECSeg/` | ECSeg-X segmentation weights | Whole-cat instance extraction and cropping |
| `artifacts/training_init/` | Whole-cat and face expert initialization checkpoints | Training and reproduction |
| `artifacts/**/SHA256SUMS` | Published SHA256 checksums | Artifact integrity verification |

The TensorRT engines were built for the reference RTX 3090 environment. Rebuild them from the ONNX artifacts when the GPU architecture, TensorRT version, or batch profile changes.

## Inference capabilities

| Capability | Details |
| --- | --- |
| Face localization | ECPose with 9 cat-face landmarks |
| Face alignment | PetFace-style three-point similarity alignment with a landmark-crop fallback |
| Recognition | Separate DINOv3-based face and whole-cat experts |
| Fusion | Validation-guided, gated whole-cat residual for the face route |
| Retrieval | Route-specific galleries with normalized inner-product similarity |
| Backends | PyTorch, ONNX Runtime CPU/CUDA, and TensorRT FP16/FP32 |
| Whole-cat cropping | ECSeg-X instance segmentation with masks, boxes, and padded crops |

## Minimal Python usage

```python
from cat_recognition import MeowID

model = MeowID(
    "artifacts/MeowID-Base",
    backend="tensorrt",
    device="cuda:0",
    registry="registries/demo",
)

model.register(
    "cat_001",
    ["images/cat_001_a.jpg", "images/cat_001_b.jpg"],
)

prediction = model.search("images/query.jpg", top_k=5)[0]
print("route:", prediction.embedding.route)
for match in prediction.matches:
    print(match.cat_id, match.score)
```

The package accepts file paths, directories, glob patterns, PIL images, RGB NumPy arrays, and iterables of supported inputs.

## Reference results

Offline retrieval on the ICW test set:

| Route | Top-1 | mAP |
| --- | ---: | ---: |
| Whole-cat expert | 51.34% | 59.00% |
| Cat-face expert | 78.80% | 83.32% |
| MeowID-Base hard routing | **75.93%** | **80.45%** |

End-to-end batch-1 measurements on one RTX 3090 over 2,846 ICW test images include image decoding, preprocessing, ECPose, alignment, embedding extraction, and routing:

| Backend | Mean latency | Throughput |
| --- | ---: | ---: |
| PyTorch FP32 | 94.23 ms | 10.61 images/s |
| ONNX Runtime CPU | 478.67 ms | 2.09 images/s |
| ONNX Runtime CUDA | 79.88 ms | 12.51 images/s |
| TensorRT FP16 | **60.00 ms** | **16.66 images/s** |

These results describe the reference evaluation environment and do not guarantee production performance.

## Model mirrors

- [Hugging Face — RicePasteM/MeowID-Base](https://huggingface.co/RicePasteM/MeowID-Base)
- [ModelScope — RicePasteM/MeowID-Base](https://modelscope.cn/models/RicePasteM/MeowID-Base)


## Citation

```bibtex
@misc{hu2026meowid,
  title  = {MeowID: A Dual-Expert Retrieval System for Individual Cat Identification},
  author = {Zhangchi Hu and Yi Shang and Haocheng Yang and Qiwei Hu and Yuzheng Li},
  year   = {2026}
}
```
