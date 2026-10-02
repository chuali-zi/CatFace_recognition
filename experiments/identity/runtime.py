"""CPU-only ONNX feature extraction. No PyTorch dependency on the board."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def oriented_rgb(image: Image.Image) -> Image.Image:
    """Decode/orient without copying an already upright RGB pixel buffer.

    May return the input itself; callers must treat its pixels as read-only.
    Load before inspecting EXIF: some decoders update orientation on load.
    """
    image.load()
    if image.getexif().get(274, 1) in range(2, 9):
        image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def preprocess(image: Image.Image, size: int = 160) -> np.ndarray:
    image = oriented_rgb(image)
    side = max(image.size)
    square = Image.new("RGB", (side, side), (128, 128, 128))
    square.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    square = square.resize((size, size), Image.Resampling.BILINEAR)
    array = np.asarray(square, dtype=np.float32).transpose(2, 0, 1) / np.float32(255)
    return np.ascontiguousarray((array - MEAN) / STD)


def read_image(path: Path, crop_box=None) -> Image.Image:
    """Decode once; optional offline box is in EXIF-corrected original pixels."""
    with Image.open(path) as source:
        image = oriented_rgb(source)
    if crop_box is not None:
        if (len(crop_box) != 4 or not all(type(value) is int for value in crop_box) or
                not (0 <= crop_box[0] < crop_box[2] <= image.width and
                     0 <= crop_box[1] < crop_box[3] <= image.height)):
            raise ValueError(f"invalid offline crop_box: {crop_box} for {image.size}")
        image = image.crop(tuple(crop_box))
    return image


def load_image(path: Path, size: int = 160, crop_box=None) -> np.ndarray:
    return preprocess(read_image(path, crop_box), size)


def normalize(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    if not np.isfinite(array).all() or np.any(norms <= 1e-12):
        raise ValueError("Embedding contains non-finite values or has zero norm")
    return array / norms


def model_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unit64(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(value)
    if value.ndim != 1 or not np.isfinite(value).all() or not np.isfinite(norm) or norm <= 0:
        raise ValueError("Expected a finite nonzero embedding vector")
    return value / norm


def make_gallery(rows: list[dict], embeddings: dict) -> tuple[list[str], list[np.ndarray]]:
    identities = sorted({row["cat_id"] for row in rows})
    vectors = [unit64(np.mean([unit64(embeddings[row["sample_id"]])
                              for row in rows if row["cat_id"] == cat], axis=0))
               for cat in identities]
    return identities, vectors


def match_gallery(embedding: np.ndarray, gallery: dict) -> tuple[str, float]:
    # float64 and the tie-break match the evaluator, including threshold boundaries.
    probe = unit64(embedding)
    return max(((cat, float(np.dot(probe, np.asarray(vector, dtype=np.float64))))
                for cat, vector in zip(gallery["cat_ids"], gallery["vectors"])),
               key=lambda item: (item[1], item[0]))


class Encoder:
    def __init__(self, model: Path, threads: int = 2):
        import onnxruntime as ort

        if threads < 1:
            raise ValueError("threads must be positive")
        self.model = Path(model)
        self.metadata = json.loads(self.model.with_suffix(".json").read_text(encoding="utf-8"))
        self.size = int(self.metadata["input_size"])
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(model), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def encode_array(self, array: np.ndarray) -> np.ndarray:
        result = self.session.run(None, {self.input_name: array[None]})[0]
        if result.shape != (1, int(self.metadata["embedding_dim"])):
            raise ValueError(f"Unexpected embedding shape: {result.shape}")
        return normalize(result)[0]

    def encode(self, path: Path, crop_box=None) -> np.ndarray:
        return self.encode_array(load_image(path, self.size, crop_box))


def embed_rows(encoder: Encoder, rows: list[dict], data_root: Path) -> dict:
    unique = {row["sample_id"]: row for row in rows}
    output = {}
    for index, (sample_id, row) in enumerate(unique.items(), 1):
        output[sample_id] = encoder.encode(data_root / row["path"], row.get("crop_box"))
        if index % 100 == 0 or index == len(unique):
            print(f"embedded {index}/{len(unique)}", flush=True)
    return output
