"""Open-set identity metrics on explicit calibration/test prediction rows."""

import math


def threshold_at_fpir(scores: list[float], target: float) -> float:
    """Choose a threshold on calibration unknowns; acceptance uses >=."""
    if not scores or not all(math.isfinite(score) for score in scores):
        raise ValueError("Need finite unknown calibration scores")
    if not 0 <= target < 1:
        raise ValueError("target_fpir must be in [0, 1)")
    ordered = sorted(scores, reverse=True)
    allowed = math.floor(target * len(ordered))
    threshold = math.nextafter(ordered[allowed], math.inf)
    if not math.isfinite(threshold):
        raise ValueError("Scores are too large to represent a finite threshold")
    return threshold


def evaluate(rows: list[dict], target_fpir: float = 0.01) -> dict:
    groups = {"calibration_unknown": [], "known": [], "unknown": []}
    seen = set()
    for row in rows:
        sample_id = row["sample_id"]
        if not isinstance(sample_id, str) or not sample_id.strip() or sample_id in seen:
            raise ValueError("sample_id must be nonempty and unique across all splits")
        seen.add(sample_id)
        if row["split"] not in groups:
            raise ValueError("Invalid prediction split")
        if not isinstance(row["cat_id"], str) or not row["cat_id"].strip():
            raise ValueError("Every probe needs a ground-truth cat_id, including unknown cats")
        if not isinstance(row["predicted_id"], str) or not row["predicted_id"].strip():
            raise ValueError("predicted_id must be the top-ranked gallery identity before thresholding")
        score = row["score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError("score must be a finite number")
        groups[row["split"]].append(row)
    if any(not group for group in groups.values()):
        raise ValueError("Need calibration_unknown, known and unknown probes")
    ids = {name: {row["cat_id"] for row in group} for name, group in groups.items()}
    if (ids["calibration_unknown"] & ids["unknown"] or
            ids["known"] & (ids["calibration_unknown"] | ids["unknown"])):
        raise ValueError("Known, calibration-unknown and test-unknown identities must be disjoint")
    unknown_ids = ids["calibration_unknown"] | ids["unknown"]
    if any(row["predicted_id"] in unknown_ids for row in rows):
        raise ValueError("An unknown identity cannot also be a gallery prediction")
    threshold = threshold_at_fpir([row["score"] for row in groups["calibration_unknown"]], target_fpir)
    known = groups["known"]
    unknown = groups["unknown"]
    correct = [row["cat_id"] == row["predicted_id"] for row in known]
    accepted = [row["score"] >= threshold for row in known]
    return {
        "target_calibration_fpir": target_fpir, "threshold": threshold,
        "n_calibration_unknown": len(groups["calibration_unknown"]),
        "n_known": len(known), "n_unknown": len(unknown),
        "top1": sum(correct) / len(known),
        "dir": sum(a and c for a, c in zip(accepted, correct)) / len(known),
        "known_misidentification_rate": sum(a and not c for a, c in zip(accepted, correct)) / len(known),
        "known_reject_rate": sum(not a for a in accepted) / len(known),
        "fpir": sum(row["score"] >= threshold for row in unknown) / len(unknown),
    }
