"""Apply the recorded visual review, never infer exhaustive labels from detections."""
import argparse
import json
from pathlib import Path

from .prepare import read_rows, write_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, default=Path(__file__).with_name("review-decisions.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = {row["sample_id"]: row for row in read_rows(args.manifest)}
    decisions = json.loads(args.decisions.read_text(encoding="utf-8"))
    reviewed = []
    for decision in decisions["decisions"]:
        row = dict(rows[decision["sample_id"]])
        if row["split"] == "train":
            raise ValueError("review decisions do not define training labels")
        if decision["annotation_scope"] != "exhaustive":
            continue
        row.update(annotation_scope="exhaustive", review_status="visually_checked",
                   review_note=decision["note"], review_index=decision["review_index"])
        reviewed.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_rows(args.out, reviewed)
    print(json.dumps({"reviewed": len(decisions["decisions"]), "exhaustive_subset": len(reviewed),
                      "out": str(args.out)}))


if __name__ == "__main__":
    main()
