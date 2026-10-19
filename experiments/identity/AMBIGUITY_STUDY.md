# Optional Ambiguity Rejection Experiment

This experiment preserves the frozen cosine threshold and adds an optional
minimum difference between the first and second identity similarities. The
default `--min-margin 0` disables it. No existing recognizer or sealed kit is
modified. This is separate from the existing `score_mode=gap`: both the original
cosine threshold and the additional similarity difference must pass here.

Run from the source repository with the existing inference Python environment.
`--kit` points to the complete v4 kit, and `--out` must be a new directory.

```bash
python -m experiments.identity.ambiguity_study --kit /home/user/catface-device-research-kit-v4 --out /home/user/ambiguity-baseline --models large224 small160
python -m experiments.identity.ambiguity_study --kit /home/user/catface-device-research-kit-v4 --out /home/user/ambiguity-margin005 --models large224 small160 --min-margin 0.005
```

These commands compare cached, hash-verified reference scores. They do not run
image inference or measure speed. Reports separate known queries, unknown
calibration queries and unknown developer queries. A preprocessing failure stays
in the denominator and is not turned into a successful identity rejection.

For actual single-image inference using the frozen detector, crop policy,
encoder and gallery:

```bash
python -m experiments.identity.ambiguity_study --mode recognize --kit /home/user/catface-device-research-kit-v4 --out /home/user/ambiguity-image --models large224 --min-margin 0.005 --images /home/user/catface-device-research-kit-v4/assets/images/0002.JPG
```

Image reports include the original decision, experimental decision, top score,
similarity difference and rejection reason. No-cat and unusable detections retain
their status. Model loading or decoding failures raise instead of inventing a
prediction. A single-identity gallery cannot satisfy an enabled ambiguity gate.

The proposed 0.005 was chosen after inspecting exposed development failures.
Reports therefore always say `calibrated=false` and `production_ready=false`.
An apparent improvement on these developer queries is not independent evidence.
Keep this rule disabled outside the experiment until its parameters are fixed
using appropriate calibration data and evaluated on untouched queries.

Repeated images, score replays and synthetic unit tests do not increase the
number of independent identity samples. Do not alter the sealed kit's manifest,
gallery or threshold to install this experiment.
