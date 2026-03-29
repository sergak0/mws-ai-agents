# Competition Notes

## Observed data reality

- The competition files are `train.csv`, `test.csv`, `sample_submition.csv`, and `solution.csv`.
- The observed target is bounded in the range `0..365`.
- The target is not binary in the released data files.

## Offline evaluation

- `solution.csv` contains `prediction` and `Usage`.
- `Usage` splits rows into `Public` and `Private`.
- This makes it possible to run an offline benchmark loop in addition to holdout validation.

## Practical modeling cues

- Use a strong tabular baseline first.
- Engineer date parts and simple text-derived signals before complex methods.
- Preserve reproducibility with saved configs, reports, and submissions.

