# Hypothesis Forge Research Instructions

## Purpose

Improve the operational-message classifier through small, controlled experiments.

The classifier predicts exactly one category:

- `incident`
- `maintenance`
- `non_incident`

## Objective

Maximize validation accuracy reported by:

```bash
python evaluate.py
```

The current best validation accuracy is `100.0%`.

Because accuracy is already 100%, keep an equal-scoring experiment only when it clearly simplifies the model, reduces noisy features, or improves interpretability.

## Files

You may modify:

- `candidate.py`
- `results.tsv`, by appending one experiment row at a time

You may read:

- `README.md`
- `candidate.py`
- `evaluate.py`
- The `train` and `validation` data in `data/cases.json`
- `results.tsv`

Do not read, modify, or run:

- `data/test_cases.json`
- `test_model.py`

The holdout test is not part of the research loop.

Never modify:

- `evaluate.py`
- `data/cases.json`
- `.gitignore`

Never delete or rewrite earlier rows in `results.tsv`.

Do not install new dependencies.

## Experiment loop

Run no more than five experiments.

For each experiment:

1. Inspect the current `candidate.py`.
2. Review previous experiments in `results.tsv`.
3. Form one clear hypothesis.
4. Modify only `candidate.py`.
5. Run:

```bash
python evaluate.py > run.log 2>&1
```

6. Read the accuracy from `run.log`.
7. Record the experiment in `results.tsv`.
8. Keep or discard the change.

Use this tab-separated results format:

```text
experiment	accuracy	status	description
```

Allowed statuses:

- `keep`
- `discard`
- `crash`

## Decision rules

Keep a change when:

- Accuracy improves, or
- Accuracy remains 100% and the implementation becomes meaningfully simpler or easier to interpret.

Discard a change when:

- Accuracy decreases.
- It adds complexity without measurable benefit.
- It targets individual validation sentences instead of a general pattern.

If keeping a change:

```bash
git add candidate.py
git commit -m "Experiment: short description"
```

If discarding an uncommitted change:

```bash
git restore candidate.py
```

Do not remove or rewrite previous rows in `results.tsv`.

## Research principles

- Change one main idea per experiment.
- State the hypothesis before changing code.
- Prefer simple, explainable models.
- Never modify the evaluator to improve the score.
- Never copy validation examples into training data.
- A crash is a valid experimental result; record it.
- Stop after five experiments and summarize what was learned.
