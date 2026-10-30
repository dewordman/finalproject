# README_RUN: running the email threat classification pipeline (Revision 2)

All commands are PowerShell, run from the project root. Every run writes into its own results
folder; nothing here deletes or overwrites earlier results or locks.

## 1. Environment

```powershell
py -3.14 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q
```

The `pytest` run must finish with no failures before you touch real data.

## 2. Data

Place the raw data as follows (see `docs/CHANGES_REV2.md` for the sources):

- `data\raw\enron_spam\*.csv`, `data\raw\spamassassin\` (CSV or folders)
- `data\raw\nazario\` with `phishing0.mbox`, `phishing1.mbox`, `phishing2.mbox`, `phishing3.mbox`, `20051114.mbox`
- `data\raw\nazario_external\phishing-2015.mbox` ... `phishing-2025.mbox` (files from the site are saved WITHOUT an extension; rename them with `.mbox`)
- No CSV may remain in `data\raw\nazario`.

## 3. Declare your decisions BEFORE running anything that selects a model

Copy `config_rev2.yaml` to a new file per experiment (never edit the tests' template, `tests\config_template.yaml`), and write down in `docs\decisions_log.md`:

- `selection.min_phishing_recall` and `selection.max_legit_fpr` (they are `null` on purpose; the code stops until you set them);
- the near-duplicate threshold, after reviewing `near_dup_review_sample`;
- keep `strip_urls` and `strip_email_addresses` **false** (Enron-Spam is tokenised, so stripping would create a source cue; see `docs\CHANGES_REV2.md`);
- `configs\decision_table.yaml` thresholds, justified from the validation reliability table.

## 4. Development stages (these never touch the sealed test partition)

```powershell
$cfg = "config_rev2.yaml"; $out = "results_rev2"
python scripts/run_pipeline.py --config $cfg --results-dir $out --stage manifest
python scripts/run_pipeline.py --config $cfg --results-dir $out --stage parse
python scripts/run_pipeline.py --config $cfg --results-dir $out --stage diagnose
python scripts/run_pipeline.py --config $cfg --results-dir $out --stage dedup
python scripts/run_pipeline.py --config $cfg --results-dir $out --stage split
python scripts/run_pipeline.py --config $cfg --results-dir $out --stage grid
python scripts/run_pipeline.py --config $cfg --results-dir $out --stage select
python scripts/run_pipeline.py --config $cfg --results-dir $out --stage audit
python scripts/suggest_mask_terms.py --config $cfg --results-dir $out
```

`suggest_mask_terms.py` only prints candidates and writes `mask_term_candidates.csv`; it never edits
a config. Choose terms by hand, put them in `text.mask_terms` of a NEW config file, and rerun
`parse` through `audit` into a NEW results folder. Because `mask_terms` changes the model text, all
stage outputs must be regenerated; the runner refuses to reuse a results folder with a changed config.

```powershell
python scripts/run_pipeline.py --config $cfg --results-dir $out --stage calibrate
python scripts/run_pipeline.py --config $cfg --results-dir $out --stage bundle
python scripts/run_pipeline.py --config $cfg --results-dir $out --stage dev_eval
python scripts/run_pipeline.py --config $cfg --results-dir $out --stage source_tests
python scripts/run_pipeline.py --config $cfg --results-dir $out --stage variations
python scripts/run_pipeline.py --config $cfg --results-dir $out --stage error_review
python scripts/run_pipeline.py --config $cfg --results-dir $out --stage external
```

`dev_eval`, `source_tests`, `variations` and `error_review` use training and validation rows only.
Compare preprocessing settings with `dev_eval`, never with `final_test`. You can run several
configurations with `scripts\run_experiments.ps1` (it stops at `dev_eval`).

## 5. Once-only evaluations (locked)

Freeze one configuration first. Then:

```powershell
python scripts/run_pipeline.py --config $cfg --results-dir $out --stage final_test --force --reason "revision 2: <why>"
python scripts/run_pipeline.py --config $cfg --results-dir $out --stage temporal
```

- The final-test lock lives in `locks\` (outside every results folder). If a lock already exists from an
  earlier revision, `--force --reason "..."` is required; it appends a line to `locks\revision_log.json`
  (append-only) and keeps the old lock in its history. Without an existing lock, omit `--force`.
- `temporal` has its own lock (`locks\nazario_temporal_lock.json`) and is run once. It scores the saved
  models on the external Nazario files and never refits.
- A forced rerun needs a FRESH results folder; the runner refuses to overwrite earlier outputs.

## 5b. Software tests (Chapter Three, T01 to T10)

Automated checks against the REAL final bundle (run after `bundle` and `source_tests`):

```powershell
python scripts/software_tests.py --results-dir results_rev5 --table configs/decision_table.yaml --log logs/predictions.csv
```

- `--table` must be the decision table the app actually uses (the app shows its version and thresholds at the top).
- The script writes `software_test_results_<time>.csv` into the results folder: one row per test with the
  expected result, the actual result and PASS, FAIL or SKIPPED. It never overwrites an earlier evidence file.
- `T01-explain` rows FAIL if a prediction returns no influential terms. That is deliberate: the thesis claims
  interpretable decision support, so an empty explanation panel must not pass.
- `LOG` is SKIPPED until the app has been used once (`logs\predictions.csv`); then rerun with `--log`.
- Also run the app by hand for the screenshots: `streamlit run app/streamlit_app.py`.

## 6. Finish

```powershell
python scripts/run_pipeline.py --config $cfg --results-dir $out --stage index
```

`index` writes `RESULTS_INDEX.md`, mapping each output to the Chapter Five section it feeds.
Code the `error_review_sheet.csv` by hand (the coding columns start empty).

## 7. What to do after a failure

1. Read the last lines of the run log in `<results>\run_logs\`.
2. Fix the cause, then rerun the SAME stage into a NEW results folder. Do not delete anything.
3. If a locked stage failed, its reservation stays in place (status `failed`); rerunning needs `--force --reason`.
4. Record what happened in `docs\decisions_log.md`.
