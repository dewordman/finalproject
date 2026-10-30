# Revision 2 change log (for Chapter Four's revision record)

Plain-English summary of what changed after the first full run, why, and which Chapter Three
requirement each change protects. Chapter Three remains the design contract.

## Why a revision was needed

The first run produced very high pooled scores, but the audit showed that the model leaned on
source and era cues rather than on threat content: personal and organisation names, the host name
of the phishing corpus, four-digit years, and raw e-mail header text. The held-out-source tests
failed badly, and the sealed test set had been scored more than once because a helper script deleted
the lock together with the results folder. Revision 2 fixes the protocol first, then the data.

## Protocol changes (protect "the final test is scored once", Chapter Three 3.10.1)

- **Global lock.** The final-test lock now lives in `locks/`, outside every results folder, and no
  script can delete it. A rerun needs `--force --reason "..."`, which appends one line to
  `locks/revision_log.json` (append-only) and keeps the previous lock in its history.
- **`dev_eval` stage.** Preprocessing choices are compared on the validation partition only, so the
  test partition is never used to choose a configuration. The sealed pickle is never opened by it.
- **Per-run folders.** Every run uses its own results, models and processed-data folders; a changed
  configuration cannot reuse an old namespace, and existing outputs are never overwritten.
- **Experiments stay on development data.** Source-held-out tests, text variations and the error
  review sheet now use training and validation rows only, so the test partition stays sealed.

## Data changes (protect source provenance, Chapter Three 3.5)

- **Original Nazario corpus** (CC BY 4.0, https://monkey.org/~jose/phishing/): the 2005 to 2007 files form
  the main phishing set; the 2015 to 2025 yearly files are an external temporal set that never enters
  training, validation, selection or the pooled test. `private-phishing4.mbox` is excluded because its
  licence status is unclear (reason code `licence_restriction`). The old Kaggle `Nazario_5` file is
  rejected if it is left in the folder.
- **Era fields.** Every record carries `era_year` and `era_group`, so source and era can be examined.
- **Manifest.** `manifest` writes a SHA-256 for every raw file with the source URL and licence.

## Text preparation changes (declared deviation from "light normalisation", Chapter Three 3.5.4)

The audit justified a small, declared set of clean-up steps, all switchable in `config`:

- flagged source terms are **deleted** (default) instead of replaced by a token, and a single-word term
  also removes the dotted host it belongs to, so no fragment such as `org` survives;
- dates, clock times, zone offsets and four-digit years 1990 to 2039 are removed (Enron's spaced numeric dates, such as `06 / 21 / 2000`, lose their year but keep the day and month numbers);
- URL and e-mail removal exist as options (`strip_urls`, `strip_email_addresses`) but are **off** by default. An early diagnostic suggested that Enron-Spam contains no links; that was wrong. Enron-Spam is distributed tokenised with spaces (for example `http : / / datesaturday . com / d / 1 . php` and `name @ host . com`), so the natural-text patterns never match it. Removing links and addresses from the other sources only, while the Enron remnants (`http`, `www`, `com`, `php`) stay, turns those words into an Enron cue. The audit of the first run that used these options showed exactly that. Chapter Three's rule (retain URL text) is therefore kept; the diagnostic table reports word-level `http`, `www` and `@` shares that work on every source;
- a leading raw header block and `X-Spam-` style lines are removed, and HTML or quoted-printable debris
  is decoded.

The same options are stored in the bundle, so training, experiments and prediction build identical text
(a parity test checks this). Choose `mask_terms` by hand after the audit; `suggest_mask_terms.py` never edits a config.

## New and changed experiments

- `diagnose`: source by era table (years, HTML, header remnants, URLs, e-mail addresses) before any model is fitted.
- `source_tests`: Enron and SpamAssassin hold-outs on development data, with the source-only baseline.
- `temporal`: replaces the old within-source Nazario test. Saved models are scored once, without refitting,
  on the external later-era files; legitimate and spam are absent, so no three-class macro score is computed.
- `variations`: sampled per class from validation; every variation reports whether it is null by design.
- `error_review`: one row per duplicate group from validation errors; coding columns start empty and no bodies are written.

## Housekeeping

`src/__init__.py` renamed; decision table de-duplicated; package versions and a `pip freeze` are written
to each results folder; the README lists the exact order of commands.

## What this does not fix

Phishing still comes from one inbox. The 2005 to 2007 files are only partly era-matched to Enron and
SpamAssassin, there is no modern legitimate source, and the pooled test partition was examined in earlier
runs, so the external temporal set is the cleaner check on unseen phishing.
