# tests/test_external_year.py
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.parse import _year_from_filename, parse_nazario_external


@pytest.mark.parametrize("name,year", [("phishing-2023.mbox", 2023), ("phishing-2015.mbox", 2015),
                                       ("phishing0.mbox", -1), ("20051114.mbox", -1), ("x-2099.mbox", -1)])
def test_year_from_filename(name, year):
    assert _year_from_filename(name) == year


def test_external_message_without_usable_date_gets_the_file_year(tmp_path):
    cfg = yaml.safe_load((Path(__file__).resolve().parent / "config_template.yaml").read_text(encoding="utf-8"))
    external, main = tmp_path / "external", tmp_path / "main"
    external.mkdir()
    main.mkdir()
    cfg["nazario"].update(main_dir=str(main), external_dir=str(external))
    cfg["language"]["min_chars_for_detection"] = 10 ** 6
    cases = [("Dated", "Sat, 17 Mar 2018 10:00:00 +0000"), ("No date", None), ("Future", "Sat, 17 Mar 2038 10:00:00 +0000")]
    chunks = []
    for index, (subject, date) in enumerate(cases):
        header = f"Subject: {subject}\n" + (f"Date: {date}\n" if date else "")
        chunks.append(f"From x{index}@example.org Fri Mar 17 10:00:00 2023\n{header}Content-Type: text/plain\n\nverify your account password {index}\n\n")
    (external / "phishing-2023.mbox").write_text("".join(chunks), encoding="utf-8")
    frame = parse_nazario_external(cfg)
    years = dict(zip(frame["subject_raw"], frame["era_year"]))
    assert years == {"Dated": 2018, "No date": 2023, "Future": 2023}
    assert set(frame["era_group"]) == {"2015+"} and set(frame["partition_id"]) == {"external_temporal"}
