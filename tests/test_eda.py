"""EDA reads only the development partition and is reproducible."""

import pandas as pd
import pytest

from arabic_reviews import config, eda

pytestmark = pytest.mark.skipif(not config.DEV_FILE.exists(), reason="split not present")


def test_eda_never_reads_the_test_partition(tmp_path, monkeypatch):
    opened = []
    real_read_csv = pd.read_csv

    def spy(path, *args, **kwargs):
        opened.append(str(path))
        return real_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", spy)
    monkeypatch.setattr(config, "EDA_REPORT", tmp_path / "eda.md")
    eda.run_eda(figures=False)
    assert str(config.TEST_FILE) not in opened
    assert str(config.DEV_FILE) in opened


def test_eda_report_is_reproducible_and_matches_the_committed_one(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EDA_REPORT", tmp_path / "eda.md")
    first = eda.run_eda(figures=False)
    assert first == eda.run_eda(figures=False)
    committed = (config.ROOT / "reports" / "eda.md").read_text(encoding="utf-8")
    assert first == committed.split("\n## Figures")[0].rstrip("\n") + "\n"


def test_eda_counts_cover_exactly_the_dev_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EDA_REPORT", tmp_path / "eda.md")
    split = pd.read_csv(config.SPLIT_FILE)
    n_dev = int((split.partition == "dev").sum())
    report = eda.run_eda(figures=False)
    assert f"Rows: **{n_dev}**" in report
    totals_row = next(line for line in report.splitlines() if line.startswith("| total | "))
    assert totals_row.endswith(f"| {n_dev} |")
