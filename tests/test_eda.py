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


def test_observation_claims_hold_on_the_dev_partition():
    dev = eda.load_dev()
    dev = dev.assign(words=dev.text.str.split().str.len())
    pos, neg, neu = config.SENTIMENTS
    sentiment_grams = eda.distinctive_ngrams(dev.text, dev.sentiment)
    assert {"بس", "جيد", "عادي"} <= set(sentiment_grams[neu])
    aspect_grams = {t for terms in eda.distinctive_ngrams(dev.text, dev.aspect).values() for t in terms}
    assert {f for forms in eda.ASPECT_FORM_EXAMPLES for f in forms} <= aspect_grams
    assert dev.set_index("rid").loc[eda.NEUTRAL_EXAMPLE_RID, "sentiment"] == neu
    for forms in (eda.NEGATIONS, eda.NEGATIONS - {"ما"}):
        rate = dev.text.map(lambda t: eda._has_negation(t, forms)).groupby(dev.sentiment).mean()
        assert rate[neg] > rate[pos] and rate[neu] > rate[pos]
    emoji = dev.text.str.contains(eda.EMOJI_RE).groupby(dev.sentiment).mean()
    assert emoji[pos] > emoji[neg] and emoji[pos] > emoji[neu]
    words = dev.groupby("sentiment").words.median()
    assert words[neg] > words[pos] and words[neu] > words[pos]
    m = eda.multi_aspect_estimate(dev)
    assert m["multi_cue_pct"] < 20 and m["multi_cue_label_among_cues_pct"] > 90


def test_emoji_pattern_covers_stars_and_dingbats():
    assert all(eda.EMOJI_RE.search(c) for c in ["⭐", "❤", "☕", "\U0001f60d"])
    assert not eda.EMOJI_RE.search("قهوة 10/10 !")
