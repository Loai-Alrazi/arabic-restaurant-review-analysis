"""Split first: near-duplicate grouping, held-out test partition and CV folds.

Grouping is a label-free quality step on the whole validated dataset. Two
reviews belong to the same group when their duplicate keys are similar
(cosine >= ``SIM_THRESHOLD`` on character 3-5-gram TF-IDF) or one key is
contained in the other (truncated copies of the same review, at least
``CONTAIN_MIN_WORDS`` words). Groups are the transitive closure of these
pairs, so no group ever spans two partitions or two CV folds.

The split is ``StratifiedGroupKFold`` on the joint label ``aspect|sentiment``:
fold ``TEST_FOLD`` of the first pass is the held-out test partition; the rest
is the development partition, which gets ``len(CV_SEEDS)`` repeats of grouped,
stratified 5-fold CV. Experiment code reads data only through ``load_dev()``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold

from . import config
from .text import dedup_key

SIM_THRESHOLD = 0.80
CONTAIN_MIN_WORDS = 4
N_SPLITS = 5
TEST_FOLD = 0
SPLIT_SEED = 42
CV_SEEDS = (42, 43, 44)
CV_COLUMNS = tuple(f"cv_fold_r{i}" for i in range(len(CV_SEEDS)))
# Studied, not used: share of pairs per similarity band for the threshold report.
STUDY_BANDS = (0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)


def similarity_matrix(keys: list[str]) -> np.ndarray:
    """Cosine similarity of character 3-5-gram TF-IDF vectors (L2-normalised)."""
    vectors = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit_transform(keys)
    return (vectors @ vectors.T).toarray()


def contained_pairs(keys: list[str], min_words: int = CONTAIN_MIN_WORDS) -> set[tuple[int, int]]:
    """Index pairs (i < j) where a key of at least ``min_words`` words occurs inside another key."""
    pairs = set()
    for i, short in enumerate(keys):
        if len(short.split()) < min_words:
            continue
        for j, other in enumerate(keys):
            if i != j and len(other) >= len(short) and short in other:
                pairs.add((min(i, j), max(i, j)))
    return pairs


def near_duplicate_pairs(texts: list[str]) -> pd.DataFrame:
    """All near-duplicate pairs as positional indices with their cosine and containment flag."""
    keys = [dedup_key(t) for t in texts]
    sim = similarity_matrix(keys)
    upper = np.triu(sim >= SIM_THRESHOLD, k=1)
    cosine = set(zip(*map(np.ndarray.tolist, np.nonzero(upper))))
    contained = contained_pairs(keys)
    rows = [
        {"i": i, "j": j, "cosine": float(sim[i, j]), "contained": (i, j) in contained}
        for i, j in sorted(cosine | contained)
    ]
    return pd.DataFrame(rows, columns=["i", "j", "cosine", "contained"])


def union_find_groups(n: int, pairs: list[tuple[int, int]]) -> np.ndarray:
    """Component label per position; each label is the smallest position in its component."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return np.array([find(x) for x in range(n)])


def _grouped_folds(y: pd.Series, groups: pd.Series, seed: int) -> np.ndarray:
    folds = np.full(len(y), -1)
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for k, (_, held) in enumerate(cv.split(np.zeros(len(y)), y, groups)):
        folds[held] = k
    return folds


@dataclass
class SplitResult:
    assignment: pd.DataFrame  # rid, group_id, partition, cv_fold_r*
    pairs: pd.DataFrame  # rid_a, rid_b, cosine, contained
    similarity_bands: dict


def make_split(data: pd.DataFrame) -> SplitResult:
    """Group near-duplicates and assign every row to dev/test and to dev CV folds."""
    data = data.sort_values("rid").reset_index(drop=True)
    texts = data.text.tolist()
    pos_pairs = near_duplicate_pairs(texts)
    rids = data.rid.to_numpy()
    groups = pd.Series(rids[union_find_groups(len(data), list(zip(pos_pairs.i, pos_pairs.j)))])

    joint = data.aspect + "|" + data.sentiment
    outer = _grouped_folds(joint, groups, SPLIT_SEED)
    partition = np.where(outer == TEST_FOLD, "test", "dev")
    out = pd.DataFrame({"rid": rids, "group_id": groups, "partition": partition})
    dev = partition == "dev"
    for col, seed in zip(CV_COLUMNS, CV_SEEDS):
        folds = np.full(len(data), -1)
        folds[dev] = _grouped_folds(joint[dev].reset_index(drop=True), groups[dev].reset_index(drop=True), seed)
        out[col] = folds

    keys = [dedup_key(t) for t in texts]
    sims = similarity_matrix(keys)[np.triu_indices(len(data), 1)]
    bands = {f">={b:.2f}": int((sims >= b).sum()) for b in STUDY_BANDS}
    pairs = pd.DataFrame(
        {
            "rid_a": rids[pos_pairs.i.to_numpy(dtype=int)],
            "rid_b": rids[pos_pairs.j.to_numpy(dtype=int)],
            "cosine": pos_pairs.cosine.round(4),
            "contained": pos_pairs.contained,
        }
    )
    return SplitResult(out, pairs, bands)


def leakage_checks(data: pd.DataFrame, assignment: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    """Counts that must all be zero: anything shared across partitions or across CV folds."""
    df = data.merge(assignment, on="rid", validate="one_to_one")
    df["key"] = df.text.map(dedup_key)
    checks = {}
    by_rid = df.set_index("rid")
    for scope, frame, col in [("partition", df, "partition")] + [
        (c, df[df.partition == "dev"], c) for c in CV_COLUMNS
    ]:
        checks[f"{scope}: groups in >1 part"] = int((frame.groupby("group_id")[col].nunique() > 1).sum())
        checks[f"{scope}: keys in >1 part"] = int((frame.groupby("key")[col].nunique() > 1).sum())
        scoped = pairs[pairs.rid_a.isin(frame.rid) & pairs.rid_b.isin(frame.rid)]
        checks[f"{scope}: near-duplicate pairs across parts"] = int(
            (by_rid.loc[scoped.rid_a, col].to_numpy() != by_rid.loc[scoped.rid_b, col].to_numpy()).sum()
        )
    checks["test rows with a CV fold"] = int((df.loc[df.partition == "test", list(CV_COLUMNS)] != -1).any(axis=1).sum())
    checks["dev rows without a CV fold"] = int((df.loc[df.partition == "dev", list(CV_COLUMNS)] == -1).any(axis=1).sum())
    return checks


def proportions(data: pd.DataFrame, assignment: pd.DataFrame) -> pd.DataFrame:
    """Class shares (percent) of each label in dev and test, and their difference in points."""
    df = data.merge(assignment, on="rid")
    rows = []
    for task, labels in (("aspect", config.ASPECTS), ("sentiment", config.SENTIMENTS)):
        share = pd.crosstab(df[task], df.partition, normalize="columns") * 100
        for label in labels:
            rows.append({"task": task, "label": label, "dev_pct": share.loc[label, "dev"],
                         "test_pct": share.loc[label, "test"]})
    out = pd.DataFrame(rows)
    out["diff_pp"] = (out.dev_pct - out.test_pct).abs()
    return out


def load_dev() -> pd.DataFrame:
    """The development partition (text, labels, group_id, CV folds) — the only data
    experiment code may read. Every row is checked against the published split."""
    dev = pd.read_csv(config.DEV_FILE, dtype={"text": str}, keep_default_na=False)
    split = pd.read_csv(config.SPLIT_FILE)
    allowed = set(split.loc[split.partition == "dev", "rid"])
    if not set(dev.rid) <= allowed or len(dev) != len(allowed):
        raise RuntimeError("dev.csv does not match the dev partition of split_v1.csv; re-run the split step")
    return dev
