"""Frozen evaluation protocol (M4): the final configuration per task, certainty levels from
out-of-fold probabilities, the expected test range from CV repeats with fresh seeds, and a
clean-tree rerun of the log rows the protocol cites.

Everything here uses ``split.load_dev()`` only; the held-out test partition is not read.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from . import config, experiments, split

FRESH_SEEDS = (100, 101, 102)  # never used for selection (PROJECT_PLAN.md 1.1)
FRESH_COLUMNS = tuple(f"cv_fold_s{s}" for s in FRESH_SEEDS)
REFERENCE = "logreg_word11"  # the simple reference model of PROJECT_PLAN.md 1.1
BASELINES = ("dummy_most_frequent", "dummy_stratified", REFERENCE)
# Certainty levels: a score band is "high" when every populated band above it is right at least
# 90% of the time out of fold, "medium" at least 70%; anything lower is "low".
LEVEL_TARGETS = (("high", 0.90), ("medium", 0.70))
BIN_WIDTH = 0.05
MIN_BIN = 30  # bands with fewer out-of-fold rows are too noisy to veto a threshold
EDGE_EPS = 1e-9  # binning and level assignment use the same tolerance at band edges
BOOTSTRAP = dict(
    resamples=2000, seed=0, level=0.95, interval="percentile",
    unit="group_id: draw as many near-duplicate groups as the test partition has, with replacement",
    metric="Macro-F1 and per-class F1 with the fixed label list and zero_division=0, as in the CV",
    paired="final minus reference and final minus each dummy, on the same resamples: interval of the "
           "difference and share of resamples with a difference <= 0",
    decision="final beats a dummy when the lower bound of the paired difference is > 0; versus the reference "
             "the result is 'above' (lower bound > 0), 'not distinguishable' (interval contains 0) or "
             "'below' (upper bound < 0); success requires beating both dummies and not 'below' the reference",
)
PROTOCOL_JSON = config.REPORTS_DIR / "protocol.json"
PROTOCOL_MD = config.REPORTS_DIR / "final_protocol.md"


def final_specs() -> dict[str, dict]:
    """The configuration chosen for each task by the logged M3 and M4 decisions."""
    return {task: experiments.resolve_m4(task)["final"] for task in experiments.TASKS}


def reference_spec() -> dict:
    return next(s for s in experiments.REGISTRY if s["name"] == REFERENCE)


def with_fresh_folds(dev: pd.DataFrame) -> pd.DataFrame:
    """Add grouped, stratified 5-fold assignments for the fresh seeds, built exactly like the
    selection folds (joint label strata, near-duplicate groups kept whole)."""
    joint = (dev.aspect + "|" + dev.sentiment).reset_index(drop=True)
    groups = dev.group_id.reset_index(drop=True)
    out = dev.reset_index(drop=True).copy()
    for col, seed in zip(FRESH_COLUMNS, FRESH_SEEDS):
        out[col] = split._grouped_folds(joint, groups, seed)
    return out


# ---- certainty levels -----------------------------------------------------------------------

def band_table(score: np.ndarray, correct: np.ndarray) -> pd.DataFrame:
    """Out-of-fold accuracy per score band [lo, lo + BIN_WIDTH); the last band includes 1."""
    edges = np.round(np.arange(0, 1, BIN_WIDTH), 2)
    idx = np.minimum(((np.asarray(score) + EDGE_EPS) / BIN_WIDTH).astype(int), len(edges) - 1)
    correct = np.asarray(correct, dtype=float)
    rows = [dict(lo=lo, n=int((idx == i).sum()),
                 accuracy=float(correct[idx == i].mean()) if (idx == i).any() else np.nan)
            for i, lo in enumerate(edges)]
    return pd.DataFrame(rows)


def threshold(table: pd.DataFrame, target: float) -> float:
    """Smallest edge of a populated band (at least MIN_BIN rows) from which every populated band
    reaches ``target``; sparse bands neither start nor veto a level."""
    populated = table[table.n >= MIN_BIN]
    for lo in populated.lo:
        if (populated.loc[populated.lo >= lo, "accuracy"] >= target).all():
            return float(lo)
    return 1.0


def certainty_thresholds(score: np.ndarray, correct: np.ndarray) -> dict[str, float]:
    table = band_table(score, correct)
    return {level: threshold(table, target) for level, target in LEVEL_TARGETS}


def certainty_level(score: float, thresholds: dict[str, float]) -> str:
    if score + EDGE_EPS >= thresholds["high"]:
        return "high"
    return "medium" if score + EDGE_EPS >= thresholds["medium"] else "low"


def level_summary(score: np.ndarray, correct: np.ndarray, thresholds: dict[str, float]) -> list[dict]:
    levels = np.array([certainty_level(s, thresholds) for s in score])
    correct = np.asarray(correct, dtype=bool)
    return [dict(level=lv, n=int((levels == lv).sum()), share=float((levels == lv).mean()),
                 accuracy=float(correct[levels == lv].mean()) if (levels == lv).any() else float("nan"))
            for lv in ("high", "medium", "low")]


# ---- expected range -------------------------------------------------------------------------

def fold_metrics(oofs: list[pd.DataFrame], labels: list[str]) -> list[float]:
    """Macro-F1 of every (repeat, fold) from out-of-fold predictions, repeat-major."""
    return [f1_score(g.y, g.pred, average="macro", labels=labels, zero_division=0)
            for oof in oofs for _, g in oof.groupby("fold", sort=True)]


def joint_fold_accuracy(sent: list[pd.DataFrame], asp: list[pd.DataFrame]) -> list[float]:
    """Share of reviews with both labels right, per (repeat, fold)."""
    out = []
    for s, a in zip(sent, asp):
        both = s.merge(a, on=["rid", "fold"], suffixes=("_s", "_a"))
        both["ok"] = (both.y_s == both.pred_s) & (both.y_a == both.pred_a)
        out += [float(g.ok.mean()) for _, g in both.groupby("fold", sort=True)]
    return out


def expected_range(values: list[float]) -> dict:
    mean, sd = float(np.mean(values)), float(np.std(values, ddof=1))
    return dict(mean=mean, sd=sd, low=max(0.0, mean - 2 * sd), high=min(1.0, mean + 2 * sd), n=len(values))


# ---- clean-tree rerun -----------------------------------------------------------------------

def clean_rerun(dev: pd.DataFrame, finals: dict[str, dict]) -> list[dict]:
    """Re-run the cited configurations on the selection folds from a committed tree, log them as
    ``rerun_<name>`` and compare their fold scores with the originally logged rows."""
    sha = experiments._git_sha()
    if sha == "unknown" or sha.endswith("+dirty"):
        raise RuntimeError(f"clean rerun needs a committed tree (git state: {sha})")
    by_name = {s["name"]: s for s in experiments.REGISTRY}
    out = []
    for task in experiments.TASKS:
        specs = [by_name[n] for n in BASELINES] + [experiments.resolve_axes(task)[1], finals[task]]
        for spec in specs:
            name = f"rerun_{spec['name']}"
            if experiments.exp_id(task, name) not in set(experiments.read_log().exp_id):
                experiments.run_experiment(task, dict(spec, name=name, notes=f"clean-tree rerun of {spec['name']}"), dev)
            log = experiments.read_log()
            row = log.loc[log.exp_id == experiments.exp_id(task, name)].iloc[0]
            orig = log.loc[log.exp_id == experiments.exp_id(task, spec["name"])].iloc[0]
            identity = experiments.row_identity(spec)
            if row.git_sha.endswith("+dirty") or row.git_sha == "unknown" or row.model != spec["model"] \
                    or any(row[k] != v or orig[k] != v for k, v in identity.items()):
                raise ValueError(f"logged rerun {task}/{name} is not a clean run of {spec['name']}")
            diff = np.abs(np.array(json.loads(row.fold_scores)) - np.array(json.loads(orig.fold_scores))).max()
            out.append(dict(task=task, name=spec["name"], original_sha=orig.git_sha, rerun_sha=row.git_sha,
                            original_mean=float(orig.cv_macro_f1_mean), rerun_mean=float(row.cv_macro_f1_mean),
                            max_fold_diff=float(diff)))
    return out


# ---- protocol -------------------------------------------------------------------------------

def _config(spec: dict) -> dict:
    return {k: v for k, v in spec.items() if k != "notes"}


def pipeline_params(spec: dict) -> dict:
    """Every scalar parameter of the built pipeline, library defaults included."""
    from .features import build_pipeline

    def plain(v) -> bool:
        if isinstance(v, (list, tuple)):
            return all(plain(x) for x in v)
        if isinstance(v, dict):
            return all(plain(x) for x in v.values())
        return v is None or isinstance(v, (bool, int, float, str))

    params = build_pipeline(spec).get_params(deep=True)
    return {k: v for k, v in sorted(params.items()) if plain(v)}


def library_versions() -> dict:
    import sklearn

    return {"scikit-learn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__}


# What M5 must reproduce exactly before it opens the test partition.
FROZEN_KEYS = ("dev_sha256", "split_sha256", "labels", "level_rule", "bootstrap", "fresh_seeds", "versions")


def fingerprint(p: dict) -> str:
    """SHA-256 of the frozen choices: data hashes, label lists, configurations with their full
    pipeline parameters, baselines, certainty thresholds and rule, bootstrap plan, versions."""
    frozen = {k: p[k] for k in FROZEN_KEYS}
    frozen["tasks"] = {t: dict(final=v["final"], reference=v["reference"], baselines=v["baselines"],
                               pipelines=v["pipelines"], thresholds=v["certainty"]["thresholds"])
                       for t, v in p["tasks"].items()}
    blob = json.dumps(frozen, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _selection_summary(r: dict) -> dict:
    keep = ("mean_delta", "wins", "n", "accepted")
    return dict(families=[dict(candidate=f["candidate"], **{k: f[k] for k in keep}) for f in r["families"]],
                family=r["family"], param=r["param"], one_se=r["one_se"], picked=r["picked"],
                gate={k: r["gate"][k] for k in keep} if r["gate"] else None,
                tfidf_steps=[dict(candidate=s["candidate"], **{k: s[k] for k in keep}) for s in r["tfidf_steps"]],
                final_vs_m3={k: r["final_vs_base"][k] for k in keep}, m3_final=r["base"]["name"])


def build_protocol(dev: pd.DataFrame | None = None, rerun: bool = True) -> dict:
    dev = split.load_dev() if dev is None else dev
    m4 = {task: experiments.resolve_m4(task) for task in experiments.TASKS}
    finals = {task: r["final"] for task, r in m4.items()}
    reruns = clean_rerun(dev, finals) if rerun else []
    fresh = with_fresh_folds(dev)
    by_name = {s["name"]: s for s in experiments.REGISTRY}
    baselines = [by_name[n] for n in BASELINES if n != REFERENCE]
    tasks, fresh_oof = {}, {}
    for task, spec in finals.items():
        labels = experiments.LABELS[task]
        selection = [experiments.oof_predictions(spec, dev, task, r) for r in range(len(split.CV_COLUMNS))]
        pooled_sel = pd.concat(selection)
        ok = (pooled_sel.y == pooled_sel.pred).to_numpy()
        thresholds = certainty_thresholds(pooled_sel.score.to_numpy(), ok)
        per_repeat = [certainty_thresholds(s.score.to_numpy(), (s.y == s.pred).to_numpy()) for s in selection]
        fresh_oof[task] = [experiments.oof_predictions(spec, fresh, task, column=c) for c in FRESH_COLUMNS]
        ref_oof = [experiments.oof_predictions(reference_spec(), fresh, task, column=c) for c in FRESH_COLUMNS]
        pooled = pd.concat(fresh_oof[task])
        tasks[task] = dict(
            final=_config(spec), reference=_config(reference_spec()), baselines=[_config(b) for b in baselines],
            pipelines={role: pipeline_params(s) for role, s in
                       [("final", spec), ("reference", reference_spec())] + [(b["name"], b) for b in baselines]},
            selection=_selection_summary(m4[task]),
            certainty=dict(score="highest class probability of the logistic-regression pipeline",
                           thresholds=thresholds, per_repeat_thresholds=per_repeat,
                           selection_levels=level_summary(pooled_sel.score.to_numpy(), ok, thresholds),
                           fresh_levels=level_summary(pooled.score.to_numpy(),
                                                      (pooled.y == pooled.pred).to_numpy(), thresholds)),
            expected_range=dict(final=expected_range(fold_metrics(fresh_oof[task], labels)),
                                reference=expected_range(fold_metrics(ref_oof, labels))),
        )
    assignment = pd.read_csv(config.SPLIT_FILE, usecols=["partition"])
    fold_rows = dev.groupby(split.CV_COLUMNS[0]).size()
    p = dict(
        version=2, built_from_commit=experiments._git_sha(),
        dev_sha256=hashlib.sha256(config.DEV_FILE.read_bytes()).hexdigest(),
        split_sha256=hashlib.sha256(config.SPLIT_FILE.read_bytes()).hexdigest(),
        dev_rows=len(dev), test_rows=int((assignment.partition == "test").sum()),
        fold_rows=float(fold_rows.mean()), train_fold_rows=float(len(dev) - fold_rows.mean()),
        labels=experiments.LABELS, fresh_seeds=list(FRESH_SEEDS),
        level_rule=dict(targets=dict(LEVEL_TARGETS), bin_width=BIN_WIDTH, min_bin=MIN_BIN, edge_eps=EDGE_EPS),
        bootstrap=BOOTSTRAP, versions=library_versions(), tasks=tasks,
        joint_expected_range=expected_range(joint_fold_accuracy(fresh_oof["sentiment"], fresh_oof["aspect"])),
        clean_rerun=reruns,
    )
    p["fingerprint"] = fingerprint(p)
    return p


def _rounded(obj, digits: int = 6):
    if isinstance(obj, float):
        return round(obj, digits)
    if isinstance(obj, dict):
        return {k: _rounded(v, digits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rounded(v, digits) for v in obj]
    return obj


def len_ratio(p: dict) -> float:
    """How many more rows the final model sees than a CV training fold."""
    return p["dev_rows"] / p["train_fold_rows"] - 1


def _range_cell(r: dict) -> str:
    return f"{r['mean']:.4f} | {r['sd']:.4f} | [{r['low']:.3f}, {r['high']:.3f}]"


def protocol_markdown(p: dict) -> str:
    log = experiments.read_log()
    lines = [
        "# Final evaluation protocol (frozen before the held-out test is read)", "",
        "This file and `reports/protocol.json` fix every choice that the one-shot test in M5 depends on. "
        "All numbers come from the development partition only; the held-out test partition has not been "
        "read by any step up to here (experiment code can only call `split.load_dev()`).", "",
        f"- Protocol fingerprint (configurations + certainty thresholds): `{p['fingerprint']}`",
        f"- Development partition: {p['dev_rows']} reviews, `data/processed/dev.csv` SHA-256 `{p['dev_sha256']}`",
        f"- Split file `data/splits/split_v1.csv` SHA-256 `{p['split_sha256']}`", "",
        "## 1. Frozen configurations", "",
        "| task | role | configuration | model | features | normalisation | class weight | parameters "
        "| selection CV Macro-F1 (3 x 5) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for task, t in p["tasks"].items():
        for role in ("final", "reference"):
            s = t[role]
            row = log.loc[log.exp_id == experiments.exp_id(task, s["name"])].iloc[0]
            lines.append(f"| {task} | {role} | `{s['name']}` | {s['model']} | {s['representation']} TF-IDF | "
                         f"{s['normalisation']} | {s['class_weight']} | `{json.dumps(s['params'])}` | "
                         f"{row.cv_macro_f1_mean:.4f} ± {row.cv_macro_f1_std:.4f} |")
    lines += [
        "", "`word11` is word unigrams, `char25` is character 2-5 grams inside word boundaries, `+` joins both in "
        "one feature union. Every pipeline parameter, library defaults included, is listed under `pipelines` in "
        "`reports/protocol.json` and is part of the fingerprint. The dummy baselines (`most_frequent`, "
        "`stratified`) are fitted and reported next to the models in M5.", "",
        "How the configurations were chosen (all on the same 3 x 5 grouped CV; rule: mean paired Macro-F1 gain "
        f">= {experiments.MIN_DELTA} and a gain in >= {experiments.MIN_WINS} of 15 folds):", "",
        "- Representation and preprocessing: `reports/ablations.md` (M3 final configuration per task: "
        + ", ".join(f"{t} `{v['selection']['m3_final']}`" for t, v in p["tasks"].items()) + ").",
    ]
    for task, t in p["tasks"].items():
        s = t["selection"]
        fams = "; ".join(f"{f['candidate']} {f['mean_delta']:+.4f}, {f['wins']}/{f['n']} "
                         f"({'accepted' if f['accepted'] else 'rejected'})" for f in s["families"])
        gate = ("not needed (not weaker-regularised than the default)" if s["gate"] is None else
                f"{s['gate']['mean_delta']:+.4f}, {s['gate']['wins']}/{s['gate']['n']} folds, "
                f"{'passed' if s['gate']['accepted'] else 'failed'}")
        f = s["final_vs_m3"]
        lines.append(
            f"- {task}, model family and regularisation (`reports/models.md`): families {fams}; family kept: "
            f"{s['family']}. One-standard-error pick on the {s['param']} grid: {s['one_se']}; acceptance check "
            f"against the default: {gate}; value used: {s['param']} = {s['picked']}. Final vs the M3 configuration: "
            f"{f['mean_delta']:+.4f} mean, {f['wins']}/{f['n']} folds, i.e. within the fold noise: the value comes "
            "from the procedure fixed before the grid was run, not from being better.")
    lines += [
        "- The optional pre-trained sentence-vector experiment was not run: its pre-declared condition was not "
        "met (`reports/error_analysis.md`, per-review categories in `reports/error_categories.csv`).", "",
        "## 2. Certainty levels", "",
        "Score: the highest class probability of the task's logistic-regression pipeline. Thresholds come from "
        "out-of-fold predictions of the frozen configuration on the three selection repeats (seeds 42-44). "
        f"Scores are cut into bands of width {BIN_WIDTH}; a level starts at the smallest band edge from which "
        f"every band with at least {MIN_BIN} out-of-fold rows is right at least as often as its target "
        "(high: 90%, medium: 70%); lower scores are low. The interface shows the level in words, never the raw "
        "probability.", "",
        "| task | high from | medium from | level | selection OOF rows | share | accuracy "
        "| fresh-seed OOF share | accuracy |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for task, t in p["tasks"].items():
        th = t["certainty"]["thresholds"]
        for sel, fr in zip(t["certainty"]["selection_levels"], t["certainty"]["fresh_levels"]):
            lines.append(f"| {task} | {th['high']:.2f} | {th['medium']:.2f} | {sel['level']} | {sel['n']} | "
                         f"{sel['share']:.3f} | {sel['accuracy']:.3f} | {fr['share']:.3f} | {fr['accuracy']:.3f} |")
    spread = "; ".join(
        f"{task}: " + ", ".join(f"{th['high']:.2f} / {th['medium']:.2f}" for th in t["certainty"]["per_repeat_thresholds"])
        for task, t in p["tasks"].items())
    lines += [
        "", f"Band counts pool the three repeats, so every review appears three times and a band of {MIN_BIN} "
        f"rows holds about {MIN_BIN // 3} distinct reviews. Thresholds (high / medium) computed on each repeat "
        f"alone: {spread}.", "",
        "Caveats: the final models are trained on the whole development partition (about "
        f"{len_ratio(p):.0%} more rows than a CV training fold), so their probabilities may be somewhat higher and "
        "the share of high-certainty answers larger. The real accuracy per level is measured on the test "
        "partition in M5 and reported as found. The fresh-seed columns use folds that played no role in setting "
        "the thresholds.", "",
        "## 3. Expected test range", "",
        "Computed after the configurations were frozen, from CV repeats with fresh seeds "
        f"{', '.join(map(str, p['fresh_seeds']))} (grouped and stratified like the selection folds, never used "
        "for any choice): mean ± 2 standard deviations of the 15 fold scores.", "",
        "| task | configuration | mean Macro-F1 | sd | expected range |", "|---|---|---|---|---|",
    ]
    for task, t in p["tasks"].items():
        for role in ("final", "reference"):
            lines.append(f"| {task} | {role} `{t[role]['name']}` | {_range_cell(t['expected_range'][role])} |")
    j = p["joint_expected_range"]
    lines += [
        "", f"Joint accuracy (both labels right, final configurations): mean {j['mean']:.4f}, sd {j['sd']:.4f}, "
        f"expected range [{j['low']:.3f}, {j['high']:.3f}].", "",
        f"A CV fold holds about {p['fold_rows']:.0f} reviews and is predicted by a model trained on about "
        f"{p['train_fold_rows']:,.0f}; the test partition holds {p['test_rows']} and is predicted by a model trained "
        f"on all {p['dev_rows']:,}, so the test score may sit slightly above the CV mean. The sampling part of the "
        f"fold sd shrinks by about sqrt({p['fold_rows']:.0f}/{p['test_rows']}) = "
        f"{(p['fold_rows'] / p['test_rows']) ** 0.5:.2f} on the larger test partition, so the range is slightly "
        "conservative. It is recorded for comparison only: it is not a pass mark, and no result outside it leads "
        "to another test run.", "",
        "## 4. Clean-tree rerun of the cited rows", "",
        "The configurations cited here were run again on the selection folds from a committed tree and compared "
        "fold by fold with the logged rows they were selected from.", "",
        "| task | configuration | original commit | rerun commit | original mean | rerun mean "
        "| largest fold difference |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in p["clean_rerun"]:
        lines.append(f"| {r['task']} | `{r['name']}` | {r['original_sha']} | {r['rerun_sha']} | "
                     f"{r['original_mean']:.4f} | {r['rerun_mean']:.4f} | {r['max_fold_diff']:.1e} |")
    b = p["bootstrap"]
    lines += [
        "", "## 5. One-shot test procedure (M5)", "",
        "1. Before it opens the test partition, `python -m arabic_reviews final` rebuilds the protocol from code "
        "and the development partition (configurations, pipeline parameters, certainty thresholds from the "
        "out-of-fold predictions, data hashes, versions; no rerun rows) and stops unless the fingerprint equals "
        "the one above.",
        "2. For each task it fits the frozen configuration, the reference model and both dummy baselines on the "
        "whole development partition.",
        "3. It reads the held-out test partition once, predicts it once per model, and records the access "
        "(time, commit, fingerprint, file hash) in `reports/test_access_log.json`. A second run is refused "
        "unless an explicit flag and a written methodological reason are given; the first result stays the "
        "reported one.",
        "4. Reported per task: Macro-F1 (primary), per-class precision / recall / F1, balanced accuracy, "
        "accuracy and the confusion matrix; joint accuracy over both tasks; accuracy and share per certainty "
        "level; the position of the test Macro-F1 relative to the expected range above.",
        f"5. Uncertainty: {b['level']:.0%} {b['interval']} bootstrap intervals with {b['resamples']} resamples "
        f"(seed {b['seed']}). Resampling unit: {b['unit']} (rows of a group are not independent). Metrics: "
        f"{b['metric']}. Paired comparisons: {b['paired']}. Decision: {b['decision']}. No claim is made from very "
        "small cells (for example prices x neutral).",
        "6. Errors on the test partition are described, but no model, threshold or feature is changed after the "
        "test has been read.", "",
    ]
    return "\n".join(lines)


def write_protocol(p: dict) -> None:
    PROTOCOL_JSON.write_text(json.dumps(_rounded(p), indent=1, ensure_ascii=False, sort_keys=True) + "\n",
                             encoding="utf-8", newline="\n")
    PROTOCOL_MD.write_text(protocol_markdown(p), encoding="utf-8", newline="\n")
