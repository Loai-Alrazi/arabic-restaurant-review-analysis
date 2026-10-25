"""One-shot held-out evaluation (M5), run exactly as ``reports/final_protocol.md`` section 5 says.

Order: refuse a repeated run without a written reason; rebuild the protocol from code and the
development partition and stop unless its fingerprint equals the frozen one; fit every model on
the whole development partition; only then read the held-out test partition once (the access is
logged before the rows are parsed), predict it once per model and report. Nothing that is fitted
or chosen here depends on the test rows.
"""

from __future__ import annotations

import hashlib
import io
import json
import platform
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from . import config, experiments, protocol, split
from .features import build_pipeline

ACCESS_LOG = config.REPORTS_DIR / "test_access_log.json"
FINAL_JSON = config.REPORTS_DIR / "final_test.json"
FINAL_MD = config.REPORTS_DIR / "final_test.md"
MODELS_DIR = config.ROOT / "models"
SCRATCH_DIR = config.ROOT / ".local" / "scratch"  # per-row predictions with texts stay local
DUMMIES = ("dummy_most_frequent", "dummy_stratified")
ROLES = ("final", "reference", *DUMMIES)
SMALL_CELL = 20  # test cells with fewer rows are described, never used for a claim


class ProtocolMismatch(RuntimeError):
    """The code or data no longer reproduce the frozen protocol."""


class RepeatedTestRun(RuntimeError):
    """The test partition was already read once."""


# ---- guards ---------------------------------------------------------------------------------

def read_access_log() -> list[dict]:
    return json.loads(ACCESS_LOG.read_text(encoding="utf-8")) if ACCESS_LOG.exists() else []


def check_access(entries: list[dict], reason: str | None) -> None:
    if entries and not (reason and reason.strip()):
        raise RepeatedTestRun(f"the test partition was already read {len(entries)} time(s) "
                              f"({ACCESS_LOG.name}); a new run needs --rerun-reason with a methodological reason")


def check_fingerprint(rebuilt: dict, frozen: dict) -> None:
    if rebuilt["fingerprint"] != frozen["fingerprint"]:
        raise ProtocolMismatch(f"rebuilt protocol fingerprint {rebuilt['fingerprint']} differs from the frozen "
                               f"{frozen['fingerprint']}; the test partition was not opened")


def provenance() -> str:
    """Commit of a clean tree whose frozen protocol and access log are the committed ones."""
    sha = experiments._git_sha()
    files = [p.relative_to(config.ROOT).as_posix() for p in (protocol.PROTOCOL_JSON, ACCESS_LOG)]
    changed = experiments._git("status", "--porcelain", "--", *files).strip()
    if sha == "unknown" or sha.endswith("+dirty") or changed:
        raise RuntimeError(f"the final run needs a committed tree (git state: {sha}; {changed or 'reports ok'})")
    return sha


def read_test_once(entries: list[dict], reason: str | None, sha: str, fingerprint: str) -> tuple[pd.DataFrame, dict]:
    """Read the test file once, log the access before parsing it, and check it against the split."""
    raw = config.TEST_FILE.read_bytes()
    entry = dict(run=len(entries) + 1, time=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 commit=sha, fingerprint=fingerprint, test_sha256=hashlib.sha256(raw).hexdigest(),
                 test_rows=None, reason=reason)
    _write_json(ACCESS_LOG, entries + [entry])
    test = pd.read_csv(io.BytesIO(raw), dtype={"text": str}, keep_default_na=False)
    entry["test_rows"] = len(test)
    _write_json(ACCESS_LOG, entries + [entry])
    published = pd.read_csv(config.SPLIT_FILE)
    expected = published.loc[published.partition == "test", ["rid", "group_id"]].sort_values("rid").reset_index(drop=True)
    got = test[["rid", "group_id"]].sort_values("rid").reset_index(drop=True)
    if not test.rid.is_unique or not got.equals(expected):
        raise RuntimeError("the test file does not match the test partition of the published split")
    for task, column in experiments.TASKS.items():
        if not test[column].isin(experiments.LABELS[task]).all():
            raise RuntimeError(f"unknown {column} labels in the test file")
    return test, entry


# ---- metrics and bootstrap ------------------------------------------------------------------

def encode(values, labels: list[str]) -> np.ndarray:
    index = {label: i for i, label in enumerate(labels)}
    return np.array([index[v] for v in values], dtype=int)


def per_class_f1(y: np.ndarray, pred: np.ndarray, k: int) -> np.ndarray:
    """F1 per class index 0..k-1 = 2TP / (true + predicted); 0 when a class is neither (zero_division=0)."""
    tp = np.bincount(y[y == pred], minlength=k)
    denom = np.bincount(y, minlength=k) + np.bincount(pred, minlength=k)
    return np.divide(2 * tp, denom, out=np.zeros(k), where=denom > 0)


def resample_indices(groups: np.ndarray, resamples: int, seed: int) -> list[np.ndarray]:
    """Row indices of each resample: as many groups as there are, drawn with replacement."""
    _, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(inverse.max() + 1)]
    rng = np.random.default_rng(seed)
    return [np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
            for _ in range(resamples)]


def bootstrap(y, preds: dict[str, np.ndarray], labels: list[str], groups: np.ndarray, plan: dict) -> dict:
    """Percentile intervals of Macro-F1 and per-class F1, and paired differences final - other."""
    k, yi = len(labels), encode(y, labels)
    codes = {role: encode(p, labels) for role, p in preds.items()}
    idx = resample_indices(groups, plan["resamples"], plan["seed"])
    f1 = {role: np.array([per_class_f1(yi[i], c[i], k) for i in idx]) for role, c in codes.items()}
    macro = {role: v.mean(axis=1) for role, v in f1.items()}
    tail = (1 - plan["level"]) / 2 * 100

    def interval(v: np.ndarray) -> dict:
        low, high = np.percentile(v, [tail, 100 - tail])
        return dict(low=float(low), high=float(high))

    point = {role: float(per_class_f1(yi, c, k).mean()) for role, c in codes.items()}
    paired = {}
    for other in (r for r in preds if r != "final"):
        diff = macro["final"] - macro[other]
        paired[other] = dict(observed=point["final"] - point[other], **interval(diff),
                             share_le_0=float((diff <= 0).mean()))
    return dict(macro_f1={role: interval(m) for role, m in macro.items()},
                per_class_f1={role: {lab: interval(v[:, j]) for j, lab in enumerate(labels)} for role, v in f1.items()},
                paired=paired, resamples=len(idx))


def decide(paired: dict) -> dict:
    ref = paired["reference"]
    verdict = "above" if ref["low"] > 0 else "below" if ref["high"] < 0 else "not distinguishable"
    beats = {d: bool(paired[d]["low"] > 0) for d in DUMMIES}
    return dict(reference=verdict, beats_dummies=beats, success=all(beats.values()) and verdict != "below")


def position(value: float, rng: dict) -> str:
    return "below" if value < rng["low"] else "above" if value > rng["high"] else "inside"


def task_metrics(y, pred, labels: list[str]) -> dict:
    frame = pd.DataFrame(dict(y=y, pred=pred))
    return dict(macro_f1=float(f1_score(y, pred, average="macro", labels=labels, zero_division=0)),
                accuracy=float(accuracy_score(y, pred)), balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                per_class=experiments.per_class_table(frame, labels).to_dict("records"),
                confusion=experiments.confusion(frame, labels).to_numpy().tolist())


# ---- run ------------------------------------------------------------------------------------

def fit_models(dev: pd.DataFrame, t: dict, task: str) -> dict:
    specs = {"final": t["final"], "reference": t["reference"], **{b["name"]: b for b in t["baselines"]}}
    X, y = dev["text"].to_numpy(), dev[experiments.TASKS[task]].to_numpy()
    return {role: build_pipeline(specs[role]).fit(X, y) for role in ROLES}


def evaluate(test: pd.DataFrame, models: dict, p: dict) -> dict:
    X, out, final_ok = test["text"].to_numpy(), {}, {}
    for task, column in experiments.TASKS.items():
        labels, y, t = experiments.LABELS[task], test[column].to_numpy(), p["tasks"][task]
        preds = {role: models[task][role].predict(X) for role in ROLES}
        score = models[task]["final"].predict_proba(X).max(axis=1)
        final_ok[task] = preds["final"] == y
        boot = bootstrap(y, preds, labels, test["group_id"].to_numpy(), p["bootstrap"])
        metrics = {role: task_metrics(y, preds[role], labels) for role in ROLES}
        out[task] = dict(
            metrics=metrics, bootstrap=boot, decision=decide(boot["paired"]),
            levels=protocol.level_summary(score, final_ok[task], t["certainty"]["thresholds"]),
            expected_range=t["expected_range"],
            position={role: position(metrics[role]["macro_f1"], t["expected_range"][role])
                      for role in ("final", "reference")},
            predictions=pd.DataFrame(dict(rid=test["rid"].to_numpy(), y=y, pred=preds["final"], score=score)),
        )
    joint = float((final_ok["sentiment"] & final_ok["aspect"]).mean())
    cells = (test.assign(ok=final_ok["sentiment"] & final_ok["aspect"])
             .groupby(["aspect", "sentiment"]).ok.agg(["size", "mean"]).reset_index())
    return dict(tasks=out, joint=dict(accuracy=joint, expected_range=p["joint_expected_range"],
                                      position=position(joint, p["joint_expected_range"])),
                cells=cells.rename(columns={"size": "n", "mean": "joint_accuracy"}).to_dict("records"))


def output_paths(run: int) -> tuple:
    if run == 1:
        return FINAL_JSON, FINAL_MD
    return tuple(p.with_name(f"{p.stem}_run{run}{p.suffix}") for p in (FINAL_JSON, FINAL_MD))


def run_final(rerun_reason: str | None = None, check_only: bool = False) -> dict:
    entries = read_access_log()
    if not check_only:
        check_access(entries, rerun_reason)
    sha = provenance()
    frozen = json.loads(protocol.PROTOCOL_JSON.read_text(encoding="utf-8"))
    dev = split.load_dev()
    rebuilt = protocol.build_protocol(dev, rerun=False)
    check_fingerprint(rebuilt, frozen)
    models = {task: fit_models(dev, rebuilt["tasks"][task], task) for task in experiments.TASKS}
    if check_only:
        return dict(fingerprint=rebuilt["fingerprint"], commit=sha, test_read=False)
    test, entry = read_test_once(entries, rerun_reason, sha, rebuilt["fingerprint"])
    result = evaluate(test, models, rebuilt)
    result.update(access=entry, fingerprint=rebuilt["fingerprint"], dev_rows=len(dev), test_rows=len(test))
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    for task, t in result["tasks"].items():
        t.pop("predictions").merge(test[["rid", "text"]], on="rid").to_csv(
            SCRATCH_DIR / f"test_predictions_{task}_run{entry['run']}.csv", index=False, encoding="utf-8")
    json_path, md_path = output_paths(entry["run"])
    _write_json(json_path, protocol._rounded(result))
    md_path.write_text(final_markdown(result, rebuilt), encoding="utf-8", newline="\n")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for task in experiments.TASKS:
        joblib.dump(models[task]["final"], MODELS_DIR / f"{task}.joblib")
    _write_json(MODELS_DIR / "model_card.json", protocol._rounded(model_card(rebuilt, result, dev, sha)))
    return result


def _write_json(path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def model_card(p: dict, result: dict, dev: pd.DataFrame, sha: str) -> dict:
    log = experiments.read_log()
    lengths = dev["text"].str.len()
    tasks = {}
    for task, t in p["tasks"].items():
        row = log.loc[log.exp_id == experiments.exp_id(task, t["final"]["name"])].iloc[0]
        r = result["tasks"][task]
        tasks[task] = dict(
            file=f"models/{task}.joblib", labels=p["labels"][task], configuration=t["final"],
            pipeline_params=t["pipelines"]["final"],
            cv_macro_f1=dict(mean=float(row.cv_macro_f1_mean), sd=float(row.cv_macro_f1_std), folds="3 x 5 grouped"),
            expected_range=t["expected_range"]["final"],
            test=dict(macro_f1=r["metrics"]["final"]["macro_f1"], interval=r["bootstrap"]["macro_f1"]["final"],
                      accuracy=r["metrics"]["final"]["accuracy"], decision=r["decision"]),
            certainty=dict(score=t["certainty"]["score"], thresholds=t["certainty"]["thresholds"],
                           test_levels=r["levels"]))
    return dict(
        name="Arabic cafe review sentiment and main-aspect classifiers", commit=sha,
        fingerprint=p["fingerprint"], trained_on=dict(partition="development", rows=p["dev_rows"],
                                                      sha256=p["dev_sha256"], split_sha256=p["split_sha256"]),
        source=dict(record=config.SOURCE["record"], doi=config.SOURCE["doi"], license=config.SOURCE["license"]),
        versions=dict(p["versions"], joblib=joblib.__version__, python=platform.python_version()),
        tasks=tasks, joint_test_accuracy=result["joint"]["accuracy"],
        input=dict(text="one Arabic cafe/restaurant review as plain text; one main aspect per review",
                   development_length_chars=dict(min=int(lengths.min()), p01=float(lengths.quantile(0.01)),
                                                 median=float(lengths.median()), p99=float(lengths.quantile(0.99)),
                                                 max=int(lengths.max()))),
        limits=["labels name one main aspect per review, not every aspect mentioned",
                "trained on one source of Saudi cafe reviews; other dialects and domains are untested",
                "certainty levels are words for score bands, not calibrated probabilities"],
    )


# ---- report ---------------------------------------------------------------------------------

def _ci(c: dict) -> str:
    return f"[{c['low']:.3f}, {c['high']:.3f}]"


def final_markdown(r: dict, p: dict) -> str:
    a = r["access"]
    lines = [
        "# Final test (held-out partition, read once)", "",
        f"- Run {a['run']}, {a['time']}, commit `{a['commit']}`" + (f", reason: {a['reason']}" if a["reason"] else ""),
        f"- Protocol fingerprint `{r['fingerprint']}` (equal to `reports/protocol.json`, checked before the test "
        "file was opened)",
        f"- Test partition: {r['test_rows']} reviews, SHA-256 `{a['test_sha256']}`; models trained on all "
        f"{r['dev_rows']} development reviews",
        f"- Intervals: {p['bootstrap']['level']:.0%} percentile bootstrap, {p['bootstrap']['resamples']} resamples "
        f"(seed {p['bootstrap']['seed']}), near-duplicate groups as the resampling unit", "",
    ]
    for task, t in r["tasks"].items():
        labels, m, b = experiments.LABELS[task], t["metrics"], t["bootstrap"]
        rng, d = t["expected_range"], t["decision"]
        lines += [
            f"## {task}", "",
            "| model | Macro-F1 | 95% interval | balanced accuracy | accuracy |", "|---|---|---|---|---|",
        ]
        for role in ROLES:
            label = f"{role} `{p['tasks'][task][role]['name']}`" if role in ("final", "reference") else f"`{role}`"
            lines.append(f"| {label} | {m[role]['macro_f1']:.4f} | {_ci(b['macro_f1'][role])} | "
                         f"{m[role]['balanced_accuracy']:.4f} | {m[role]['accuracy']:.4f} |")
        lines += ["", "| paired difference (final minus) | observed | 95% interval | share of resamples <= 0 |",
                  "|---|---|---|---|"]
        for other, pr in b["paired"].items():
            lines.append(f"| {other} | {pr['observed']:+.4f} | {_ci(pr)} | {pr['share_le_0']:.3f} |")
        beats = ", ".join(f"{k}: {'yes' if v else 'no'}" for k, v in d["beats_dummies"].items())
        lines += [
            "", f"Decision (frozen rule): beats the dummies ({beats}); versus the reference: {d['reference']}; "
            f"success: {'yes' if d['success'] else 'no'}.", "",
            f"Expected range from the fresh CV seeds: final [{rng['final']['low']:.3f}, {rng['final']['high']:.3f}] "
            f"-> test {m['final']['macro_f1']:.4f} is {t['position']['final']}; reference "
            f"[{rng['reference']['low']:.3f}, {rng['reference']['high']:.3f}] -> {m['reference']['macro_f1']:.4f} is "
            f"{t['position']['reference']}.", "",
            "Per class (final model):", "",
            "| class | precision | recall | F1 | 95% interval of F1 | test rows |", "|---|---|---|---|---|---|",
        ]
        for row in m["final"]["per_class"]:
            lines.append(f"| {row['label']} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | "
                         f"{_ci(b['per_class_f1']['final'][row['label']])} | {row['support']} |")
        lines += ["", "Confusion matrix (final model; rows = true, columns = predicted):", "",
                  "| true \\ predicted | " + " | ".join(labels) + " |", "|---" * (len(labels) + 1) + "|"]
        for label, counts in zip(labels, m["final"]["confusion"]):
            lines.append(f"| {label} | " + " | ".join(map(str, counts)) + " |")
        pairs = sorted(((counts[j], labels[i], labels[j]) for i, counts in enumerate(m["final"]["confusion"])
                        for j in range(len(labels)) if i != j and counts[j]), reverse=True)[:3]
        errors = sum(n for n, _, _ in pairs)
        lines += ["", "Most frequent confusions: " + ("; ".join(f"{t_} -> {p_}: {n}" for n, t_, p_ in pairs) or "none")
                  + f" ({errors} of {sum(sum(c) for c in m['final']['confusion']) - sum(np.diag(m['final']['confusion']))} "
                  "errors).", "", "Certainty levels on the test partition (thresholds frozen in the protocol):", "",
                  "| level | test rows | share | accuracy | fresh-seed OOF share | OOF accuracy |", "|---|---|---|---|---|---|"]
        fresh = {lv["level"]: lv for lv in p["tasks"][task]["certainty"]["fresh_levels"]}
        for lv in t["levels"]:
            fr = fresh[lv["level"]]
            acc, fr_acc = ("n/a" if x["n"] == 0 else f"{x['accuracy']:.3f}" for x in (lv, fr))
            lines.append(f"| {lv['level']} | {lv['n']} | {lv['share']:.3f} | {acc} | {fr['share']:.3f} | {fr_acc} |")
        lines.append("")
    j = r["joint"]
    lines += [
        "## Both labels", "",
        f"Joint accuracy (both labels right, final models): {j['accuracy']:.4f}; expected range "
        f"[{j['expected_range']['low']:.3f}, {j['expected_range']['high']:.3f}] -> {j['position']}.", "",
        "| aspect | sentiment | test rows | joint accuracy |", "|---|---|---|---|",
    ]
    for c in r["cells"]:
        note = " (too small for a claim)" if c["n"] < SMALL_CELL else ""
        lines.append(f"| {c['aspect']} | {c['sentiment']} | {c['n']} | {c['joint_accuracy']:.3f}{note} |")
    lines += [
        "", "These numbers are reported as found. No model, threshold or feature was changed after the test "
        "partition was read; a second reading of the test partition needs a written methodological reason and "
        "is logged in `reports/test_access_log.json`.", "",
    ]
    return "\n".join(lines)
