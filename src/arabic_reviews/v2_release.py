"""Freezing and building version 2 (``docs/V2_PLAN.md``) from the results of ``v2.run_experiments``.

``build`` derives the version-2 protocol from ``reports/v2/results.json``, refuses results computed on
other data, by other code or under other library versions, refuses any protocol whose fingerprint is not
the released one (``v2_fingerprint``), fits the version-2 pipelines on the whole development partition
(local files) and writes the version-2 model card. ``build(verify=True)`` also recomputes every result,
after the fingerprint check. Everything runs inside ``v2.guard()``; the held-out test is never read.

This module is not among ``v2.SOURCES``: it only freezes and builds from the results, so changing it does
not change what the results depend on. Its own hash (``release_sha256``) is part of the protocol
fingerprint instead, so a change to it needs a new release build.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import joblib
import pandas as pd

from . import config, experiments, final, protocol, split, v2_fingerprint
from .features import build_pipeline
from .v2 import (ASPECT_LABELS, CARD_FILE, DROP_SLACK, EDGE_SLACK, EDGE_WIDTH, GRID, LABELS, MIN_GAIN, MIN_GROUPS,
                 MIN_NEUTRAL_GAIN, MIN_OTHER_CHANGE, MIN_WINS, MODELS_DIR, NEUTRAL, OUTER_SEEDS, PROTOCOL_FILE,
                 RESULTS_JSON, TARGETS, _round, _run, candidates, check_frozen_cues, file_sha256, frozen_specs, guard,
                 runtime_versions, same_runtime, source_hashes)


def release_sha256() -> str:
    """SHA-256 of this module (line endings normalised): the code that fits the models and writes the card."""
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def protocol_v2(r: dict, specs: dict[str, dict], results_sha256: str) -> dict:
    """The fingerprinted version-2 protocol. ``results_sha256`` (line endings normalised) binds every number
    and decision of results.json and ``release_sha256`` the code of this module, so an edited results file
    or release code changes the fingerprint."""
    pipelines = {task: protocol.pipeline_params(spec) for task, spec in specs.items()}
    body = dict(results_sha256=results_sha256, release_sha256=release_sha256(),
        version="v2", plan="docs/V2_PLAN.md", released=r["released"], adopted=r["adopted"],
        configurations={task: experiments._spec_summary(spec) | {"params": spec.get("params", {}),
                                                                 "cues": bool(spec.get("cues"))}
                        for task, spec in specs.items()},
        pipelines=pipelines,
        certainty={task: dict(score="top class probability", thresholds_by_class=r["certainty"][task]["thresholds"])
                   for task in specs},
        cues_sha256=r["cues_sha256"], dev=r["dev"], versions=r["versions"], sources=r["sources"],
        results_commit=r["git_sha"],
        rules=dict(min_gain=MIN_GAIN, min_neutral_gain=MIN_NEUTRAL_GAIN, min_other_change=MIN_OTHER_CHANGE,
                   min_wins=MIN_WINS, outer_seeds=OUTER_SEEDS, grid=GRID, targets=dict(TARGETS),
                   min_groups=MIN_GROUPS, edge=[EDGE_WIDTH, EDGE_SLACK], drop_slack=DROP_SLACK),
    )
    body = _round(json.loads(json.dumps(body, ensure_ascii=False, default=list)))
    fingerprint = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return dict(body, fingerprint=fingerprint)


def comparable(text: str) -> dict:
    """results.json without what a faithful recomputation may change: the commit of the run, and the
    Python patch release (the runtime policy of ``v2.same_runtime`` accepts another patch release)."""
    r = json.loads(text)
    r.pop("git_sha", None)
    if "python" in r.get("versions", {}):
        r["versions"] = dict(r["versions"], python=".".join(r["versions"]["python"].split(".")[:2]))
    return r


def verify_results() -> None:
    """Recompute every result (about 20 minutes) into a temporary folder, without adding log rows, and
    stop unless it equals the committed results.json (``comparable``)."""
    with tempfile.TemporaryDirectory() as tmp:
        _run(None, check_v1=True, out_dir=Path(tmp), append=False)
        fresh = (Path(tmp) / RESULTS_JSON.name).read_text(encoding="utf-8")
    if comparable(fresh) != comparable(RESULTS_JSON.read_text(encoding="utf-8")):
        raise RuntimeError("recomputed results differ from reports/v2/results.json; the build was not started")


def released_protocol() -> tuple[dict, dict[str, dict], dict, pd.DataFrame]:
    """The checks of ``build``, inside ``guard()``: results.json computed on this data and cue list, by this
    code and under these library versions, and a protocol with the released fingerprint. Returns the
    results, the specifications, the protocol and the development partition."""
    cue_sha = check_frozen_cues()
    results_text = RESULTS_JSON.read_text(encoding="utf-8")
    r = json.loads(results_text)
    dev = split.load_dev()
    if r["cues_sha256"] != cue_sha or r["dev"]["sha256"] != file_sha256(config.DEV_FILE) \
            or r["dev"]["split_sha256"] != file_sha256(config.SPLIT_FILE):
        raise RuntimeError("results.json was computed on other data or another cue list; re-run v2-experiments")
    if r.get("sources") != source_hashes():
        raise RuntimeError("results.json was computed by other code; re-run v2-experiments")
    if not same_runtime(r["versions"]):
        raise RuntimeError("the installed library versions differ from those of results.json; install "
                           "requirements.txt or re-run v2-experiments")
    frozen = frozen_specs()
    specs = {"sentiment": candidates(frozen["sentiment"])[r["released"]], "aspect": frozen["aspect"]}
    p = protocol_v2(r, specs, hashlib.sha256(results_text.replace("\r\n", "\n").encode("utf-8")).hexdigest())
    released = v2_fingerprint.FROZEN_FINGERPRINT
    if released is not None and p["fingerprint"] != released:
        raise RuntimeError(f"the version-2 protocol fingerprint {p['fingerprint'][:12]}... is not the released "
                           f"{released[:12]}...: results.json, the release code, the code or the data changed")
    return r, specs, p, dev


def build(verify: bool = False) -> dict:
    """Freeze the version-2 protocol from ``results.json`` and fit the version-2 pipelines on the whole
    development partition (local files), then write the version-2 model card. The protocol must have the
    released fingerprint, which binds the hashes of results.json and of this module, so an edited results
    file or release code is refused before anything is fitted or written. With ``verify`` every result is
    then recomputed and must equal the committed one. Models are saved only once their digests pass."""
    if verify:
        with guard():
            released_protocol()
        verify_results()
    with guard() as opened:
        r, specs, p, dev = released_protocol()
        runtime = runtime_versions()
        X = dev["text"].to_numpy()
        models = {task: build_pipeline(spec).fit(X, dev[experiments.TASKS[task]].to_numpy())
                  for task, spec in specs.items()}
        digests = {task: final.dev_digest(model, X) for task, model in models.items()}
        recorded = json.loads(final.DIGESTS_FILE.read_text(encoding="utf-8"))
        same_versions = recorded["versions"] == final.runtime_versions()
        for task in specs:
            same_model = task == "aspect" or r["released"] == "V1"
            if same_model and same_versions and digests[task] != recorded["digests"][task]:
                raise RuntimeError(f"the version-2 {task} model should equal the version-1 one but its digest differs")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        for task, model in models.items():
            joblib.dump(model, MODELS_DIR / f"{task}.joblib")
        card = model_card_v2(r, p, specs, digests, dev)
        outputs = {path.resolve().relative_to(config.ROOT.resolve()).as_posix() for path in (PROTOCOL_FILE, CARD_FILE)}
        build_record = dict(read=sorted(opened["read"]), written=sorted(opened["written"] | outputs),
                            versions=runtime, verified=verify)  # not part of the fingerprint
        PROTOCOL_FILE.write_text(json.dumps(dict(p, build=build_record), ensure_ascii=False, indent=1) + "\n",
                                 encoding="utf-8", newline="\n")
        CARD_FILE.write_text(json.dumps(card, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    return dict(fingerprint=p["fingerprint"], released=r["released"], digests=digests, read=build_record["read"],
                verified=verify)


def model_card_v2(r: dict, p: dict, specs: dict, digests: dict, dev: pd.DataFrame) -> dict:
    lengths = dev.text.str.len()
    sel = r["selection"]["scores"]
    return _round(dict(
        name="Arabic cafe review sentiment and main-aspect classifiers, version 2", version="v2",
        fingerprint=p["fingerprint"], released_sentiment=r["released"], adopted_new_sentiment_model=r["adopted"],
        evaluation="development data only (grouped CV); no independent test exists for version 2, and the old "
                   "held-out test was not used",
        trained_on=dict(partition="development", rows=len(dev), sha256=r["dev"]["sha256"],
                        split_sha256=r["dev"]["split_sha256"]),
        source=dict(record=config.SOURCE["record"], doi=config.SOURCE["doi"], license=config.SOURCE["license"]),
        versions=runtime_versions(),
        tasks={task: dict(
            file=f"models/v2/{task}.joblib", labels=LABELS if task == "sentiment" else ASPECT_LABELS,
            configuration=p["configurations"][task], same_as_v1=(task == "aspect" or r["released"] == "V1"),
            dev_digest=digests[task],
            certainty=dict(thresholds_by_class=r["certainty"][task]["thresholds"],
                           outer_levels=r["certainty"][task]["outer_levels"],
                           v1_thresholds=r["certainty"][task]["v1_thresholds"],
                           v1_rule_outer_levels=r["certainty"][task]["v1_rule_outer_levels"],
                           measured_on="outer folds (seeds 300-302) of the development partition, every review "
                                       "three times; frequencies observed on development data, not guarantees"),
            cv=(dict(selection=sel[r["released"]], outer=r["outer_scores"][r["released"]])
                if task == "sentiment" else r["aspect"]),
        ) for task in specs},
        nested_vs_v1=dict(gain=r["nested"]["bootstrap"]["macro"], neutral=r["nested"]["bootstrap"][NEUTRAL],
                          chosen_counts=r["nested"]["chosen_counts"]),
        experiment=dict(
            chosen_on_selection_folds=r["selection"]["chosen"], adopted=r["adopted"],
            candidates={name: dict(selection_macro_f1=s["macro"]["mean"], selection_neutral_f1=s["per_class"][NEUTRAL],
                                   outer_macro_f1=r["outer_scores"][name]["macro"]["mean"],
                                   outer_neutral_f1=r["outer_scores"][name]["per_class"][NEUTRAL])
                        for name, s in sel.items()},
            stability_vs_v1=(None if r["stability"] is None else
                             dict(gain=r["stability"]["bootstrap"]["macro"], neutral=r["stability"]["bootstrap"][NEUTRAL])),
            rule="adopted only if the nested CV estimate of the whole selection procedure gains >= 0.01 Macro-F1 "
                 "with a 95% interval above 0, a positive neutral F1 change and no positive or negative F1 drop "
                 "larger than 0.02 (docs/V2_PLAN.md section 5)"),
        joint_accuracy=r["joint_accuracy"],
        input=dict(development_length_chars=dict(min=int(lengths.min()), median=float(lengths.median()),
                                                 p99=float(lengths.quantile(0.99)), max=int(lengths.max()))),
    ))
