"""The notebook helpers and the committed notebook: the test-file lock, decisions read from the log,
test results read from the saved report only, and a notebook that was run top to bottom."""

import inspect
import json
import re
import subprocess
import sys
import textwrap

import pytest

from arabic_reviews import config, experiments, final, walkthrough

NOTEBOOK = config.ROOT / "notebooks" / "project_walkthrough.ipynb"


def test_lock_refuses_every_way_of_opening_the_locked_file(tmp_path):
    # Audit hooks cannot be removed, so the lock is exercised in a separate interpreter.
    script = textwrap.dedent(f"""
        import os, pathlib, pandas as pd
        from arabic_reviews import config, walkthrough
        locked = pathlib.Path({str(tmp_path)!r}) / "test.csv"
        other = locked.with_name("dev.csv")
        for path in (locked, other):
            path.write_text("rid,text\\n1,x\\n", encoding="utf-8")
        hard = locked.with_name("hard_link.csv")
        os.link(locked, hard)
        soft = locked.with_name("soft_link.csv")
        try:
            os.symlink(locked, soft)
        except OSError:  # symbolic links need a privilege on Windows
            soft = None
        config.TEST_FILE = locked
        walkthrough.lock_test_partition()
        attempts = {{
            "open": lambda: open(locked, encoding="utf-8"),
            "relative open": lambda: (os.chdir(locked.parent), open("test.csv")),
            "bytes path": lambda: open(os.fsencode(locked), "rb"),
            "hard link": lambda: open(hard, "rb"),
            "pathlib": locked.read_bytes,
            "pandas": lambda: pd.read_csv(locked),
            "os.open": lambda: os.open(locked, os.O_RDONLY),
            "write": lambda: locked.write_text("x"),
        }}
        if soft is not None:
            attempts["symbolic link"] = lambda: open(soft, "rb")
        for name, attempt in attempts.items():
            try:
                attempt()
            except PermissionError:
                continue
            raise SystemExit(f"{{name}} was not refused")
        assert len(pd.read_csv(other)) == 1 and locked.exists()
        print("all refused")
    """)
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=tmp_path)
    assert run.returncode == 0, run.stderr + run.stdout
    assert "all refused" in run.stdout


def test_the_test_file_is_named_only_to_lock_it():
    assert inspect.getsource(walkthrough).count("TEST_FILE") == \
        inspect.getsource(walkthrough.lock_test_partition).count("TEST_FILE") > 0


# The fresh-clone branch (download, validate, split) is exercised end to end by the CI notebook job,
# which starts without data and then requires the whole tracked tree to be unchanged.
@pytest.mark.skipif(not (config.RAW_FILE.exists() and config.DEV_FILE.exists()), reason="data not prepared")
def test_prepare_local_data_keeps_existing_files_and_published_reports():
    table = walkthrough.prepare_local_data()
    assert table.state.tolist() == ["present; size, MD5 and SHA-256 verified", "present"]
    assert walkthrough.CHECKS["published data manifest, validation and split reports unchanged"]["result"] == "pass"


def test_check_records_passes_and_stops_on_a_failure():
    walkthrough.check("toy check passes", True, "detail")
    assert walkthrough.CHECKS["toy check passes"]["result"] == "pass"
    with pytest.raises(AssertionError, match="toy check fails"):
        walkthrough.check("toy check fails", False, "why")
    assert walkthrough.CHECKS["toy check fails"]["result"] == "FAIL"
    walkthrough.skip("toy check skipped", "reason")
    assert walkthrough.CHECKS["toy check skipped"]["result"] == "not applicable"
    for name in ("toy check passes", "toy check fails", "toy check skipped"):
        del walkthrough.CHECKS[name]


@pytest.mark.parametrize("task, accepted", [
    ("sentiment", ["logreg_char25", "logreg_word11+char25", "a6_balanced"]),
    ("aspect", ["logreg_char25", "a6_balanced"]),
])
def test_ablation_table_follows_the_logged_decisions(task, accepted):
    table = walkthrough.ablation_table(task)
    assert table.loc[table.decision == "accepted", "candidate"].tolist() == accepted
    assert len(table) == len(experiments.AXIS1_CHAIN) - 1 + sum(len(v) for _, _, v in experiments.AXES)


def test_model_selection_matches_the_frozen_configurations():
    frozen = walkthrough.frozen_specs()["final"]
    for task in walkthrough.TASKS:
        s = walkthrough.model_selection(task)
        assert s["summary"]["final"] == frozen[task]["name"]
        assert frozen[task]["params"] == {s["summary"]["param"]: s["summary"]["used"]}


def test_final_result_tables_are_read_from_the_saved_report():
    r = json.loads(final.FINAL_JSON.read_text(encoding="utf-8"))
    p = json.loads((config.REPORTS_DIR / "protocol.json").read_text(encoding="utf-8"))
    models = walkthrough.heldout_model_table(r, p)
    for task in walkthrough.TASKS:
        assert models.loc[(task, "final"), "macro_f1"] == round(r["tasks"][task]["metrics"]["final"]["macro_f1"], 4)
        assert models.loc[(task, "final"), "configuration"] == p["tasks"][task]["final"]["name"]
        matrix = walkthrough.heldout_confusions(r)[f"{task} (test, final model)"]
        assert int(matrix.to_numpy().sum()) == r["test_rows"]
    paired = walkthrough.paired_table(r)
    assert set(paired.task_success) == {"yes"}
    levels = walkthrough.heldout_levels(r, p)
    assert levels.groupby(level="task").test_rows.sum().tolist() == [r["test_rows"]] * 2


# ---- the committed notebook -------------------------------------------------------------------

def _code_cells():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def _source(cell) -> str:
    return "".join(cell["source"])


def _text(cell) -> str:
    parts = []
    for out in cell.get("outputs", []):
        parts.append("".join(out.get("text", "")))
        parts.append("".join(out.get("data", {}).get("text/plain", "")))
    return "".join(parts)


def test_notebook_was_run_top_to_bottom_without_errors():
    cells = _code_cells()
    assert [c["execution_count"] for c in cells] == list(range(1, len(cells) + 1))
    for c in cells:
        assert not [o for o in c["outputs"] if o["output_type"] == "error"], _source(c)[:80]
        assert not [o for o in c["outputs"] if o.get("name") == "stderr"], _source(c)[:80]


def test_notebook_never_evaluates_on_the_test_partition():
    cells = _code_cells()
    forbidden = re.compile(r"run_final|read_test_once|cmd_final|save_predictions|_log_access|"
                           r"arabic_reviews final(?! --check)")
    for c in cells:
        assert not forbidden.search(_source(c)), _source(c)[:80]
    lock = next(i for i, c in enumerate(cells) if "lock_test_partition()" in _source(c))
    lock_cell = _source(cells[lock])
    # the full data (all partitions) is dropped in the cell that installs the lock, before the lock
    assert lock_cell.index("del raw, validation, validated, split_result") < lock_cell.index("lock_test_partition()")
    full_data = re.compile(r"\b(raw|validation|validated|split_result|read_source|run_validation|make_split|"
                           r"RAW_FILE|VALIDATED_FILE|TEST_FILE)\b")
    for c in cells[lock + 1:]:
        names = set(full_data.findall(_source(c)))
        assert names <= {"TEST_FILE"}, (names, _source(c)[:80])
        if names:  # the one deliberate attempt, which must be refused
            assert "except PermissionError" in _source(c) and "refused:" in _text(c)
    assert sum("TEST_FILE" in _source(c) for c in cells) == 1


def test_notebook_outputs_show_the_frozen_protocol_and_passing_checks():
    text = "".join(_text(c) for c in _code_cells())
    assert final.FROZEN_FINGERPRINT[:16] in text
    assert "equals the frozen protocol" in text
    assert "FAIL" not in text
    assert "the held-out test was read exactly once" in text
