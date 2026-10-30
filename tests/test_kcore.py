import numpy as np
import pandas as pd

from utils.data_processing import load_results, process_results
from utils.kcore_subsetting import evaluate_all_kcore_candidates, subset_kcore_data


def test_kcore_on_model_frame(db_con):
    """The k-core groups courses by (name, distance_miles); with name = course_id that is
    one group per course, and runners 10 and 11 finished both courses."""
    frame = process_results(load_results(db_con))
    flagged = subset_kcore_data(frame, alpha=2, beta=2)
    core = flagged[flagged["in_kcore"]]
    assert set(core["participant_id"]) == {"10", "11"}
    assert set(core["name"]) == {"100-50k", "200-100mi"}
    # Course-completion closure: the rest of the 50K field is flagged too
    assert set(flagged.loc[flagged["in_closure"], "participant_id"]) == {"12", "13"}


def _frame(edges):
    """A minimal model frame from (runner, course, finished) triples."""
    return pd.DataFrame(
        [{"participant_id": r, "name": c, "distance_miles": 31.0, "finished": f} for r, c, f in edges]
    )


def test_dnfs_count_as_edges():
    # Runner c DNF'd course Y; a finishes-only graph would leave c with one course.
    frame = _frame([("a", "X", True), ("a", "Y", True), ("b", "X", True), ("b", "Y", True),
                    ("c", "X", True), ("c", "Y", False)])
    flagged = subset_kcore_data(frame, alpha=2, beta=3)
    assert set(flagged.loc[flagged["in_kcore"], "participant_id"]) == {"a", "b", "c"}


def _brute_core(edges, alpha, beta):
    runners = {r for r, _, _ in edges}
    courses = {c for _, c, _ in edges}
    pairs = {(r, c) for r, c, _ in edges}
    while True:
        r2 = {r for r in runners if sum((r, c) in pairs for c in courses) >= alpha}
        c2 = {c for c in courses if sum((r, c) in pairs for r in r2) >= beta}
        if (r2, c2) == (runners, courses):
            return runners, courses
        runners, courses = r2, c2


def test_sweep_matches_brute_force():
    rng = np.random.default_rng(0)
    edges = [(f"r{rng.integers(60)}", f"c{rng.integers(12)}", bool(rng.random() < 0.9)) for _ in range(500)]
    frame = _frame(edges)
    sweep = evaluate_all_kcore_candidates(frame, min_alpha=2, min_beta=3, verbose=False, use_cache=False)
    assert len(sweep)
    for row in sweep.itertuples():
        runners, courses = _brute_core(edges, row.alpha, row.beta)
        assert (row.kcore_runners, row.kcore_courses) == (len(runners), len(courses))
        flagged = subset_kcore_data(frame, row.alpha, row.beta)
        assert set(flagged.loc[flagged["in_kcore"], "participant_id"]) == runners
    # Every (α, β) past the sweep's edge has an empty core.
    last = sweep[sweep.alpha == sweep.alpha.max()]
    assert _brute_core(edges, int(last.alpha.max()) + 1, 3) == (set(), set())
