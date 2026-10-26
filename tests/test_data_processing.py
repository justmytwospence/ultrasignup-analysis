import pandas as pd

from utils.data_processing import filter_races_with_dnfs, load_results, process_results

MODEL_FRAME_COLUMNS = {
    "event_distance_id", "participant_id", "name", "distance_miles", "time_ms",
    "finished", "pace_min_mi", "gender", "age",
    "course_id", "race_name", "race_date", "year", "distance_km", "status", "source",
}


def test_load_results_sources(mirror_con):
    default = load_results(mirror_con)
    assert set(default["source"]) == {"results_api", "legacy_2025"}
    with_hidden = load_results(mirror_con, sources=("results_api", "legacy_2025", "participant_history"))
    assert len(with_hidden) == len(default) + 1


def test_model_frame_contract(mirror_con):
    frame = process_results(load_results(mirror_con))
    assert MODEL_FRAME_COLUMNS <= set(frame.columns)
    assert (frame["name"] == frame["course_id"]).all()
    assert frame["age"].dtype.kind == "i"
    assert frame["participant_id"].map(type).eq(str).all()


def test_status_mapping_and_filters(mirror_con):
    frame = process_results(load_results(mirror_con))
    race1 = frame[frame["event_distance_id"] == 1].set_index("participant_id")
    # finished, finished_small_field and unofficial finish; dnf stays as a DNF
    assert race1.loc[["10", "11", "12"], "finished"].all()
    assert not race1.loc["13", "finished"]
    assert pd.isna(race1.loc["13", "time_ms"])
    # dns, dq, unknown age, gender X and the 1-hour 50K are gone
    assert set(race1.index) == {"10", "11", "12", "13"}
    # fixed-time, bike and 5K races are gone; the legacy 100-miler stays
    assert set(frame["event_distance_id"]) == {1, 2}


def test_units(mirror_con):
    frame = process_results(load_results(mirror_con)).set_index(["event_distance_id", "participant_id"])
    row = frame.loc[(1, "10")]
    assert row["time_ms"] == 5 * 3600 * 1000
    assert abs(row["distance_miles"] - 50 / 1.609344) < 1e-9
    assert abs(row["pace_min_mi"] - 300 / (50 / 1.609344)) < 1e-9


def test_filter_races_with_dnfs(mirror_con):
    frame = process_results(load_results(mirror_con))
    kept = filter_races_with_dnfs(frame)
    assert set(kept["name"]) == {"100-50k"}
