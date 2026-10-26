from utils.data_processing import load_results, process_results
from utils.kcore_subsetting import subset_kcore_data


def test_kcore_on_model_frame(mirror_con):
    """The k-core groups courses by (name, distance_miles); with name = course_id that is
    one group per course, and runners 10 and 11 finished both courses."""
    frame = process_results(load_results(mirror_con))
    flagged = subset_kcore_data(frame, alpha=2, beta=2)
    core = flagged[flagged["in_kcore"]]
    assert set(core["participant_id"]) == {"10", "11"}
    assert set(core["name"]) == {"100-50k", "200-100mi"}
    # Course-completion closure: the rest of the 50K field is flagged too
    assert set(flagged.loc[flagged["in_closure"], "participant_id"]) == {"12", "13"}
