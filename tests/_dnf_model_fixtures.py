"""Small deterministic inputs for the frozen pre-extraction model references."""

import numpy as np
import pandas as pd


def reporting_editions():
    frame = pd.DataFrame({
        "did": [101, 102, 103, 104],
        "series_id": [9, 2, 9, 2],
        "year": [2004, 2006, 2010, 2011],
        "distance_km": [50.0, 80.0, 100.0, 42.0],
        "n_starters": [5, 3, 4, 2],
        "n_finishers": [3, 3, 4, 2],
        "n_dnf": [2, 0, 0, 0],
        "n_dns": [1, 1, 0, 0],
        "kind": ["reports", "reports, zero DNFs", "unknown", "unknown"],
        "known_reporting": [True, True, False, False],
    })
    frame["log_distance"] = np.log(frame["distance_km"] / 50)
    return frame


def dnf_input_dict():
    f_course = np.array([0, 0, 1, 1, 0, 0])
    f_runner = np.array([0, 1, 2, 3, 0, 3])
    f_age = np.array([-1.0, 0.5, 0.0, 1.0, -0.5, 0.2])
    return dict(
        courses=np.array(["course-a", "course-b"]),
        runners=pd.Index([10, 11, 12, 13]),
        course_x=np.log(np.array([50.0, 100.0]) / 50),
        runner_gender=np.array([0, 1, 0, 1]),
        log_pace=np.log(np.array([8.0, 9.0, 8.5, 10.0, 7.5, 9.5])),
        f_course=f_course,
        f_runner=f_runner,
        f_gender=np.array([0, 1, 0, 1, 0, 1]),
        f_x=np.log(np.array([50.0, 50.0, 100.0, 100.0, 50.0, 50.0]) / 50),
        f_age=f_age,
        f_age2=f_age**2,
        edition_ids=np.array([101, 102, 103]),
        f_edition=np.array([0, 0, 1, 1, 2, 2]),
        s_course=np.array([0, 0, 1, 1, 1]),
        s_runner=np.array([0, 1, 2, 3, 0]),
        s_gender=np.array([0, 1, 0, 1, 0]),
        s_age=np.array([-1.0, 0.5, 0.0, 1.0, -0.5]),
        dnf=np.array([0, 1, 0, 0, 1]),
        starts=pd.DataFrame({"event_distance_id": [101, 101, 102, 102, 104]}),
        editions=pd.DataFrame({
            "course_idx": [0, 1, 0],
            "reports_dnfs": [True, True, False],
            "n_finishers": [2, 2, 2],
            "p_reports_prior": [0.8, 0.7, 0.6],
            "p_lists_dns": [0.4, 0.3, 0.2],
        }, index=pd.Index([101, 102, 103], name="event_distance_id")),
        unknown=np.array([False, False, True]),
    )
