from unittest.mock import patch

import pandas as pd
import pytest

from utils.data_loading import load_motion_capture_data, load_participant_details


def _frame(rows):
    """Build a DataFrame that mimics the Excel sheet structure."""
    return pd.DataFrame(rows)


def _load_from_rows(rows):
    """Run the loader against an in-memory frame and return the parsed rows."""
    frame = _frame(rows)

    with patch("pandas.read_excel", return_value=frame) as mock_read_excel:
        participants = load_participant_details("fake-path.xlsx")

    mock_read_excel.assert_called_once_with("fake-path.xlsx")
    return participants


# RTSA side is derived from the pair of shoulder-flag columns.
@pytest.mark.parametrize(
    ("rtsa_r", "rtsa_l", "expected"),
    [
        pytest.param(1, 1, "both", id="both"),
        pytest.param(1, 0, "right", id="right"),
        pytest.param(0, 1, "left", id="left"),
        pytest.param(0, 0, None, id="none"),
        pytest.param(None, None, None, id="missing"),
        pytest.param(float("nan"), 1, "left", id="nan-and-left"),
        pytest.param(2, 0, None, id="unexpected-nonbinary"),
    ],
)
def test_load_participant_details_maps_rtsa_side(rtsa_r, rtsa_l, expected):
    participants = _load_from_rows(
        [
            {
                "fname": "participant-a",
                "RTSA-R": rtsa_r,
                "RTSA-L": rtsa_l,
                "TSA-R": 0,
                "TSA-L": 0,
                "R-DOM": 1,
                "L-DOM": 0,
                "Age": 74,
            }
        ]
    )

    assert participants[0]["rtsa_side"] == expected


# TSA side uses the same mapping rules as RTSA.
@pytest.mark.parametrize(
    ("tsa_r", "tsa_l", "expected"),
    [
        pytest.param(1, 1, "both", id="both"),
        pytest.param(1, 0, "right", id="right"),
        pytest.param(0, 1, "left", id="left"),
        pytest.param(0, 0, None, id="none"),
        pytest.param(None, None, None, id="missing"),
        pytest.param(float("nan"), 1, "left", id="nan-and-left"),
        pytest.param("yes", 0, None, id="unexpected-string"),
    ],
)
def test_load_participant_details_maps_tsa_side(tsa_r, tsa_l, expected):
    participants = _load_from_rows(
        [
            {
                "fname": "participant-b",
                "RTSA-R": 0,
                "RTSA-L": 0,
                "TSA-R": tsa_r,
                "TSA-L": tsa_l,
                "R-DOM": 1,
                "L-DOM": 0,
                "Age": 74,
            }
        ]
    )

    assert participants[0]["tsa_side"] == expected


# Dominant arm is a hard validation: exactly one of the two flags must be set.
@pytest.mark.parametrize(
    ("r_dom", "l_dom", "expected"),
    [
        pytest.param(1, 0, "right", id="right"),
        pytest.param(0, 1, "left", id="left")
    ],
)
def test_load_participant_details_maps_dominant_arm(r_dom, l_dom, expected):
    participants = _load_from_rows(
        [
            {
                "fname": "participant-c",
                "RTSA-R": 0,
                "RTSA-L": 0,
                "TSA-R": 0,
                "TSA-L": 0,
                "R-DOM": r_dom,
                "L-DOM": l_dom,
                "Age": 74,
            }
        ]
    )

    assert participants[0]["dominant_arm"] == expected

@pytest.mark.parametrize(
    ("r_dom", "l_dom"),
    [
        pytest.param(1, 1, id="both-set"),
        pytest.param(0, 0, id="neither-set"),
        pytest.param(None, None, id="missing"),
        pytest.param(float("nan"), float("nan"), id="nan-and-nan"),
        pytest.param("unexpected", 0, id="unexpected-string"),
    ],
)
def test_load_participant_details_rejects_invalid_dominant_arm_flags(r_dom, l_dom):
    with pytest.raises(ValueError, match="exactly one dominant arm flag"):
        _load_from_rows(
            [
                {
                    "fname": "participant-d",
                    "RTSA-R": 0,
                    "RTSA-L": 0,
                    "TSA-R": 0,
                    "TSA-L": 0,
                    "R-DOM": r_dom,
                    "L-DOM": l_dom,
                    "Age": 74,
                }
            ]
        )


# These fields are copied directly into the output record.
@pytest.mark.parametrize(
    ("filename", "age"),
    [
        pytest.param("participant-d", 74, id="expected-values"),
        pytest.param("participant-e", "74", id="string-age"),
        pytest.param(None, None, id="missing-values"),
    ],
)
def test_load_participant_details_preserves_filename_and_age(filename, age):
    participants = _load_from_rows(
        [
            {
                "fname": filename,
                "RTSA-R": 0,
                "RTSA-L": 0,
                "TSA-R": 0,
                "TSA-L": 0,
                "R-DOM": 1,
                "L-DOM": 0,
                "Age": age,
            }
        ]
    )

    assert participants[0]["filename"] == filename
    assert participants[0]["age"] == age


def test_load_participant_details_initializes_nested_arm_rotation_fields_to_none():
    participants = _load_from_rows(
        [
            {
                "fname": "participant-rotations",
                "RTSA-R": 0,
                "RTSA-L": 0,
                "TSA-R": 0,
                "TSA-L": 0,
                "R-DOM": 1,
                "L-DOM": 0,
                "Age": 74,
            }
        ]
    )

    participant = participants[0]
    expected_arm_metrics = {
        "humerothoracic_rotation": None,
        "glenohumeral_rotation": None,
        "total_rotation_x": None,
        "total_rotation_y": None,
        "total_rotation_z": None,
    }

    assert participant["left"] == expected_arm_metrics
    assert participant["right"] == expected_arm_metrics


@pytest.mark.parametrize("op_side", ["left", "right"])
def test_load_participant_details_allows_dynamic_side_access_for_arm_metrics(op_side):
    participants = _load_from_rows(
        [
            {
                "fname": "participant-op-side",
                "RTSA-R": 0,
                "RTSA-L": 0,
                "TSA-R": 0,
                "TSA-L": 0,
                "R-DOM": 1,
                "L-DOM": 0,
                "Age": 74,
            }
        ]
    )

    participant = participants[0]
    participant[op_side]["humerothoracic_rotation"] = 1.23

    assert participant[op_side]["humerothoracic_rotation"] == 1.23


def test_load_participant_details_uses_independent_left_and_right_arm_dictionaries():
    participants = _load_from_rows(
        [
            {
                "fname": "participant-independent-arms",
                "RTSA-R": 0,
                "RTSA-L": 0,
                "TSA-R": 0,
                "TSA-L": 0,
                "R-DOM": 1,
                "L-DOM": 0,
                "Age": 74,
            }
        ]
    )

    participant = participants[0]
    participant["left"]["total_rotation_x"] = 2.5

    assert participant["left"]["total_rotation_x"] == 2.5
    assert participant["right"]["total_rotation_x"] is None


@pytest.mark.parametrize(
    ("rtsa_r", "rtsa_l", "tsa_r", "tsa_l"),
    [
        pytest.param(1, 0, 1, 0, id="right-right"),
        pytest.param(0, 1, 0, 1, id="left-left"),
        pytest.param(1, 1, 0, 1, id="both-left"),
        pytest.param(0, 1, 1, 1, id="left-both"),
    ],
)
def test_load_participant_details_rejects_same_arm_rtsa_and_tsa(
    rtsa_r, rtsa_l, tsa_r, tsa_l
):
    with pytest.raises(ValueError, match="RTSA and TSA on the same arm"):
        _load_from_rows(
            [
                {
                    "fname": "participant-e",
                    "RTSA-R": rtsa_r,
                    "RTSA-L": rtsa_l,
                    "TSA-R": tsa_r,
                    "TSA-L": tsa_l,
                    "R-DOM": 1,
                    "L-DOM": 0,
                    "Age": 74,
                }
            ]
        )


def test_load_participant_details_returns_empty_list_for_empty_input():
    assert _load_from_rows([]) == []


def test_load_motion_capture_data_uses_filename_and_default_directory():
    expected = pd.DataFrame([[1.0, 2.0]])

    with patch("numpy.loadtxt", return_value=expected) as mock_loadtxt:
        result = load_motion_capture_data("participant-a.tsv")

    assert result is expected
    mock_loadtxt.assert_called_once_with(
        "./raw_data/participant-a.tsv",
        delimiter="\t",
        skiprows=1,
        usecols=range(1, 19),
    )


def test_load_motion_capture_data_allows_custom_data_directory():
    expected = pd.DataFrame([[3.0, 4.0]])

    with patch("numpy.loadtxt", return_value=expected) as mock_loadtxt:
        result = load_motion_capture_data("participant-b.tsv", data_dir="./data")

    assert result is expected
    mock_loadtxt.assert_called_once_with(
        "./data/participant-b.tsv",
        delimiter="\t",
        skiprows=1,
        usecols=range(1, 19),
    )