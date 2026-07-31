import copy
import pickle
import numpy as np
import pytest
from pathlib import Path

from modules.data_loading import (
    RotationData,
    ArmRotationDetails,
    ParticipantDetails,
)
from schema import Heatmap
from modules.data_saving import save_data_to_pickle


@pytest.fixture
def participant_details() -> list[ParticipantDetails]:
    """Create a representative participant dataset."""

    heatmap = Heatmap(
        elevation=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        poe=np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64),
        ir_er=np.array([[9.0, 10.0], [11.0, 12.0]], dtype=np.float64),
        cumulative_motion=np.array([[13.0, 14.0], [15.0, 16.0]], dtype=np.float64),
        sample_count=np.array([[1, 2], [3, 4]], dtype=np.int32),
    )

    rotation_data = RotationData(
        trace_total=np.float64(123.45),
        heatmap=heatmap,
    )

    arm = ArmRotationDetails(
        humerothoracic=rotation_data,
        glenohumeral=rotation_data,
    )

    return [
        ParticipantDetails(
            filename=Path("test_participant_001.csv"),
            rtsa_side="left",
            tsa_side=None,
            dominant_arm="right",
            age=65,
            left=arm,
            right=arm,
        )
    ]


class TestSaveDataToPickle:
    def test_saves_file_to_specified_path(
        self,
        tmp_path: Path,
        participant_details,
    ):
        file_path = tmp_path / "participants.pkl"

        save_data_to_pickle(participant_details, file_path)

        assert file_path.exists()
        assert file_path.is_file()

    def test_does_not_modify_input_object(
        self,
        tmp_path: Path,
        participant_details,
    ):
        original = copy.deepcopy(participant_details)
        file_path = tmp_path / "participants.pkl"

        save_data_to_pickle(participant_details, file_path)

        assert participant_details == original

    def test_saved_pickle_matches_original_data(
        self,
        tmp_path: Path,
        participant_details,
    ):
        file_path = tmp_path / "participants.pkl"

        save_data_to_pickle(participant_details, file_path)

        with open(file_path, "rb") as f:
            loaded = pickle.load(f)

        assert loaded == participant_details