"""
Progress bar management for the analysis pipeline.

Author: Christopher Millward
"""

from tqdm import tqdm
from schema import ParticipantDetails

# Main progress bar manager
class ProgressManager:
    """Class to manage progress bars for the analysis pipeline.
    
    Attributes:
        n_outer (int): Number of outer iterations (participants).
        n_inner (int): Number of inner iterations (arms * bins).
        outer (tqdm): Outer progress bar.
        inner (list[tqdm]): List of inner progress bars for each participant.

    Methods:
        update_inner(i: int, n: int = 1): Update the inner progress bar for participant i by n steps.
        update_outer(n: int = 1): Update the outer progress bar by n steps.
        close(): Close all progress bars.

    """
    def __init__(self, participant_details: list[ParticipantDetails]):

        n_participants = len(participant_details)
        n_arms = 2
        n_elev_bins = (
            participant_details[0]
            .right
            .humerothoracic
            .heatmap
            .shape[0]
        )
        n_poe_bins = (
            participant_details[0]
            .right
            .humerothoracic
            .heatmap
            .shape[1]
        )

        self.n_outer = n_participants
        self.n_inner = n_elev_bins * n_poe_bins * n_arms

        self.outer = tqdm(
            total=self.n_outer,
            desc="Entire Analysis",
            position=0,
            leave=True,
        )

        self.inner = [
            tqdm(
                total=self.n_inner,
                desc=f"Participant [{i}]",
                position=i + 1,
                leave=True,
            )
            for i in range(self.n_outer)
        ]

    def update_inner(self, i: int, n: int = 1) -> None:
        """
        Update the inner progress bar for participant i by n steps.

        Args:
            i (int): The index of the participant.
            n (int): The number of steps to update the progress bar by.

        Returns:
            None
        """
        self.inner[i].update(n)

    def update_outer(self, n: int = 1) -> None:
        """
        Update the outer progress bar by n steps.

        Args:
            n (int): The number of steps to update the progress bar by.

        Returns:
            None
        """
        self.outer.update(n)

    def close(self) -> None:
        """
        Close all progress bars.

        Returns:
            None
        """
        for bar in self.inner:
            bar.close()

        self.outer.close()


# Global progress manager
_manager: ProgressManager | None = None


def initialize_pbar(participant_details: list[ParticipantDetails]) -> None:
    """
    Function to initialize the global progress manager with participant details.

    Args:
        participant_details (list[ParticipantDetails]): A list of participant details.
    Returns:
        None
    """
    global _manager

    _manager = ProgressManager(participant_details)


def get_pbar_manager() -> ProgressManager:
    """
    Function to get the global progress manager to be used for API calls.

    Returns:
        ProgressManager: The global progress manager.
    """
    if _manager is None:
        raise RuntimeError(
            "ProgressManager has not been initialized. "
            "Call progress.initialize_pbar() first."
        )

    return _manager
