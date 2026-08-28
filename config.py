from pathlib import Path
# Directories
PROJ_ROOT = Path(__file__).resolve().parents[0]
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw_normalized_data"
OUTPUTS_DIR = PROJ_ROOT / "outputs"

# Specific files
RAW_PARTICIPANT_DETAILS_PATH = RAW_DATA_DIR / "participant_details.xlsx"
RESULTS_PICKLE_PATH = OUTPUTS_DIR / "all_results.pkl"
RESULTS_HEATMAP_CSV_PATH = OUTPUTS_DIR / "heatmap_results_data.csv"

CUMULATIVE_MOTION_RAINCLOUD_PATH = OUTPUTS_DIR / "cumulative_motion_raincloud.png"
OPERATED_CUMULATIVE_MOTION_HEATMAP_PATH = OUTPUTS_DIR / "operated_cumulative_motion_heatmap.png"

# Analysis and Testing Parameters
ORTHONORMAL_TOLERANCE = 5e-4
DETERMINANT_TOLERANCE = 5e-4
SMALLEST_CLINICALLY_RELEVANT_ANGLE = 1e-3
TEST_PRECISION_TOLERANCE = 1e-9
TEST_SINGULARITY_TOLERANCE = 1e-7

