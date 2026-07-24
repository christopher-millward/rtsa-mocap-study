from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[0]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw_normalized_data"

ORTHONORMAL_TOLERANCE = 5e-4
DETERMINANT_TOLERANCE = 5e-4
SMALLEST_CLINICALLY_RELEVANT_ANGLE = 1e-3

RESULTS_PATH = PROJ_ROOT / "outputs" / "results.csv"
