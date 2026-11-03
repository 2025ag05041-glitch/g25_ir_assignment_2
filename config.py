"""Project-wide paths and default settings."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "Dataset"
METADATA_DIR = ROOT_DIR / "data" / "metadata"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
CATEGORY_MAPPING_FILE = METADATA_DIR / "category_mapping.csv"
CORPUS_METADATA_FILE = PROCESSED_DIR / "corpus_metadata.csv"
RESULTS_DIR = ROOT_DIR / "reports" / "results"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

RANDOM_SEED = 42

# Documents with this label are kept in the corpus (dashboard, search,
# recommendation) but left out of classification / clustering evaluation
# because only two such documents exist.
OFF_DOMAIN_LABEL = "General (Off-domain)"

CATEGORIES = [
    "Machine Learning",
    "Language & Search Technology",
    "AI Systems & Cognition",
    "Computing & Infrastructure",
    "Finance & Economics",
    OFF_DOMAIN_LABEL,
]

# Fixed colour per category so a category keeps its colour on every chart.
CATEGORY_COLORS = {
    "Machine Learning": "#2a78d6",
    "Language & Search Technology": "#eb6834",
    "AI Systems & Cognition": "#1baf7a",
    "Computing & Infrastructure": "#eda100",
    "Finance & Economics": "#e87ba4",
    OFF_DOMAIN_LABEL: "#8a8985",
}
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                 "#008300", "#6250d6", "#e34948"]

# Default experiment settings
TEST_SIZE = 0.3
NB_ALPHA = 1.0
DEFAULT_MIN_DF = 2
NEAR_DUPLICATE_JACCARD = 0.5
NEAR_DUPLICATE_COSINE = 0.9
