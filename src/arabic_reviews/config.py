"""Project paths, data-source metadata and label vocabularies."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT / "reports"

SOURCE = {
    "title": "Arabic Café Reviews (Multi-Aspect Sentiment Analysis of Arabic Café Reviews)",
    "creator": "Hmood Al-Dossari",
    "record": "https://zenodo.org/records/17781916",
    "doi": "10.5281/zenodo.17781916",
    "version": "1",
    "publication_date": "2025-12-01",
    "license": "CC-BY-4.0",
    "filename": "cleaned01_fixed.xlsx",
    "url": "https://zenodo.org/records/17781916/files/cleaned01_fixed.xlsx?download=1",
    "size_bytes": 125720,
    "md5": "9b546b96eb7d16a374f648b4dc840d59",
    "sha256": "f171ef057abcae5946106534c8ef26cc9ef32a9d94fb8ec3e1843898f8f6d901",
}
RAW_FILE = RAW_DIR / SOURCE["filename"]
MANIFEST_FILE = DATA_DIR / "source_manifest.json"
VALIDATED_FILE = PROCESSED_DIR / "validated.csv"
QUALITY_REPORT = REPORTS_DIR / "data_quality.md"
LINEAGE_FILE = REPORTS_DIR / "lineage.json"

# Raw column -> project name.
RAW_COLUMNS = {"Review": "text", "Attribute": "aspect", "Result": "sentiment"}
EXPECTED_ROWS = 3000
ASPECTS = ("المأكولات", "المشروبات", "الخدمة", "الجلسات", "الاسعار")
SENTIMENTS = ("ايجابي", "سلبي", "محايد")

# Split (M2). Only the text-free assignment file is published.
SPLITS_DIR = DATA_DIR / "splits"
SPLIT_FILE = SPLITS_DIR / "split_v1.csv"
DEV_FILE = PROCESSED_DIR / "dev.csv"
TEST_FILE = PROCESSED_DIR / "test.csv"
SPLIT_REPORT = REPORTS_DIR / "split.md"
NEAR_DUPLICATES_FILE = REPORTS_DIR / "near_duplicates.csv"
EDA_REPORT = REPORTS_DIR / "eda.md"
FIGURES_DIR = REPORTS_DIR / "figures"
