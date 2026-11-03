"""Create the submission ZIP (G25_Assignment2.zip) in the project folder.

Usage: python scripts/make_submission.py
Leaves out the virtual environment, caches, git data and the assignment brief.
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZIP_NAME = "G25_Assignment2.zip"
TOP_FOLDER = "G25_Assignment2"

INCLUDE = ["app.py", "config.py", "requirements.txt", "README.md", ".streamlit",
           "src", "views", "scripts", "tests", "Dataset", "data", "reports", "evidence",
           "G025_Contribution.xlsx"]
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".venv", ".git"}

REQUIRED = ["app.py", "requirements.txt", "README.md", "G025_Contribution.xlsx",
            "reports/G25_IR_Assignment2_Report.pdf", "data/metadata/category_mapping.csv",
            "data/processed/corpus_metadata.csv"]


def files_to_add():
    for name in INCLUDE:
        path = ROOT / name
        if not path.exists():
            print("WARNING: missing", name)
            continue
        paths = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
        for p in paths:
            if not SKIP_PARTS & set(p.relative_to(ROOT).parts) and p.suffix != ".pyc":
                yield p


def main():
    out = ROOT / ZIP_NAME
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in files_to_add():
            zf.write(p, f"{TOP_FOLDER}/{p.relative_to(ROOT).as_posix()}")

    with zipfile.ZipFile(out) as zf:
        bad = zf.testzip()
        names = zf.namelist()
    missing = [r for r in REQUIRED if f"{TOP_FOLDER}/{r}" not in names]
    datasets = sum(1 for n in names if n.startswith(f"{TOP_FOLDER}/Dataset/") and n.endswith(".txt"))
    vlab = [n for n in names if "/evidence/virtual_lab/" in n and n.lower().endswith((".png", ".jpg", ".jpeg"))]
    videos = [n for n in names if n.lower().endswith((".mp4", ".mkv", ".webm", ".mov", ".avi"))]
    print(f"Created {out.name}: {len(names)} files, {out.stat().st_size / 1e6:.1f} MB, "
          f"{datasets} dataset documents")
    print("CRC check:", "OK" if bad is None else f"corrupt file {bad}")
    print("Required files missing:", missing or "none")
    print(f"Virtual Lab screenshots: {len(vlab)}  |  screen recordings: {len(videos)}")
    if not vlab or not videos:
        print("REMINDER: add the BITS Virtual Lab screenshots and screen recording (see README) "
              "and run this script again before submitting.")


if __name__ == "__main__":
    main()
