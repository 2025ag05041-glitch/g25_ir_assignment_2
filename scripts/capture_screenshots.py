"""Capture screenshots of the running app for the report.

1. Start the app:       streamlit run app.py --server.port 8501
2. In another terminal: python scripts/capture_screenshots.py [url] [output_folder]

Needs `pip install playwright` and either `playwright install chromium`
or Google Chrome installed (used through channel="chrome").
"""
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[1] / "reports" / "screenshots"

# (file name, sidebar page, tab or None, viewport height)
SHOTS = [
    ("01_dashboard", "Dashboard", None, 1250),
    ("02_dataset_browse", "Dataset & Upload", "Browse documents", 1300),
    ("03_duplicate_analysis", "Dataset & Upload", "Duplicate analysis", 1500),
    ("04_preprocessing", "Preprocessing & Features", None, 1900),
    ("05_naive_bayes", "Text Mining (NB & K-Means)", "Naive Bayes classification", 1700),
    ("06_kmeans", "Text Mining (NB & K-Means)", "K-Means clustering", 1900),
    ("07_recommendations", "Recommendations", "Related documents", 1500),
    ("08_search_by_text", "Recommendations", "Search by text", 900),
    ("09_perf_classification", "Performance Analysis", "Classification", 1500),
    ("10_perf_clustering", "Performance Analysis", "Clustering", 1700),
    ("11_perf_comparison", "Performance Analysis", "Comparison", 1300),
    ("12_perf_experiments", "Performance Analysis", "Experiments", 1100),
]


def wait_idle(page, timeout=240):
    start, quiet = time.time(), 0
    while time.time() - start < timeout:
        busy = (page.locator('[data-testid="stStatusWidget"]').count()
                + page.locator('[data-testid="stSpinner"]').count())
        quiet = quiet + 1 if not busy else 0
        if quiet >= 4:
            return
        time.sleep(0.5)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
        except Exception:
            browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.goto(BASE)
        wait_idle(page)
        for name, nav, tab, height in SHOTS:
            page.set_viewport_size({"width": 1400, "height": height})
            page.locator('[data-testid="stSidebarNav"]').get_by_text(nav, exact=True).click()
            time.sleep(1.5)
            wait_idle(page)
            if tab:
                page.get_by_role("tab", name=tab).click()
                time.sleep(1.5)
                wait_idle(page)
            time.sleep(1)
            page.screenshot(path=str(OUT / f"{name}.png"))
            print("saved", name)
        browser.close()


if __name__ == "__main__":
    main()
