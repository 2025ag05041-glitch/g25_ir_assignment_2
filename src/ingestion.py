"""Loading the base corpus and uploaded documents into one table.

Document text and metadata are kept separate on disk: the raw text stays in
Dataset/*.txt and the metadata (IDs, labels, source) is written to
data/processed/corpus_metadata.csv.
"""
import io
import json
import re

import pandas as pd

from config import (CATEGORY_MAPPING_FILE, CORPUS_METADATA_FILE, DATASET_DIR,
                    OFF_DOMAIN_LABEL)

METADATA_COLUMNS = ["doc_id", "title", "category", "file_name", "source",
                    "source_url", "revision_id", "char_count", "word_count"]

RETRIEVED_PATTERN = re.compile(r'Retrieved from\s+"(.*?)"', re.S)


def read_text(raw_bytes):
    """Decode bytes as UTF-8 (dropping a BOM) and normalise line endings."""
    text = raw_bytes.decode("utf-8-sig", errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def parse_wikipedia_text(text):
    """Split a Wikipedia text export into title, body and source info.

    The first line is the article title and the last line is the
    'Retrieved from "<url>"' footer, which we move into metadata.
    """
    title = text.strip().split("\n", 1)[0].strip()
    source_url, revision_id = "", ""
    match = RETRIEVED_PATTERN.search(text)
    if match:
        # the URL is sometimes broken across lines in the export
        source_url = re.sub(r"\s+", "", match.group(1))
        rev = re.search(r"oldid=(\d+)", source_url)
        revision_id = rev.group(1) if rev else ""
        text = text[:match.start()]
    return title, text.strip(), source_url, revision_id


def load_category_mapping():
    mapping = pd.read_csv(CATEGORY_MAPPING_FILE)
    return dict(zip(mapping["file_name"], mapping["category"]))


def load_base_corpus():
    """Read every .txt file in Dataset/ and attach IDs, labels and metadata."""
    mapping = load_category_mapping()
    records = []
    for i, path in enumerate(sorted(DATASET_DIR.glob("*.txt")), start=1):
        text = read_text(path.read_bytes())
        title, body, url, rev = parse_wikipedia_text(text)
        records.append({
            "doc_id": f"DOC{i:03d}",
            "title": title or path.stem.replace("_", " "),
            "category": mapping.get(path.name, "Unlabelled"),
            "file_name": path.name,
            "source": "Wikipedia",
            "source_url": url,
            "revision_id": rev,
            "text": body,
        })
    return add_length_columns(pd.DataFrame(records))


def add_length_columns(df):
    df = df.copy()
    df["char_count"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len().fillna(0).astype(int)
    return df


def save_metadata(df, path=CORPUS_METADATA_FILE):
    """Write metadata only (no text) so text and metadata stay separate."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df[METADATA_COLUMNS].to_csv(path, index=False)


def is_evaluation_document(category):
    return category not in (OFF_DOMAIN_LABEL, "Unlabelled", "", None)


# ---------------------------------------------------------------- uploads

def parse_upload(file_name, raw_bytes, default_category="Unlabelled"):
    """Turn one uploaded file (.txt, .csv or .json) into a list of records.

    Returns (records, warnings). CSV/JSON need a 'text' column/field and may
    have 'title', 'category', 'doc_id' and 'source'.
    """
    name = file_name.lower()
    warnings = []
    if name.endswith(".txt"):
        title, body, url, rev = parse_wikipedia_text(read_text(raw_bytes))
        rows = [{"title": title, "text": body, "source_url": url,
                 "revision_id": rev, "category": default_category}]
    elif name.endswith(".csv"):
        rows = pd.read_csv(io.BytesIO(raw_bytes)).to_dict("records")
    elif name.endswith(".json"):
        data = json.loads(read_text(raw_bytes))
        rows = data if isinstance(data, list) else data.get("documents", [])
    else:
        return [], [f"{file_name}: unsupported file type (use .txt, .csv or .json)"]

    records = []
    for n, row in enumerate(rows, start=1):
        text = str(row.get("text", "") or "").strip()
        if not text or text.lower() == "nan":
            warnings.append(f"{file_name} row {n}: empty text, skipped")
            continue
        category = row.get("category") or default_category
        records.append({
            "doc_id": str(row.get("doc_id", "") or "").strip(),
            "title": str(row.get("title") or f"{file_name} #{n}").strip(),
            "category": str(category).strip(),
            "file_name": file_name,
            "source": str(row.get("source") or "Uploaded"),
            "source_url": str(row.get("source_url") or ""),
            "revision_id": str(row.get("revision_id") or ""),
            "text": text,
        })
    if not records and not warnings:
        warnings.append(f"{file_name}: no documents found")
    return records, warnings


def merge_uploads(base_df, uploaded_records):
    """Append uploaded records, giving each one a unique ID (UPL001, ...)."""
    if not uploaded_records:
        return base_df
    used = set(base_df["doc_id"])
    counter = 1
    rows = []
    for rec in uploaded_records:
        rec = dict(rec)
        if not rec["doc_id"] or rec["doc_id"] in used:
            while f"UPL{counter:03d}" in used:
                counter += 1
            rec["doc_id"] = f"UPL{counter:03d}"
        used.add(rec["doc_id"])
        rows.append(rec)
    merged = pd.concat([base_df, pd.DataFrame(rows)], ignore_index=True)
    return add_length_columns(merged)


def corpus_statistics(df):
    words = df["word_count"]
    return {
        "documents": len(df),
        "categories": df["category"].nunique(),
        "total_words": int(words.sum()),
        "total_mb": df["char_count"].sum() / (1024 * 1024),
        "avg_words": float(words.mean()) if len(df) else 0.0,
        "min_words": int(words.min()) if len(df) else 0,
        "max_words": int(words.max()) if len(df) else 0,
        "empty_documents": int((words == 0).sum()),
    }
