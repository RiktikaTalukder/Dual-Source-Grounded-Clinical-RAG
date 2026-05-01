"""
pmc_parser.py
Parses PMC Open Access XML articles (JATS format) into clean JSON files.

Input:  data/pmc_literature/xml/        — directory containing .xml files
Output: data/pmc_literature/pmc_sample/ — one .json file per article, max 500 articles

Each JSON contains:
  pmc_id          — PMC article ID (string)
  title           — article title (string)
  abstract        — full abstract text (string)
  body_paragraphs — list of body paragraph strings

Usage:
  python src/pmc_parser.py
"""

import os
import json
import glob
from lxml import etree
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────
RAW_DIR      = "data/pmc_literature/xml/"
OUT_DIR      = "data/pmc_literature/pmc_sample/"
MAX_ARTICLES = 500
MIN_PARA_LEN = 40
CLINICAL_KEYWORDS = [
    "patient", "clinical trial", "randomized controlled",
    "hospital admission", "icu", "intensive care",
    "mortality", "treatment outcome", "adverse event",
    "cohort study", "diagnosis", "therapeutic",
    "pharmacotherapy", "comorbidity", "sepsis",
    "heart failure", "diabetes", "hypertension",
    "renal", "cardiac", "pulmonary"
]

# Require at least 2 keywords to match (not just any 1)
MIN_KEYWORD_MATCHES = 2
# ─────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)


def get_text(element):
    if element is None:
        return ""
    parts = [t for t in element.itertext() if t.strip()]
    return " ".join(parts).strip()


def extract_pmc_id(root):
    for el in root.iter("article-id"):
        if el.get("pub-id-type", "") == "pmc" and el.text:
            raw = el.text.strip()
            return raw if raw.startswith("PMC") else "PMC" + raw
    return None


def extract_title(root):
    title_el = root.find(".//article-title")
    if title_el is not None:
        text = get_text(title_el)
        if text:
            return text

    tg = root.find(".//title-group")
    if tg is not None:
        for child in tg.iter("article-title"):
            text = get_text(child)
            if text:
                return text

    return ""


def extract_abstract(root):
    abstract_el = root.find(".//abstract")
    if abstract_el is None:
        return ""

    paragraphs = []
    for p in abstract_el.iter("p"):
        text = get_text(p)
        if text:
            paragraphs.append(text)

    if paragraphs:
        return " ".join(paragraphs)

    return get_text(abstract_el)


def extract_body_paragraphs(root):
    body_el = root.find(".//body")
    if body_el is None:
        return []

    paragraphs = []

    for p in body_el.iter("p"):
        parent = p.getparent()
        skip = False
        while parent is not None:
            tag = parent.tag.lower() if parent.tag else ""
            if tag in ("fig", "table-wrap", "supplementary-material"):
                skip = True
                break
            parent = parent.getparent()
        if skip:
            continue

        text = get_text(p)
        if len(text) >= MIN_PARA_LEN:
            paragraphs.append(text)

    return paragraphs


def parse_one_xml(xml_path):
    try:
        tree = etree.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        tqdm.write(f"  SKIP (parse error): {os.path.basename(xml_path)} — {e}")
        return None

    pmc_id = extract_pmc_id(root)

    if not pmc_id:
        pmc_id = os.path.splitext(os.path.basename(xml_path))[0]

    title      = extract_title(root)
    abstract   = extract_abstract(root)
    body_paras = extract_body_paragraphs(root)

    if not title and not abstract and not body_paras:
        return None

    return {
        "pmc_id":           pmc_id,
        "title":            title,
        "abstract":         abstract,
        "body_paragraphs":  body_paras,
    }


def parse_all():
    xml_files = glob.glob(
        os.path.join(RAW_DIR, "**", "*.xml"), recursive=True
    )

    if len(xml_files) == 0:
        print(f"ERROR: No XML files found in {RAW_DIR}")
        print("Make sure you ran download_pmc.py first.")
        return

    print(f"Found {len(xml_files):,} XML files in {RAW_DIR}")
    print(f"Will parse up to {MAX_ARTICLES} articles.\n")

    saved   = 0
    skipped = 0

    for xml_path in tqdm(xml_files, desc="Parsing articles"):
        if saved >= MAX_ARTICLES:
            break

        article = parse_one_xml(xml_path)

        if article is None:
            skipped += 1
            continue

# ── Clinical keyword filter (strict: require 2+ matches) ──────
        combined_text = (article["title"] + " " + article["abstract"]).lower()
        matches = sum(1 for kw in CLINICAL_KEYWORDS if kw in combined_text)
        if matches < MIN_KEYWORD_MATCHES:
            skipped += 1
            continue
        # ─────────────────────────────────────────────────────────────
        out_filename = article["pmc_id"] + ".json"
        out_path = os.path.join(OUT_DIR, out_filename)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(article, f, indent=2, ensure_ascii=False)

        saved += 1

    print(f"\nDone!")
    print(f"  Saved:   {saved} articles → {OUT_DIR}")
    print(f"  Skipped: {skipped} articles (no content / parse errors / non-clinical)")

    if saved < MAX_ARTICLES:
        print(f"\nWARNING: Only saved {saved} articles (needed {MAX_ARTICLES}).")
        print("Download more archives with download_pmc.py (increase NUM_ARCHIVES).")


if __name__ == "__main__":
    parse_all()