"""
PDF Dictionary Extraction Module for Africa AI Project.

Extracts vocabulary, parts of speech, definitions, and examples from:
- Ibibio Dictionary (Kaufman, 1972)
- Igbo Dictionary (Williamson & Blench, 2nd ed.)
- Bini Dictionary (Melzian, 1937)

Outputs structured JSON entries matching:
{
    "language": "Ibibio" | "Igbo" | "Bini",
    "headword": str,
    "pos": str | None,
    "definition": str,
    "examples": list[str]
}
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pymupdf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Known POS abbreviations across dictionaries
KNOWN_POS_SET = {
    # Standard POS with dot
    "n.", "v.", "adj.", "adv.", "prep.", "conj.", "pron.", "num.", "p.n.", "n.p.", "v.p.",
    "aux. v.", "ext. suff.", "infl. suff.", "int.", "dem.", "poss.", "quant.", "enc.",
    "ideo.", "interj.", "art.", "part.", "pref.", "suff.",
    # Bini specific
    "v.tr.", "v.i.", "v.i. & v.tr.", "v.tr. & v.i.", "n. & v.", "adj. & adv.",
    # Ibibio specific (often without dot in Kaufman)
    "n", "v", "tv", "iv", "iv, tv", "tv, iv", "aj", "adv", "ideo", "interj", "inter]",
    "conj", "prep", "pron", "num", "aj, adv", "adv, aj"
}

# Mapping to canonical POS forms
POS_CANONICAL_MAP = {
    "n": "n.",
    "n.": "n.",
    "v": "v.",
    "v.": "v.",
    "tv": "v.tr.",
    "iv": "v.i.",
    "iv, tv": "v.",
    "tv, iv": "v.",
    "v.tr.": "v.tr.",
    "v.i.": "v.i.",
    "v.i. & v.tr.": "v.",
    "v.tr. & v.i.": "v.",
    "aj": "adj.",
    "adj.": "adj.",
    "adv": "adv.",
    "adv.": "adv.",
    "aj, adv": "adj./adv.",
    "adv, aj": "adv./adj.",
    "prep": "prep.",
    "prep.": "prep.",
    "conj": "conj.",
    "conj.": "conj.",
    "pron": "pron.",
    "pron.": "pron.",
    "num": "num.",
    "num.": "num.",
    "p.n.": "p.n.",
    "n.p.": "n.p.",
    "v.p.": "v.p.",
    "aux. v.": "aux. v.",
    "ext. suff.": "ext. suff.",
    "infl. suff.": "infl. suff.",
    "int.": "interj.",
    "interj": "interj.",
    "interj.": "interj.",
    "inter]": "interj.",
    "dem.": "dem.",
    "poss.": "poss.",
    "quant.": "quant.",
    "enc.": "enc.",
    "ideo": "ideo.",
    "ideo.": "ideo."
}


def normalize_pos(raw_pos: Optional[str]) -> Optional[str]:
    """Normalize raw POS string to canonical format."""
    if not raw_pos:
        return None
    cleaned = raw_pos.strip()
    return POS_CANONICAL_MAP.get(cleaned.lower(), cleaned)


def clean_page_text(text: str, language: str) -> str:
    """
    Remove headers, footers, page numbers and non-lexical artifacts.
    """
    if not text:
        return ""

    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # Common page numbers (digits or roman numerals)
        if re.match(r"^\d+$", line):
            continue
        if re.match(r"^[ivxlcdmIVXLCDM]+$", line):
            continue

        if language.lower() == "igbo":
            # Running header in Williamson Igbo dictionary
            if "Igbo Dictionary: KayWilliamson" in line:
                continue
            if line == "IGBO DICTIONARY":
                continue
            # Single letter alphabet section markers e.g. "A." or "B."
            if re.match(r"^[A-Z]\.\s*$", line):
                continue

        elif language.lower() == "ibibio":
            # Running headers/footers in Kaufman dictionary
            if "DOCUMENT RESUME" in line or "FINAL REPORT" in line:
                continue
            if "IBIBIO DICTIONARY" in line:
                continue

        elif language.lower() == "bini":
            if "A CONCISE DICTIONARY OF THE BINI LANGUAGE" in line.upper():
                continue

        lines.append(line)

    return "\n".join(lines)


def is_igbo_pos_line(line: str) -> bool:
    """Check if a line represents an Igbo POS or cross-reference."""
    cleaned = line.lower().strip()
    if cleaned in KNOWN_POS_SET:
        return True
    if re.match(r"^(n|v|adj|adv|num|prep|conj|pron|p\.n\.|n\.p\.|v\.p\.|aux\. v\.|ext\. suff\.|infl\. suff\.|int\.|dem\.|poss\.|quant\.|enc\.|ideo\.)\b", cleaned):
        return True
    if re.match(r"^(see|cf\.)\s+", cleaned):
        return True
    return False


def parse_igbo_text(text: str) -> List[Dict[str, Any]]:
    """
    Parse cleaned text from the Igbo Dictionary.
    Handles headwords, multi-line definitions, subentries, and examples.
    """
    cleaned = clean_page_text(text, "Igbo")
    lines = [l.strip() for l in cleaned.splitlines() if l.strip()]
    
    entries: List[Dict[str, Any]] = []
    i = 0
    n = len(lines)
    current_entry: Optional[Dict[str, Any]] = None

    while i < n:
        line = lines[i]
        next_line = lines[i + 1] if i + 1 < n else ""
        next_is_pos = is_igbo_pos_line(next_line)

        # Check if line is a headword followed by a POS line
        if next_is_pos and len(line.split()) <= 6 and not line.startswith("-"):
            if current_entry:
                entries.append(current_entry)

            raw_pos = next_line.strip()
            pos = normalize_pos(raw_pos) if raw_pos.lower() in KNOWN_POS_SET else None
            
            # Start collecting definition
            def_lines = []
            examples = []
            
            if raw_pos.lower().startswith("see ") or raw_pos.lower().startswith("cf."):
                def_lines.append(raw_pos)

            i += 2
            while i < n:
                curr = lines[i]
                next_l = lines[i + 1] if i + 1 < n else ""
                
                # If next pair is a new headword + POS
                if is_igbo_pos_line(next_l) and len(curr.split()) <= 6 and not curr.startswith("-"):
                    break
                
                # Check for example format or subentries
                if ":" in curr:
                    parts = curr.split(":", 1)
                    if parts[0].strip():
                        def_lines.append(parts[0].strip())
                    if parts[1].strip():
                        examples.append(parts[1].strip())
                elif curr.startswith("-") or (current_entry and len(def_lines) > 0 and len(curr.split()) >= 4 and any(w in curr for w in ["He ", "She ", "It ", "They ", "We ", "This ", "That ", "The "])):
                    # Phrasal subentry or translated example
                    examples.append(curr)
                else:
                    def_lines.append(curr)
                i += 1

            full_def = " ".join(def_lines).strip()
            # Clean headword (remove trailing numbers e.g. 'akpụ 1.' -> 'akpụ 1.')
            current_entry = {
                "language": "Igbo",
                "headword": line,
                "pos": pos,
                "definition": full_def if full_def else (raw_pos if not pos else ""),
                "examples": examples
            }
        else:
            # Subentry or continuation line
            if current_entry:
                if line.startswith("-") or any(w in line for w in ["He ", "She ", "It ", "They ", "We ", "This ", "The "]):
                    current_entry["examples"].append(line)
                elif ":" in line:
                    parts = line.split(":", 1)
                    if parts[0].strip():
                        current_entry["definition"] = (current_entry["definition"] + " " + parts[0].strip()).strip()
                    if parts[1].strip():
                        current_entry["examples"].append(parts[1].strip())
                else:
                    if current_entry["definition"]:
                        current_entry["definition"] += " " + line
                    else:
                        current_entry["definition"] = line
            i += 1

    if current_entry:
        entries.append(current_entry)

    return entries


def is_ibibio_pos_line(line: str) -> bool:
    """Check if a line represents an Ibibio POS or cross-reference."""
    cleaned = line.lower().strip().rstrip(".")
    if cleaned in {p.rstrip(".") for p in KNOWN_POS_SET}:
        return True
    if re.match(r"^(n|v|tv|iv|aj|adv|num|ideo|conj|prep|interj|inter\])\b", cleaned):
        return True
    if re.match(r"^(see|cf|var of|pl|syn)\b", cleaned):
        return True
    return False


def parse_ibibio_text(text: str) -> List[Dict[str, Any]]:
    """
    Parse cleaned text from the Ibibio Dictionary.
    Handles 2-column formatting artifacts, tone markings, and subentries.
    """
    cleaned = clean_page_text(text, "Ibibio")
    lines = [l.strip() for l in cleaned.splitlines() if l.strip()]

    entries: List[Dict[str, Any]] = []
    i = 0
    n = len(lines)
    current_entry: Optional[Dict[str, Any]] = None

    while i < n:
        line = lines[i]
        next_line = lines[i + 1] if i + 1 < n else ""
        next_is_pos = is_ibibio_pos_line(next_line)

        # In Ibibio, headwords are single words or short phrases, not starting with hyphen or 'lit'
        is_potential_headword = (
            next_is_pos 
            and not line.startswith("-") 
            and not line.lower().startswith("lit ")
            and not line.lower().startswith("syn ")
            and len(line.split()) <= 5
        )

        if is_potential_headword:
            if current_entry:
                entries.append(current_entry)

            raw_pos = next_line.strip()
            pos_match = re.match(r"^(\.?\b[a-zA-Z, ]+\b)", raw_pos)
            pos = None
            if pos_match:
                candidate = pos_match.group(1).strip(". ")
                pos = normalize_pos(candidate)

            def_lines = []
            examples = []

            # Check if next_line has additional info like "n var of..." or "see..."
            if raw_pos.startswith("see ") or raw_pos.startswith("var of ") or raw_pos.startswith("pl "):
                def_lines.append(raw_pos)
            elif pos_match and len(raw_pos) > len(pos_match.group(0)):
                extra = raw_pos[len(pos_match.group(0)):].strip(" .")
                if extra:
                    def_lines.append(extra)

            i += 2
            while i < n:
                curr = lines[i]
                next_l = lines[i + 1] if i + 1 < n else ""

                if (
                    is_ibibio_pos_line(next_l) 
                    and not curr.startswith("-") 
                    and not curr.lower().startswith("lit ")
                    and not curr.lower().startswith("syn ")
                    and len(curr.split()) <= 5
                ):
                    break

                if (
                    curr.startswith("-") 
                    or curr.lower().startswith("lit ") 
                    or curr.lower().startswith("syn ") 
                    or curr.lower().startswith("cf ") 
                    or "." in curr and any(c.isupper() for c in curr)
                ):
                    examples.append(curr)
                else:
                    def_lines.append(curr)
                i += 1

            full_def = " ".join(def_lines).strip()
            current_entry = {
                "language": "Ibibio",
                "headword": line,
                "pos": pos,
                "definition": full_def,
                "examples": examples
            }
        else:
            if current_entry:
                if (
                    line.startswith("-") 
                    or line.lower().startswith("lit ") 
                    or line.lower().startswith("syn ") 
                    or line.lower().startswith("cf ")
                ):
                    current_entry["examples"].append(line)
                else:
                    if current_entry["definition"]:
                        current_entry["definition"] += " " + line
                    else:
                        current_entry["definition"] = line
            i += 1

    if current_entry:
        entries.append(current_entry)

    return entries


def parse_bini_text(text: str) -> List[Dict[str, Any]]:
    """
    Parse text formatted according to Melzian's Bini Dictionary format:
    e.g. 'ágbà, n., jaw, chin.' or multi-line dictionary text.
    """
    cleaned = clean_page_text(text, "Bini")
    lines = [l.strip() for l in cleaned.splitlines() if l.strip()]

    entries: List[Dict[str, Any]] = []
    
    # Regex for standard Melzian line: <headword>, <pos>, <definition>
    melzian_pattern = re.compile(
        r"^([^\s,]+(?: [^\s,]+)?),\s*([a-z\.\s&,]+?),\s*(.+)$",
        re.IGNORECASE
    )

    current_entry: Optional[Dict[str, Any]] = None

    for line in lines:
        match = melzian_pattern.match(line)
        if match:
            if current_entry:
                entries.append(current_entry)

            headword = match.group(1).strip()
            raw_pos = match.group(2).strip()
            rest = match.group(3).strip()

            pos = normalize_pos(raw_pos)

            # Separate definition from examples/notes if present (separated by semicolon)
            examples = []
            if ";" in rest:
                parts = rest.split(";")
                definition = parts[0].strip()
                for p in parts[1:]:
                    p_clean = p.strip()
                    if p_clean:
                        examples.append(p_clean)
            else:
                definition = rest

            current_entry = {
                "language": "Bini",
                "headword": headword,
                "pos": pos,
                "definition": definition,
                "examples": examples
            }
        else:
            # Continuation line or subentry
            if current_entry:
                if line.startswith("-") or ";" in line:
                    current_entry["examples"].append(line)
                else:
                    current_entry["definition"] = (current_entry["definition"] + " " + line).strip()
            elif "," in line:
                # Fallback comma-separated line
                parts = [p.strip() for p in line.split(",", 2)]
                if len(parts) >= 2:
                    current_entry = {
                        "language": "Bini",
                        "headword": parts[0],
                        "pos": normalize_pos(parts[1]) if parts[1].lower() in KNOWN_POS_SET else None,
                        "definition": parts[2] if len(parts) > 2 else parts[1],
                        "examples": []
                    }

    if current_entry:
        entries.append(current_entry)

    return entries


def parse_dictionary_text(language: str, text: str) -> List[Dict[str, Any]]:
    """
    Dispatch text parsing to the appropriate language parser.
    """
    lang = language.lower()
    if lang == "igbo":
        return parse_igbo_text(text)
    elif lang == "ibibio":
        return parse_ibibio_text(text)
    elif lang == "bini":
        return parse_bini_text(text)
    else:
        raise ValueError(f"Unsupported language: {language}. Must be Igbo, Ibibio, or Bini.")


def extract_from_pdf(
    pdf_path: str,
    language: str,
    start_page: int = 0,
    limit_pages: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Extract structured dictionary entries from a PDF file using PyMuPDF (fitz).
    Handles column-based layouts for Kaufman Ibibio dictionary and standard layouts.
    """
    if not os.path.exists(pdf_path):
        logger.warning("PDF file not found: %s", pdf_path)
        return []

    logger.info("Opening PDF: %s (Language: %s)", pdf_path, language)
    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)
    
    end_page = total_pages
    if limit_pages is not None:
        end_page = min(start_page + limit_pages, total_pages)

    logger.info("Extracting pages %d to %d of %d...", start_page + 1, end_page, total_pages)

    full_text = ""
    for page_idx in range(start_page, end_page):
        page = doc[page_idx]
        
        # For Kaufman Ibibio Dictionary, extract column by column to prevent horizontal OCR interleaving
        if language.lower() == "ibibio":
            rect = page.rect
            mid_x = rect.width / 2.0
            col1 = page.get_text(clip=pymupdf.Rect(0, 0, mid_x, rect.height))
            col2 = page.get_text(clip=pymupdf.Rect(mid_x, 0, rect.width, rect.height))
            full_text += col1 + "\n" + col2 + "\n"
        else:
            full_text += page.get_text() + "\n"

    doc.close()

    if not full_text.strip():
        logger.warning(
            "No extractable digital text found in %s (likely scanned images without embedded OCR layer).",
            pdf_path
        )
        return []

    entries = parse_dictionary_text(language, full_text)
    logger.info("Extracted %d entries for %s.", len(entries), language)
    return entries


def extract_all_dictionaries(
    pdf_dir: str,
    output_path: str,
    limit_pages: Optional[int] = None
) -> Dict[str, Any]:
    """
    Extract all dictionary PDFs found in pdf_dir and save structured JSON to output_path.
    """
    pdf_dir_path = Path(pdf_dir)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Configuration for each dictionary
    dictionary_configs = [
        {
            "language": "Igbo",
            "filename": "Igbo Dictionary.pdf",
            "start_page": 33,  # Dictionary body starts after introductory preface/bibliography
        },
        {
            "language": "Ibibio",
            "filename": "IBIBIO DICTIONARY.pdf",
            "start_page": 20,  # Dictionary body starts after ERIC report header and intro
        },
        {
            "language": "Bini",
            "filename": "Melzian - A Concise Dictionary of the Bini Language of Southern Nigeria (1937).pdf",
            "start_page": 0,
        }
    ]

    all_entries: List[Dict[str, Any]] = []
    stats: Dict[str, int] = {}

    for config in dictionary_configs:
        lang = config["language"]
        pdf_path = str(pdf_dir_path / config["filename"])
        
        entries = extract_from_pdf(
            pdf_path=pdf_path,
            language=lang,
            start_page=config["start_page"],
            limit_pages=limit_pages
        )
        all_entries.extend(entries)
        stats[lang] = len(entries)

    # Save to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, ensure_ascii=False, indent=2)

    logger.info("Saved %d total entries to %s", len(all_entries), output_file)
    logger.info("Extraction statistics: %s", stats)

    return {
        "total_entries": len(all_entries),
        "by_language": stats,
        "output_path": str(output_file)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract African language dictionaries (Igbo, Ibibio, Bini) from PDF files."
    )
    parser.add_argument(
        "--pdf-dir",
        default="dictionaries_raw",
        help="Directory containing the dictionary PDFs (default: dictionaries_raw)"
    )
    parser.add_argument(
        "--output",
        default="data/raw/dictionary_extracted.json",
        help="Output JSON file path (default: data/raw/dictionary_extracted.json)"
    )
    parser.add_argument(
        "--limit-pages",
        type=int,
        default=None,
        help="Limit number of pages to process per dictionary (useful for fast testing)"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Quick sample extraction mode (processes 15 pages per dictionary)"
    )
    parser.add_argument(
        "--language",
        choices=["Igbo", "Ibibio", "Bini", "all"],
        default="all",
        help="Specific language dictionary to extract (default: all)"
    )

    args = parser.parse_args()

    limit = 15 if args.sample and args.limit_pages is None else args.limit_pages

    if args.language != "all":
        # Extract single language
        lang_to_file = {
            "Igbo": ("Igbo Dictionary.pdf", 33),
            "Ibibio": ("IBIBIO DICTIONARY.pdf", 20),
            "Bini": ("Melzian - A Concise Dictionary of the Bini Language of Southern Nigeria (1937).pdf", 0)
        }
        fname, start = lang_to_file[args.language]
        pdf_path = os.path.join(args.pdf_dir, fname)
        entries = extract_from_pdf(pdf_path, args.language, start_page=start, limit_pages=limit)
        
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        logger.info("Saved %d %s entries to %s", len(entries), args.language, output_file)
    else:
        extract_all_dictionaries(
            pdf_dir=args.pdf_dir,
            output_path=args.output,
            limit_pages=limit
        )


if __name__ == "__main__":
    main()
