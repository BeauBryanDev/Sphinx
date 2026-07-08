#!/usr/bin/env python3
"""
parse_dickson.py — Scrape DicksonDictionary.pdf into pandas → CSV.

Entry format: [translit] english translation {GAR1 GAR2 ...}

Outputs:
    dickson.csv             clean: translit, translation, gardiner_seq
    dickson_quarantine.csv  any unknown Gardiner code; includes raw_line
    parse_report.txt        counts, top unknown codes, sample dropped lines

Usage:
    python parse_dickson.py
    python parse_dickson.py --pdf DicksonDictionary.pdf --layout
"""

import argparse
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from gardiner_Map import GARDINER_MAP


ENTRY_RE = re.compile(
    r'\[([^\[\]]+)\]'
    r'\s*'
    r'([^{}\[\]]*?)'
    r'\s*'
    r'\{([^{}\[\]]+)\}'
)

# Gardiner code = letter prefix (1-2 chars, e.g. A, Aa) + digits + optional
# single-letter variant suffix (lowercase or uppercase).
# Canonical form strips the trailing variant letter.
VARIANT_RE = re.compile(r'^([A-Z][a-z]?\d+)[A-Za-z]$')


def normalize_code(code: str) -> tuple[str, bool]:
    """
    Strip variant suffix to canonical form if doing so yields a known
    Gardiner code. Returns (canonical, was_normalized).
    """
    if code in GARDINER_MAP:
        return code, False
    m = VARIANT_RE.match(code)
    if m and m.group(1) in GARDINER_MAP:
        return m.group(1), True
    return code, False

PAGE_HEADER_RE = re.compile(
    r'^(?:'
    r'Dictionary of Middle Egyptian'
    r'|[A-Z]\s*[–\-]\s*[A-Z][\w ,\-/]+'
    r')\s*$'
)

PAGE_NUMBER_RE = re.compile(r'^\d{1,4}\s*$')

# A line is "noise" if it has no bracket structure at all (no [ or { or }).
# Such lines are title-page words, copyright notes, TOC entries that survived
# the header filter — they cannot be part of any entry's bracket structure.
NOISE_LINE_RE = re.compile(r'^[^\[\]{}]*$')


def extract_text(pdf_path: Path, layout: bool) -> str:
    cmd = ['pdftotext']
    if layout:
        cmd.append('-layout')
    cmd += [str(pdf_path), '-']
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return res.stdout


def reassemble(text: str) -> tuple[str, list[str]]:
    """
    Filter page headers, page numbers, and bracket-free noise lines.
    Join everything else with single spaces — one giant string the regex
    can match against. Multi-line entries reassemble for free.
    Returns (joined_text, dropped_lines).
    """
    kept, dropped = [], []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if PAGE_HEADER_RE.match(line) or PAGE_NUMBER_RE.match(line):
            continue
        if NOISE_LINE_RE.match(line):
            dropped.append(line)
            continue
        kept.append(line)
    return ' '.join(kept), dropped


def parse_text(joined: str) -> list[dict]:
    rows = []
    for m in ENTRY_RE.finditer(joined):
        # raw_line = the matched span itself (the actual entry as parsed)
        rows.append({
            'translit'    : m.group(1).strip(),
            'translation' : m.group(2).strip(),
            'gardiner_raw': m.group(3).strip(),
            'raw_line'    : joined[m.start():m.end()],
        })
    return rows


def validate(rows: list[dict]) -> tuple[list[dict], list[dict], Counter, int]:
    clean, quarantine = [], []
    unknown_codes = Counter()
    n_normalized = 0
    for r in rows:
        codes_orig = r['gardiner_raw'].split()
        canon, unknown = [], []
        any_normalized = False
        for c in codes_orig:
            cc, was_norm = normalize_code(c)
            if was_norm:
                any_normalized = True
                n_normalized += 1
            if cc in GARDINER_MAP:
                canon.append(cc)
            else:
                unknown.append(c)
                canon.append(cc)
        if unknown:
            unknown_codes.update(unknown)
            quarantine.append({
                'translit'        : r['translit'],
                'translation'     : r['translation'],
                'gardiner_seq'    : ' '.join(canon),
                'gardiner_seq_orig': ' '.join(codes_orig),
                'unknown_codes'   : ' '.join(unknown),
                'raw_line'        : r['raw_line'],
            })
        else:
            row_out = {
                'translit'    : r['translit'],
                'translation' : r['translation'],
                'gardiner_seq': ' '.join(canon),
            }
            if any_normalized:
                row_out['gardiner_seq_orig'] = ' '.join(codes_orig)
            else:
                row_out['gardiner_seq_orig'] = ''
            clean.append(row_out)
    return clean, quarantine, unknown_codes, n_normalized


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pdf', default='DicksonDictionary.pdf')
    ap.add_argument('--out-dir', default='.')
    ap.add_argument('--no-layout', dest='layout', action='store_false',
                    help='disable pdftotext -layout (default on)')
    ap.set_defaults(layout=True)
    args = ap.parse_args()

    pdf = Path(args.pdf)
    out_dir = Path(args.out_dir)
    if not pdf.exists():
        sys.exit(f"ERROR: {pdf} not found")

    print(f"Extracting text  : {pdf}  layout={args.layout}")
    text = extract_text(pdf, layout=args.layout)
    n_lines = text.count('\n')
    print(f"  raw chars      : {len(text):,}")
    print(f"  raw lines      : {n_lines:,}")

    print("Reassembling entries...")
    joined, dropped = reassemble(text)
    print(f"  joined chars   : {len(joined):,}")
    print(f"  dropped lines  : {len(dropped):,}")

    print("Parsing entries...")
    rows = parse_text(joined)
    print(f"  candidates     : {len(rows):,}")

    print("Validating Gardiner codes (with variant normalization)...")
    clean, quarantine, unknown_codes, n_normalized = validate(rows)
    print(f"  clean          : {len(clean):,}")
    print(f"  quarantined    : {len(quarantine):,}")
    print(f"  variants stripped : {n_normalized:,}")
    print(f"  unique unknowns: {len(unknown_codes):,}")

    df_clean = pd.DataFrame(clean, columns=['translit','translation','gardiner_seq','gardiner_seq_orig'])
    df_quar  = pd.DataFrame(quarantine, columns=['translit','translation','gardiner_seq','gardiner_seq_orig','unknown_codes','raw_line'])
    clean_path  = out_dir / 'dickson.csv'
    quar_path   = out_dir / 'dickson_quarantine.csv'
    report_path = out_dir / 'parse_report.txt'

    df_clean.to_csv(clean_path, index=False)
    df_quar.to_csv(quar_path, index=False)

    with open(report_path, 'w') as f:
        f.write("DicksonDictionary parse report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Source PDF        : {pdf}\n")
        f.write(f"Layout mode       : {args.layout}\n")
        f.write(f"Raw lines         : {n_lines:,}\n")
        f.write(f"Joined chars      : {len(joined):,}\n")
        f.write(f"Dropped lines     : {len(dropped):,}\n")
        f.write(f"Candidate entries : {len(rows):,}\n")
        f.write(f"Clean entries     : {len(clean):,}\n")
        f.write(f"Quarantined       : {len(quarantine):,}\n")
        f.write(f"Variants stripped : {n_normalized:,}\n")
        f.write(f"Unique unknowns   : {len(unknown_codes):,}\n")
        f.write("\nTop 30 unknown Gardiner codes:\n")
        for code, n in unknown_codes.most_common(30):
            f.write(f"  {code:10s} {n:6d}\n")
        f.write("\nFirst 15 dropped orphan lines:\n")
        for o in dropped[:15]:
            f.write(f"  {o!r}\n")

    print(f"\nWrote: {clean_path}")
    print(f"       {quar_path}")
    print(f"       {report_path}")


if __name__ == '__main__':
    main()
