#!/usr/bin/env python3
"""
mine_determinatives.py — auto-mine determinative-like Gardiner codes from
dickson.csv.

A code is flagged as determinative-like if it tends to appear as the last
token of a word entry (Egyptian determinatives are word-final classifiers).

Output: determinatives.json
    {
      "source": "...",
      "criteria": "...",
      "codes": {
         "A1": {"last_count": 612, "total_count": 670, "last_rate": 0.913,
                "category": "man"},
         ...
      }
    }
"""
import json
from collections import Counter
from pathlib import Path

import pandas as pd

DICKSON_CSV = Path(__file__).parent / 'dickson.csv'
OUT_JSON    = Path(__file__).parent / 'determinatives.json'

# Min thresholds for inclusion. Two paths to acceptance:
#   strong-signal : ≥5 word-final occurrences AND last_rate ≥ 0.50
#   high-rate     : ≥2 word-final AND last_rate ≥ 0.70
MIN_LAST_STRONG = 5
RATE_STRONG     = 0.50
MIN_LAST_HIGH   = 2
RATE_HIGH       = 0.70


# Gardiner classifier-family categories.
# Tag is derived from the code's letter prefix; codes the user knows are
# atypical (D-class is mostly phonetic with a few exceptions, etc.) can be
# overridden via OVERRIDE.
PREFIX_CATEGORY = {
    'A' : 'man',
    'B' : 'woman',
    'C' : 'god',
    'D' : 'body_part',
    'E' : 'mammal',
    'F' : 'animal_part',
    'G' : 'bird',
    'H' : 'bird_part',
    'I' : 'reptile',
    'K' : 'fish',
    'L' : 'insect',
    'M' : 'plant',
    'N' : 'celestial_or_geographical',
    'O' : 'building_or_place',
    'P' : 'ship',
    'Q' : 'furniture',
    'R' : 'temple_object',
    'S' : 'crown_or_dress',
    'T' : 'weapon_tool',
    'U' : 'tool_craft',
    'V' : 'rope_basket',
    'W' : 'vessel',
    'X' : 'bread_or_food',
    'Y' : 'writing_or_abstract',
    'Z' : 'stroke_or_marker',
    'Aa': 'unclassified',
}

# Hand overrides for codes whose semantic determinative role differs from
# the prefix-family default.
OVERRIDE = {
    'A40': 'god',           # seated god
    'A41': 'king',          # seated king
    'A42': 'noble',         # noble with flagellum
    'A50': 'noble',         # seated noble
    'A52': 'noble',         # noble crouched
    'D54': 'motion',        # walking legs
    'D55': 'motion',        # legs walking back
    'D56': 'motion',        # leg
    'F35': 'good_or_abstract',  # heart and trachea (nfr)
    'I9' : 'reptile_phonetic',  # cerastes — primarily phonetic 'f'
    'M3' : 'wood',
    'N5' : 'sun_or_time',
    'N25': 'foreign_land',
    'O1' : 'house_or_place',
    'O49': 'town_or_place',
    'X1' : 'bread_or_phonetic_t',  # X1 also phonetic 't'
    'Y1' : 'abstract',      # papyrus roll → abstract concept determinative
    'Z1' : 'singular_or_ideogram',
    'Z2' : 'plural_three',
    'Z3' : 'plural_three_horizontal',
}


def category_for(code: str) -> str:
    if code in OVERRIDE:
        return OVERRIDE[code]
    if code.startswith('Aa'):
        return PREFIX_CATEGORY['Aa']
    return PREFIX_CATEGORY.get(code[0], 'unknown')


def main() -> None:
    df = pd.read_csv(DICKSON_CSV)
    last_count = Counter()
    all_count  = Counter()

    for seq in df['gardiner_seq'].dropna():
        codes = seq.split()
        if not codes:
            continue
        last_count[codes[-1]] += 1
        for c in codes:
            all_count[c] += 1

    selected = {}
    for code, total in all_count.items():
        if total < 2:
            continue
        last = last_count.get(code, 0)
        rate = last / total
        accept = (
            (last >= MIN_LAST_STRONG and rate >= RATE_STRONG)
            or (last >= MIN_LAST_HIGH and rate >= RATE_HIGH)
        )
        if not accept:
            continue
        selected[code] = {
            'last_count' : last,
            'total_count': total,
            'last_rate'  : round(rate, 3),
            'category'   : category_for(code),
        }

    # Sort by last_count desc → most prominent first
    selected = dict(sorted(selected.items(),
                           key=lambda kv: (-kv[1]['last_count'], kv[0])))

    out = {
        'source'  : 'auto-mined from dickson.csv',
        'criteria': (f'last_count>={MIN_LAST_STRONG} AND last_rate>={RATE_STRONG} '
                     f'OR last_count>={MIN_LAST_HIGH} AND last_rate>={RATE_HIGH}'),
        'count'   : len(selected),
        'codes'   : selected,
    }

    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # Console summary
    by_cat = Counter(meta['category'] for meta in selected.values())
    print(f"Determinatives extracted: {len(selected)}")
    print(f"Output: {OUT_JSON}")
    print("\nBy category:")
    for cat, n in by_cat.most_common():
        print(f"  {cat:30s} {n:>4}")
    print("\nTop 15 by word-final frequency:")
    for code, meta in list(selected.items())[:15]:
        print(f"  {code:6s} last={meta['last_count']:>4}  rate={meta['last_rate']:.2f}  "
              f"category={meta['category']}")



if __name__ == '__main__':
    main()
