# Dataset Details

This document expands on `data/README.md` with the full dataset description
used in the manuscript.

## LIAR

- 12,791 short political statements (after removing 26 duplicate/null rows
  from the original ~12,836-row release across train/test/valid splits).
- Six-point truthfulness scale, collapsed to binary:
  `REAL` = {true, mostly-true, half-true}; `FAKE` = {barely-true, false, pants-fire}.
- Class balance: 7,134 REAL (55.8%) / 5,657 FAKE (44.2%) — mild 1.26:1 imbalance.
- Average statement length: 18.0 words (median 17, max 467).
- Task framing: short-claim verification, largely without source-formatting
  signal.

## Kaggle Fake-and-Real-News (ISOT mirror)

- Public Kaggle mirror of the ISOT Fake News Dataset. Per the ISOT
  methodology, REAL articles were sourced from Reuters.com and FAKE articles
  from sites flagged by PolitiFact — a source-conflated collection design
  that is the reason this study exists.
- 39,105 articles after removing 5,793 duplicates (13.0% of raw corpus) and
  rows with missing content: 21,197 REAL (54.2%) / 17,908 FAKE (45.8%),
  mild 1.18:1 imbalance.
- Average article length: 417.7 words (median 375) — an order of magnitude
  longer than LIAR statements.
- Task framing: full-article style and content classification.

## Why this pairing

The paper does **not** use ISOT/Kaggle and LIAR as a matched political-news
pair the way prior work ([1] Hoy & Koulouri, 2022) evaluated across
same-domain political-news datasets. Instead it deliberately pairs a
full-article, source-conflated dataset (Kaggle/ISOT) against a
short-claim, fact-checking dataset (LIAR) with a very different label
provenance. This is a harder, more different cross-dataset test than [1]'s
original setup, which is why the observed 50.4-point collapse is *larger*
than the ~30-point drop [1] reported, not merely a replication of the same
number.

## Known artifact: the "Reuters" token

REAL articles in the Kaggle/ISOT corpus were literally scraped from
Reuters.com, so the token "Reuters" (via the `CITY (Reuters) -` dateline and
in-body mentions) appears in 83.0% of REAL articles vs. 1.2% of FAKE
articles — a 71.1x disproportion, verified directly from corpus counts
(see `results/token_frequency_by_class.csv`, reproduced by
`src/token_frequency_check.py`). This is the central shortcut the paper
investigates; see `docs/METHODOLOGY.md` for how it's isolated.
