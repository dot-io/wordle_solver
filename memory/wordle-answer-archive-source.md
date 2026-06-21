---
name: wordle-answer-archive-source
description: Where to refresh the dated Wordle answer history that feeds the recency prior
metadata:
  type: reference
---

The recency prior in `solver.rs` reads `wordle-history.csv` (cols: `id,solution,print_date,days_since_launch,editor`, chronological by date).

Refresh it from the full archive CSV:
`curl -sL https://stuckonwordle.s3.amazonaws.com/wordle/history.csv -o wordle-history.csv`
(linked from https://stuckonwordle.com/all-wordle-answers.html — the on-page table is JS-rendered, so scrape the S3 CSV, not the HTML).

Empirical facts that justify the model (as of 2026-06-01, 1809 answers): only 13 distinct words ever repeated, each once, and **no word has ever repeated within 524 games**. Hence the prior is a hard ~500-game cooldown (`COOLDOWN`) then exponential recovery (`TAU`), not a decay from t=0.
