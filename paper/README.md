# Paper: submission material

- `main.tex` - the manuscript (compiles standalone with the `article` class).
- `references.bib` - verified references.

For a TMLR submission, download `tmlr.sty` from
https://github.com/JmlrOrg/tmlr-style-file and switch the preamble as noted in
`main.tex`. Numbers marked `\todo{...}` are pending the current experiment batch;
every non-todo number traces to `docs/results/*.json` in this repository.

Build: `pdflatex main && bibtex main && pdflatex main && pdflatex main`
(or `latexmk -pdf main`).

## Submission checklist (TMLR)

- [ ] Author affiliation filled in `main.tex` (currently `[affiliation]`)
- [ ] Swap preamble to `tmlr.sty` (see header comment)
- [x] Fresh literature sweep (last: 2026-07-30, no preempting work; see
      `docs/sota-review.md` section 8) - re-run just before submission
- [x] Maganti (arXiv 2604.19514) reconciliation in Related Work and Section 4.3
- [x] LLM-assistance disclosure (Acknowledgments)
- [x] Per-seed statistics artifact (`docs/results/stats_summary.json`)
- [ ] Decide co-authors, if any
