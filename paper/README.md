# Paper: submission material

- `main.tex` - the manuscript (compiles standalone with the `article` class).
- `references.bib` - verified references.

For a TMLR submission, download `tmlr.sty` from
https://github.com/JmlrOrg/tmlr-style-file and switch the preamble as noted in
`main.tex`. Numbers marked `\todo{...}` are pending the current experiment batch;
every non-todo number traces to `docs/results/*.json` in this repository.

Build: `pdflatex main && bibtex main && pdflatex main && pdflatex main`
(or `latexmk -pdf main`).
