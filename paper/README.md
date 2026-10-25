# Paper — ACMSE 2027 short paper (double-anonymous)

Every number in `main.tex` is a macro from `numbers.tex` or a table from `tables/`, both written by
`make_tables.py` from `results/tables/`. Nothing is typed by hand; if a result changes, rerun and recompile.

```
python -m slmtriage.analysis            # regenerate results/tables from the frozen data and run files
python paper/make_tables.py             # -> paper/tables/*.tex, paper/numbers.tex
latexmk -pdf -cd paper/main.tex         # -> paper/main.pdf   (needs acmart; TeX Live 2021+ or Overleaf)
```

Until `data/annotation/annotator_{A,B}.csv` exist, the noise-ceiling macros render as **[pending]** and
`tables/tab_noise.tex` is a placeholder, so the PDF always compiles.

Checks before submission (ACMSE: 5–6 pages including references, ACM template, double-anonymous):
- page count printed by the build cell is ≤ 6;
- no author names, affiliations, repository user names or acknowledgements anywhere (`anonymous` option hides the author block);
- `refs.bib` entries marked `VERIFY` opened and confirmed;
- the `[pending]` macros are gone and the H3/H4 verdict macros read as the pre-registration's wording.
