# Data provenance

**Dataset:** Eight source-held-out Stanford Gset graphs: G1, G6, G11, G14, G18, G22, G32, and G43.

**Original source:** https://web.stanford.edu/~yyye/yyye/Gset/

**Split:** Entire source graphs are held out from compression, affine-basis,
decision-weight, and observation-set selection.

**Integrity:** Every allowlisted graph has a pinned SHA-256 digest.

**Fields used:** The audit parses graph topology only. The third Gset edge
field (the supplied edge weight) is deliberately ignored, so the manuscript
reports unweighted depth-one MaxCut results and makes no weighted-Gset claim.

Run from the repository root:

```bash
python -m pip install -e code
python code/scripts/download_data.py
```

Downloaded third-party files remain governed by the source terms documented in
`THIRD_PARTY.md`. When redistribution is not explicit, the package fetches
the data into an external cache instead of committing the source bytes.
