# Bank Marketing Executed Evidence

This directory contains one versioned execution snapshot of the flagship Bank Marketing notebook:

- `bank_marketing_executed.html`: a static HTML rendering with code, outputs, figures, and a visible provenance header.
- `assets/figure-01.png` through `assets/figure-05.png`: external figure assets with meaningful alternative text and manifest-bound hashes.
- `bank_marketing_provenance.json`: the source commit and tree, source-notebook hash, pinned UCI archive identity, environment versions, exact HTML hash, and figure hashes.

This is a historical execution snapshot for review. It is not a live report, deployed model, or policy recommendation. The Bank result remains a weak late-period ranking result under the stated source-order stress test.

Regenerate from a clean committed source tree with:

```bash
python scripts/build_bank_evidence.py
python scripts/verify_bank_evidence.py
```

The ancestry verifier requires a full Git checkout. A downloaded source ZIP can expose the HTML and manifest for inspection, but it does not contain the commit history needed for the ancestry check.

The source notebook stays stripped. Evidence generation writes the executed figures into this directory and does not silently replace the curated preview figures under `assets/figures/`.
