# Phase 2 duplicate-audit method

Exact duplicates are defined solely by identical file-content SHA-256 and assigned connected components. Near-duplicate candidates are generated using deterministic 64-bit DCT pHash and dHash; a shared 16-bit hash bucket plus min(Hamming distance) <= 8 is a screening rule, not a duplicate determination. No near-duplicate candidate is merged automatically. The blinded review CSVs exclude split, labels, source directory, model predictions, filename-group and local absolute path.
