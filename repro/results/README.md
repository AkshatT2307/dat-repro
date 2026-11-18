# Archived experiment records

Original six runs are under epoch128; six five-epoch continuation branches are
under continuation5. JSONL metrics, validation histories, summaries, configurations,
and per-image/per-tile evaluation CSVs are retained. These are experiment records,
not checkpoints. Full weights and datasets remain on the training machine.

Machine-specific absolute paths were normalized for publication. Original
manifest hashes remain historical provenance, not hashes of rewritten paths.
Data split filenames and source metadata are in data/. No training images or
seismic section arrays are included here.

To regenerate plots from these records without training, from repro/:

```bash
mkdir -p runs
cp -a results/epoch128 results/continuation5 runs/
.venv/bin/python plot_validation.py
.venv/bin/python plot_continuations.py
```

Generating new inference examples requires trained weights and downloaded datasets.
