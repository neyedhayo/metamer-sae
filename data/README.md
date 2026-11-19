# data/

Nothing large is committed. Layout produced by the CLI:

```
data/
├── manifests/          # committed: availability, stimulus_manifest.csv, stimulus_allocation.csv (split ledger)
├── external/           # git-ignored: Feather checkout (pinned commit), imagenette2-320
└── images/             # git-ignored: source/ standard_resnet50/ adversarial_resnet50/ (<id>.png, SHA-256 in manifest)
```

Never run Feather's broad `download_large_files.py`; `metamer-sae download` fetches exactly the configured IDs and
validates every file (size, strict decode, SHA-256, atomic move).
