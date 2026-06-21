# Obsolete files after V21

You may delete this leftover file if it exists in your deployed EC atlas directory:

```text
ec-recognizer/data/curves_compact.json
```

It is not used by V20 or V21. Runtime curve lookup now uses lazy shards:

```text
ec-recognizer/data/curve_shards/
ec-recognizer/data/search/
ec-recognizer/data/tau_index.json
```

Keeping `curves_compact.json` will not break anything, but it wastes deployment size and may make uploads slower.
