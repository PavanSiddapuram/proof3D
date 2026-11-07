# eval

| Path | What |
|---|---|
| `configs/*.yaml` | Experiment configs, validated by `PipelineConfig` (unknown keys fail) |
| `manifest.example.yaml` | Dataset manifest format: every photo by content hash |
| `results/` | One JSON per (product, condition, held-out photo), written by the pipeline |
| `scripts/make_report.py` | `make report`: rebuilds every table from `results/` |

Anything that produces a reported number is a script here, never a notebook.
