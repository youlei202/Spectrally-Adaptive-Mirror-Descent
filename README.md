# Spectrally Adaptive Mirror Descent with Streaming Sketches

This repository contains a fully reproducible implementation of Spectrally Adaptive Mirror Descent with Streaming Sketches (SAMD-SS) together with all baselines, data generators, metrics, experiments, and notebooks necessary to reproduce the figures and tables in the accompanying paper.

## Project layout

```
.
├── environment.yml
├── run_all.sh
├── src/
│   ├── algorithms/          # Optimizers: SAMD-SS, SGD, AdaGrad, ONS
│   ├── sketches/            # Streaming sketch implementations
│   ├── data/                # Synthetic and real data loaders
│   ├── losses/              # Differentiable losses (squared & logistic)
│   ├── metrics/             # Regret, log-det, stability, complexity
│   ├── utils/               # Config, logging, linalg, reproducibility
│   └── experiments/         # Config-driven experiment runner
├── tests/                   # Unit and sanity tests (pytest)
├── artifacts/               # Logs, figures, tables produced by sweeps
└── notebooks/
    └── main_experiments.ipynb
```

## Getting started

1. Create the conda environment and activate it
   ```bash
   conda env create -f environment.yml
   conda activate samd-ss
   ```
2. Run the full reproducibility pipeline
   ```bash
   ./run_all.sh
   ```

The script pins random seeds, executes the pytest suite, launches all experiment configs, and finally rebuilds the notebook with Papermill. Outputs are stored under `artifacts/`.

## Tests

Run the unit tests directly when iterating:

```bash
pytest -q
```

## License

The code is released under the MIT License. See `LICENSE` (to be provided by the authors) for details.
