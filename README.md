# RoLL: Robust Low-Rank Learning via Nesterov Momentum

This repository provides the implementation of the proposed RoLL method, together with utility functions and representative scripts for simulated and real-world datasets.

- **`function/`**: General utility functions supporting the algorithm implementation.
- **`dataset/`**: Real-world datasets.

The implementation can be readily extended to other loss functions by modifying the loss, the corresponding gradient functions, and, when needed, the proximal operators.

## Main scripts

- **`simulation_data_generation.py`**: Generates synthetic datasets for simulation studies.
- **`split_noisy_data.py`**: Splits real multi-label datasets.
- **`split_real_data.py`**: Splits clean multi-label datasets and flips a proportion of training labels.
- **`main_RoLL_simulation.py`**: Runs RoLL on simulated datasets and reports evaluation metrics.
- **`main_RoLL_emotions.py`**: Runs RoLL on the `emotions` dataset and reports evaluation metrics.

## Usage

Run the simulation experiments with:

```bash
python main_RoLL_simulation.py
```

Run the real-data experiments with:

```bash
python main_RoLL_emotions.py
```
