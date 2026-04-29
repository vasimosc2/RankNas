# RankNAS

RankNAS is a resource-aware Neural Architecture Search (NAS) framework for discovering efficient TakuNet models for image classification on constrained embedded devices, such as the Arduino Nano 33 BLE Sense.

The project combines an evolutionary search strategy with an optional RankNet surrogate model to reduce the number of fully trained candidate architectures. Candidate models are evaluated using classification performance and deployment-related constraints such as RAM usage, flash memory usage, TFLite size, and training cost.

## Main Features

- Evolutionary search for TakuNet architectures
- Optional RankNet-based surrogate selection
- Hardware-aware model filtering
- CIFAR-100 image classification support
- TFLite conversion and deployment-oriented evaluation
- Pareto filtering based on accuracy, RAM, and flash memory
- Result export for thesis experiments and comparison plots
- Arduino deployment utilities

## Repository Structure

```text
RankNas/
├── Arduino/                    # Arduino deployment-related code
├── Retrain/                    # Retraining utilities
├── SurrogateComparisson/       # RankNet surrogate model and comparison logic
├── plotting/                   # Scripts for visualizing results
├── results/                    # Stored experiment results
├── utils/                      # Helper utilities
├── TakuNet.py                  # TakuNet model definition
├── TrainBestModels.py          # Script for retraining selected models
├── evolutionary_run.py         # Main NAS execution script
├── search_strategy.py          # Evolutionary search implementation
├── data_processing.py          # Dataset loading and preprocessing
├── compute_ram_show.py         # RAM estimation utility
├── config.json                 # Search space and training configuration
└── env.sh                      # Environment setup helper
```

## Configuration

The search space and training parameters are defined in `config.json`.

The configuration includes:

- Stem block parameters
- Taku block/stage parameters
- Downsampling parameters
- Refiner block parameters
- Optimizer choices
- Training settings
- Early stopping parameters
- Adaptive dropout settings
- RAM and flash memory constraints

The default setup is designed for CIFAR-100 classification and embedded deployment constraints.

## Setup

Create and activate a Python virtual environment:

```bash
python3.11 -m venv venv
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Running the NAS Search

Run the default evolutionary search:

```bash
python evolutionary_run.py
```

Example with custom options:

```bash
python evolutionary_run.py \
  --time 12 \
  --population_size 6 \
  --lr_strategy cosine \
  --hardwareConstrains true \
  --use_ranknet true
```

## Important Arguments

| Argument | Description |
|---|---|
| `--time` | Total NAS search time in hours |
| `--population_size` | Number of models per generation |
| `--lr_strategy` | Learning-rate schedule: `cosine`, `linear`, or `step` |
| `--hardwareConstrains` | Enables hardware-aware filtering |
| `--performaceStoppage` | Enables performance-based stopping during training |
| `--early_stopping_acc` | Enables accuracy-based early stopping |
| `--midway_callback` | Enables midway training callback |
| `--use_ranknet` | Enables or disables RankNet surrogate selection |

## Outputs

Each NAS run creates a dated experiment folder under:

```text
NAS/<date>/
```

The main outputs include:

```text
NAS/<date>/results/Best_Models_Results_NAS.csv
NAS/<date>/results/Pareto_Optimal_Models.csv
NAS/<date>/results/History/
NAS/<date>/saved_models/
```

The Pareto-optimal models are also exported to:

```text
ThesisResults/
```

The Pareto filtering keeps models that are not dominated with respect to:

- Test accuracy
- Estimated RAM usage
- Estimated flash memory usage

## Hardware-Aware Search

When hardware constraints are enabled, candidate architectures that exceed the configured RAM or flash limits are skipped before training. These limits are controlled in `config.json`.

Default constraints include:

- Maximum RAM consumption
- Additional RAM overhead
- Maximum flash consumption
- Additional flash overhead
- Data precision multiplier
- Model precision multiplier
