# DaVinci - GPR Bridge Scan Rebar Detection Subnet

**Subnet 46** on Bittensor Mainnet | [Validator Guide](docs/VALIDATOR.md) | [Miner Guide](docs/MINER.md)

---

## Overview

DaVinci is a Bittensor subnet that incentivizes the development of accurate GPR (Ground Penetrating Radar) bridge scan rebar detection models. Miners compete to build the best ONNX models for classifying rebar presence in GPR scan windows, while validators evaluate predictions against ground-truth labeled data.

### Key Features

- **Owner-Provided Evaluation Data**: Subnet owner drops labeled GPR scan data; validators load and evaluate automatically
- **Binary Classification**: Models classify each GPR scan window as rebar-present (1) or rebar-absent (0)
- **Winner-Takes-All**: Best performing model receives 99% of emissions

---

## Incentive Mechanism

### How Scoring Works

Models are scored using **F1 Score** (harmonic mean of precision and recall):

```
Score = F1
```

Example: A model with 92% precision and 88% recall has an F1 score of 0.90.

### Winner Selection

DaVinci uses a **threshold + commit-time mechanism** to reward innovation:

1. **Find Best Score**: Identify the highest-scoring model
2. **Define Winner Set**: All models within a configurable threshold of the best score qualify
3. **Select Winner**: Within the winner set, the **earliest on-chain commit wins**

This means:
- If you match the current best model, the original pioneer keeps winning
- To become the new winner, you must **improve by more than the threshold**
- Incremental copycats cannot displace innovators

### Reward Distribution

| Category | Share | Description |
|----------|-------|-------------|
| **Winner** | 99% | Model that pioneered the best performance |
| **Non-winners** | 1% | Shared proportionally by score among valid models |
| **Copiers** | 0% | Detected duplicates receive nothing |

---

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│     Miners      │     │   Validators    │     │    Bittensor    │
│                 │     │                 │     │      Chain      │
│  Train models   │     │  Fetch models   │     │                 │
│  Upload to HF   │────►│  Run inference  │────►│  Set weights    │
│  Commit hash    │     │  Score & rank   │     │  Distribute TAO │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Components

- **Miners**: Train ML models, export to ONNX, upload to HuggingFace, register on-chain
- **Validators**: Download models, run sandboxed inference, calculate scores, set weights
- **Pylon**: Chain interaction layer handling metagraph sync and weight submission

---

## Quick Start

### For Validators

See the [Validator Setup Guide](docs/VALIDATOR.md) for complete setup instructions.

### For Miners

See the [Miner Guide](docs/MINER.md) for complete setup instructions.

---

## Model Requirements

| Requirement | Specification |
|-------------|---------------|
| **Format** | ONNX (`.onnx` file) |
| **Max Size** | 200 MB |
| **License** | MIT (verified via HuggingFace metadata) |
| **Input** | `(batch, 1, depth, width)` float32 GPR scan windows |
| **Output** | `(batch,)` or `(batch, 1)` float32 rebar detection scores |

---

## Development Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- Docker (for validators)

### Install

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone <repository-url>
cd DAVINCI-subnet
uv sync

# Run tests
uv run pytest davinci/tests/ -v
```

### Development Workflow

```bash
# Linting
uv run ruff check .
uv run ruff check --fix .

# Formatting
uv run ruff format .

# Testing
uv run pytest davinci/tests/ -v
uv run pytest davinci/tests/ --cov=davinci
```
