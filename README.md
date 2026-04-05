# Synthetic Sentinel

**A Multi-Model Approach to AI-Generated Misinformation Detection**

*ARTI407 - NLP Final Project*

## Overview

Synthetic Sentinel is an NLP pipeline that distinguishes **human-written text** from **AI-generated content** by combining:

- **RoBERTa** for contextual classification
- **Perplexity** as a GPT-2 based predictability signal
- **Burstiness** as a sentence-length variance signal

The current repository is organized as both a runnable prototype and a presentation-ready course project.

## Repository Structure

This codebase is now split into 4 explicit modules so each team member can present a clear ownership area:

- `module1_data/` for dataset construction and feature engineering
- `module2_modeling/` for baseline and hybrid-model training
- `module3_evaluation/` for metrics, plots, and error analysis
- `module4_demo/` for the Streamlit demo and report generation

Root-level scripts such as `train_model.py` and `app.py` are kept as compatibility wrappers so existing commands still work.

```text
Final_Project/
|-- module1_data/
|   |-- data_loader.py
|-- module2_modeling/
|   |-- train_model.py
|-- module3_evaluation/
|   |-- evaluate.py
|   |-- quick_experiments.py
|-- module4_demo/
|   |-- app.py
|   |-- generate_report.py
|-- data/
|-- models/
|-- results/
|-- docs/
|-- data_loader.py
|-- train_model.py
|-- evaluate.py
|-- quick_experiments.py
|-- app.py
|-- generate_report.py
```

## Team Framing for Final Presentation

To make the 4-person contribution split obvious to the instructor, the project is framed as four visible modules:

1. Data Pipeline and Dataset Governance
2. Modeling and Training
3. Evaluation and Linguistic Analysis
4. Demo and Reporting

Presentation-ready material:

- `docs/MODULE_BREAKDOWN.md`
- `docs/PPT_OUTLINE.md`

## Quick Start

```bash
pip install -r requirements.txt
python data_loader.py
python train_model.py
python evaluate.py
python quick_experiments.py
streamlit run app.py
python generate_report.py
```

## Reproducing Model Artifacts

The trained weights file `models/best_roberta.pt` may be excluded from some distributions because of size limits. To reproduce results:

1. Run the data pipeline: `python data_loader.py`
2. Train the models: `python train_model.py`
3. Run evaluation: `python evaluate.py`
4. Generate final report assets: `python generate_report.py`

## Project Status

The current workspace is a usable course-project prototype, not a production-ready detector.

- The pipeline can fall back to synthetic/template-generated data when real datasets are unavailable.
- When that fallback is used, metrics should be interpreted as synthetic benchmark results, not broad real-world generalization.
- `data/dataset_metadata.json` records the current source composition of the generated dataset.

## Training Notes

By default, the training script now uses the hyperparameters you pass on the command line.

Examples:

```bash
python train_model.py --epochs 3 --lr 2e-5 --batch-size 16 --max-len 256
python train_model.py --freeze-roberta
python train_model.py --max-train-samples 2000 --max-val-samples 500
python train_model.py --baseline-only
```

## Data Notes

The intended sources are:

- TweepFake via HuggingFace
- FakeNewsNet-style local CSV data
- Custom synthetic samples

If external or local real-source data is unavailable, the loader generates synthetic fallback samples so the rest of the pipeline remains runnable. That keeps the project reproducible, but it also makes the benchmark easier and less realistic.

## New Evaluation Additions

To strengthen the final presentation, the project now includes:

- `Human-edited AI` gray-zone interpretation in the demo
- quick challenge experiments for edited AI text
- source-held-out baseline diagnostics
- improved visual explanations in the Streamlit app

Useful generated artifacts:

- `results/quick_challenge_experiment.json`
- `results/quick_challenge_chart.png`
- `results/source_held_out_baseline.csv`
- `results/source_held_out_baseline.png`
- `results/source_metrics.csv`

## Outputs

- `data/train.csv`, `data/val.csv`, `data/test.csv`
- `data/dataset_metadata.json`
- `models/best_roberta.pt`
- `models/training_results.json`
- `models/training_history.json`
- `results/evaluation_results.json`
- `results/*.png`

## Demo Behavior

The Streamlit app now refuses to run inference if trained weights are missing. It also provides:

- a gray-zone tag such as `Possible Human-Edited AI`
- visual signal explanation for the current example
- comparison charts for probability, confidence, perplexity, burstiness, and sentence rhythm

## Limitations

- Current results can be inflated if the dataset is dominated by synthetic templates.
- GPT-2 perplexity is only a rough proxy for newer AI writing styles.
- Human-edited AI text remains difficult.
- English is the only supported language in the current implementation.

## License

Academic project for SAIT ARTI407, Winter 2026.
