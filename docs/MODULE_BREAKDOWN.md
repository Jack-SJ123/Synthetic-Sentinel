# Group 9 Module Breakdown

This project is intentionally framed as four visible modules so each team member can present a clear ownership area during the final presentation.

## Module 1: Data Pipeline and Dataset Governance

Primary responsibility:
- Data ingestion
- Synthetic fallback handling
- Cleaning, deduplication, feature generation
- Dataset metadata and data-quality caveats

Main files:
- `module1_data/data_loader.py`
- `data_loader.py` (compatibility wrapper)
- `data/train.csv`
- `data/val.csv`
- `data/test.csv`
- `data/dataset_metadata.json`

Suggested presenter:
- Data Architect

What to say in the PPT:
- We designed the dataset pipeline.
- We combined multiple text sources.
- We engineered burstiness and perplexity features.
- We added metadata so the source composition is auditable.
- We explicitly report when the benchmark is synthetic-heavy.

## Module 2: Modeling and Training

Primary responsibility:
- Baseline model design
- Hybrid model architecture
- Training configuration
- Hyperparameter control
- Reproducible experimentation

Main files:
- `module2_modeling/train_model.py`
- `train_model.py` (compatibility wrapper)
- `models/training_results.json`
- `models/training_history.json`
- `models/best_roberta.pt`

Suggested presenter:
- ML Engineer

What to say in the PPT:
- We built both a baseline and a stronger hybrid model.
- We used TF-IDF plus stylometric features for the baseline.
- We used RoBERTa plus perplexity and burstiness for the hybrid model.
- We aligned the code with the documented training pipeline.
- We support full training, frozen-encoder ablations, and subset experiments.

## Module 3: Evaluation and Linguistic Analysis

Primary responsibility:
- Test-set evaluation
- Model comparison
- Error analysis
- Interpretation of burstiness and perplexity
- Honesty about limitations and benchmark realism

Main files:
- `module3_evaluation/evaluate.py`
- `evaluate.py` (compatibility wrapper)
- `results/evaluation_results.json`
- `results/model_comparison.csv`
- `results/confusion_matrix.png`
- `results/prob_dist.png`
- `results/burstiness.png`
- `results/roc_curve.png`

Suggested presenter:
- Linguistic Analyst

What to say in the PPT:
- We measured accuracy, precision, recall, F1, and ROC-AUC.
- We compared baseline and hybrid results.
- We visualized prediction distributions and burstiness behavior.
- We analyzed where the model fails.
- We flagged that current metrics are inflated by synthetic data composition.

## Module 4: Product Demo and Communication Layer

Primary responsibility:
- Interactive demo
- User-facing outputs
- Storytelling and usability
- Final report generation
- Presentation framing for instructor review

Main files:
- `module4_demo/app.py`
- `module4_demo/generate_report.py`
- `app.py` (compatibility wrapper)
- `generate_report.py` (compatibility wrapper)
- `Final_Report.docx`
- `README.md`

Suggested presenter:
- UX and Business Lead

What to say in the PPT:
- We turned the model into an interactive demo.
- We exposed probability, label, perplexity, and burstiness to users.
- We made the app fail safely when weights are missing.
- We generated a report directly from project artifacts.
- We organized the system so instructor-facing deliverables are easy to follow.

## Recommended Slide Mapping

Use one owner per module slide:

1. Problem and system overview
2. Module 1: Data pipeline
3. Module 2: Modeling and training
4. Module 3: Evaluation and linguistic analysis
5. Module 4: Demo and product value
6. Limitations and next steps

## One-Line Team Summary

Group 9 built the project as a four-part pipeline: data, modeling, evaluation, and demo/reporting, with each member owning one visible module.
