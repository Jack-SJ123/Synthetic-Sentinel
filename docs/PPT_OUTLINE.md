# Final PPT Outline

Use this as the presentation backbone so the instructor can immediately see the 4-person contribution split.

## Slide 1: Title

Title:
- Synthetic Sentinel

Subtitle:
- A Multi-Model NLP Pipeline for Detecting AI-Generated Text

Bottom row:
- Group 9
- Four members, four modules

## Slide 2: Problem and Motivation

Key points:
- AI-generated text is easy to produce and hard to verify.
- We wanted an end-to-end NLP project, not just a classifier notebook.
- Our goal was to detect likely AI-written text and explain the score.

## Slide 3: System Architecture

Show the pipeline as four blocks:
- Module 1: Data Pipeline
- Module 2: Modeling and Training
- Module 3: Evaluation and Analysis
- Module 4: Demo and Reporting

Speaker note:
- Each team member owned one module so the project is easy to audit.

## Slide 4: Module 1 - Data Pipeline

Owner:
- Data Architect

Show:
- Data sources
- Cleaning
- Burstiness
- Perplexity
- Train/val/test split
- Dataset metadata

Important honesty note:
- Current benchmark in this workspace is synthetic-heavy, and we report that explicitly.

## Slide 5: Module 2 - Modeling and Training

Owner:
- ML Engineer

Show:
- Baseline: TF-IDF + Logistic Regression
- Hybrid: RoBERTa + perplexity + burstiness
- Training controls
- Warmup scheduler
- Optional frozen-encoder experiments

## Slide 6: Module 3 - Evaluation and Linguistic Analysis

Owner:
- Linguistic Analyst

Show:
- Accuracy / Precision / Recall / F1 / ROC-AUC
- Confusion matrix
- Probability distribution
- Burstiness analysis
- Error analysis
- Human-edited AI challenge results

Speaker note:
- Explain what the metrics do and do not mean under a synthetic-heavy benchmark.

## Slide 7: Module 4 - Demo and Product Layer

Owner:
- UX and Business Lead

Show:
- Streamlit screenshot
- Single-text analysis
- Batch CSV mode
- Generated report

Important point:
- The app now blocks inference when trained weights are missing, so it fails safely.

## Slide 8: New Experiment - Human-Edited AI Challenge

Show:
- Original AI text vs edited version
- Baseline drop under humanized edits
- Hybrid robustness under the same edits

Speaker note:
- This is our attempt to move beyond overly clean binary examples.

## Slide 9: What We Improved

Show before/after style bullets:
- Removed misleading inference when weights are missing
- Aligned training code with documented parameters
- Added dataset source metadata
- Rewrote report and README to match actual artifacts
- Made team ownership explicit

## Slide 10: Limitations

Be direct:
- Current dataset is 100% synthetic fallback in this workspace
- Metrics are therefore optimistic
- GPT-2 perplexity is a limited signal
- Human-edited AI text remains difficult

## Slide 11: Next Steps

Show:
- Restore real-source datasets
- Add source-held-out evaluation
- Expand beyond English
- Improve robustness against edited AI text
- Package as API or browser workflow

## Slide 12: Team Contributions

Use a 4-column layout:
- Member 1: Data pipeline and governance
- Member 2: Modeling and training
- Member 3: Evaluation and linguistic analysis
- Member 4: Demo, reporting, and presentation framing

Closing line:
- The project is designed so each contribution maps to a visible technical module.
