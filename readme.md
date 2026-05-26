# PATS: Pedagogically Adaptive Tutoring System

Official implementation of the framework for a pedagogically adaptive tutoring system that adjusts instructional strategies based on inferred student traits (mastery level and behavioral tendencies).

## Citation
Please refer to our paper:
> **Leveraging Large Language Models for Adaptive Tutoring System via Pedagogical Knowledge-Augmented Prompting** (to be updated after acceptance)

## Project Structure

```text
codes/
  analysis.py      # Student modeling (trait inference) and strategy recommendation rules
  config.py        # Environment variables, model types, prompt settings, and file paths
  data_loader.py   # Dataset parsing and loading utilities for CIMA and TSCC
  evaluation.py    # Automatic evaluation metrics (ROUGE-L, BERTScore, DialogRPT)
  main.py          # Main tutoring simulation loop and prompt formulation engine
  models.py        # Local model loading (LoRA/QLoRA) and API client setup (OpenAI/Gemini)
  utils.py         # Logging setup and JSON report saving functions
  requirements.txt # Dependencies required to run the pipeline
