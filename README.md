---
title: Tax Policy OpenEnv
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---
# AI Tax Policy Optimization Environment

## Overview and Motivation
This environment simulates economic policy by challenging an AI to optimize a country's tax rate. It provides a real-world task where the agent balances GDP growth, unemployment, and inequality.

## Spaces Definition
* **Observation Space**: GDP (float), Tax Rate (float %), Unemployment (float %), Inequality (float 0-1).
* **Action Space**: Tax change (float between -5.0 and +5.0).

## Tasks and Expected Difficulty
1.  **Easy**: Maximize GDP by aggressively lowering taxes.
2.  **Medium**: Balance GDP and unemployment.
3.  **Hard**: Balance GDP, unemployment, and inequality, requiring precise adjustments.

## Setup and Usage Instructions
1.  Clone this repository.
2.  Install requirements: `pip install -r requirements.txt`
3.  Set your Hugging Face API key: `export HF_TOKEN="your_token"`
4.  Run inference: `python inference.py`

## Baseline Performance Scores
* **Easy**: ~0.85
* **Medium**: ~0.70
* **Hard**: ~0.55