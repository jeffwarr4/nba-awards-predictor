# 🐝 NBA Awards Predictor

A data-driven machine learning project that predicts **NBA MVP rankings and vote share probabilities** using real-time player performance and team context.

Built as part of the Stinger Collectibles Analytics platform, this project combines sports analytics, machine learning, and automated data pipelines.

---

## 📊 Overview

The NBA Awards Predictor:

- Predicts MVP vote share using a trained regression model
- Ranks players based on MVP likelihood
- Tracks week-over-week movement (↑ ↓ NEW)
- Publishes outputs for web, social media, and analytics

---

## 🚀 Features

- 📈 Machine Learning model (Ridge Regression)
- 🔄 Automated data pipeline (NBA API ingestion)
- 📊 Weekly MVP rankings (Top 10)
- 🎨 Canva-ready outputs for social media
- 🌐 GitHub Pages integration for live data

---

## 🧠 Model Details

### Key Features

- Player stats:
  - Points, rebounds, assists
  - Steals, blocks
  - Games played

- Team context:
  - Win percentage
  - Conference rank

### Output

- `ridge_pred_vote_share` → predicted MVP vote %
- `mvp_score` → ranking score
- Final player ranking

---

## ⚙️ Data Pipeline

### Step 1: Pull Player Stats

```python
python scripts/build_current_candidates.py
```

### Step 2: Pull Team Standings
```python
python scripts/build_team_standings.py
```

### Step 3: Run Model + Rankings
```python
python scripts/build_rankings_ridge_np_v1.py
```
## 📤 Outputs
### Core Files
- web_current_top10.csv
- web_prior_top10.csv
- model_all_candidates.csv

### Canva files
- canva_top5_flat.csv
- canva_6_10_flat.csv

## 🔁 Automation
Runs weekly via GitHub Actions
```YAML
schedule:
  - cron: "0 18 * * 2"  # Tuesdays 10am Pacific
```
## 🌐 Live Data
All outputs are published via GitHub Pages:
https://jeffwarr4.github.io/nba-awards-predictor/nba/mvp/

## 💡 Why This Project Matters
- End-to-end data pipeline
- Real-world machine learning application
- Automated workflows (GitHub Actions)
- Data product for web + social + analytics

## 👤 Author
### Jeff Warren

Senior Program Manager | Data & Analytics | ML-Driven Products | Microsoft Fabric | Power BI | Founder @ Stinger Collectibles

GitHub: https://github.com/jeffwarr4 
