# NFL Prediction Elite Pro

Advanced, production-ready Streamlit application for player prop predictions, built on statistical modeling, ensemble learning, and Monte Carlo simulation. It pulls weekly NFL data with `nfl_data_py`, builds an enhanced player/game context, and outputs a recommended OVER/UNDER decision with confidence, edge, distribution, and rich diagnostics.

- UI: Streamlit with interactive controls and Plotly visualizations
- Modeling: Empirical Bayes + EWMA, opponent/weather/injury adjustments, regression ensemble, Monte Carlo simulation
- Outputs: Recommended decision, confidence, edge vs line, prediction distribution, model weights, advanced metrics, and history

---

## Features

- Hierarchical Bayesian priors by position
- EWMA recency weighting (half-life tuning)
- Multi-model ensemble:
  - Bayesian Ridge
  - Random Forest
  - Gradient Boosting
  - Ridge Regression
- Opponent strength adjustment (allowed stats vs league)
- Weather and injury impact modeling
- Volatility regime detection and uncertainty quantification
- Monte Carlo simulation (lognormal/negative binomial/Poisson) with uncertainty sources
- Market inputs (optional, currently mock)
- Rich visualizations: histogram with line/mean, confidence gauge, metrics and tables
- Session history and summary stats

---

## Project Structure

- `enhanced_nfl_app_production.py` — Streamlit UI, data loading, inputs, visualizations, and display logic
- `enhanced_nfl_model_production.py` — Modeling engine and utilities:
  - `EnhancedNFLPredictor` with fit/predict flow
  - Feature engineering, priors, regression ensemble, and simulation
  - Context preparation (`prepare_enhanced_player_context`, `prepare_player_context`)
  - `compute_enhanced_prop_prediction(...)` entrypoint used by the UI
- `requirements.txt` — Python dependencies and version constraints

---

## Quickstart

Prerequisites:
- Python 3.10+ recommended
- macOS/Linux/Windows

1) Clone the repository
- If you haven't cloned it yet:
  - GitHub: https://github.com/DeVReV27/nfl_prediction_pro

2) Create and activate a virtual environment
- macOS/Linux:
  - `python3 -m venv .venv`
  - `source .venv/bin/activate`
- Windows (PowerShell):
  - `py -m venv .venv`
  - `.venv\Scripts\Activate.ps1`

3) Install dependencies
- `pip install -r requirements.txt`

4) Run the app
- `streamlit run enhanced_nfl_app_production.py`

The app will open in your browser (typically http://localhost:8501).

Note:
- On first run, `nfl_data_py` may download data into a local cache.

---

## Usage Guide

1) Configuration (sidebar)
- Season (year)
- Model Parameters: last N games, recency half-life, prior strength (κ)
- Advanced: distribution family, number of simulation draws, bet confidence threshold

2) Selection
- Team -> Player -> Opponent
- Stat Category (e.g. Passing Yards, Receptions, Rush+Rec Yards, Fantasy Score)

3) Betting Parameters
- Sportsbook line
- Optional snaps target for usage scaling
- Fantasy system (if using Fantasy Score)

4) Environmental Factors
- Optional weather inputs (temperature, wind, precipitation)
- Injury status and rest days

5) Advanced Options
- Market analysis (opening/current line, sharp/public percentages)
- Model ensemble toggles and optional manual weights

6) Generate
- Click “Get Enhanced Prediction”
- Review:
  - Decision (OVER/UNDER), Confidence, Edge, Predicted Mean
  - Distribution histogram and confidence gauge
  - Advanced metrics (volatility, opponent, injury, weather, uncertainty)
  - Model weights and adjustment factors
  - Statistical details and history

---

## Configuration (.env)

The app does not require secrets to run in its current state. Weather and market data are mocked for demonstration. If you later integrate external APIs, you may add a `.env` file (not tracked by git) with keys such as:

```
# Example placeholders if you add real integrations later
WEATHER_API_KEY=<your_key>
SPORTSBOOK_API_KEY=<your_key>
OPENAI_API_KEY=<your_key_if_used>
```

This repository is configured to exclude `.env` from git.

---

## Notes and Limitations

- Data coverage comes from `nfl_data_py`. Column availability can vary by season and schema; the code includes fallbacks where possible.
- Weather and market inputs are mock in this build. Replace with real API calls for production usage.
- Requirements pinning: Some upper bounds may be conservative or forward-looking. If you encounter pip resolution errors, consider relaxing to the latest stable versions for your environment (e.g., ensure `plotly`, `matplotlib`, and `streamlit` versions exist on PyPI for your platform).

---

## Troubleshooting

- Pip resolution/version errors:
  - Relax conflicting pins in `requirements.txt` to known-available versions.
- Streamlit won’t launch / port in use:
  - Try `streamlit run enhanced_nfl_app_production.py --server.port 8502`.
- No data or selection lists empty:
  - Verify `nfl_data_py` downloaded weekly data and that your chosen season has records.

---

## Acknowledgements

- [nfl_data_py](https://github.com/nflverse/nfl_data_py)
- [Streamlit](https://streamlit.io/)
- [Plotly](https://plotly.com/python/)
- scikit-learn, SciPy, NumPy, pandas

---

## Disclaimer

This project is for educational and entertainment purposes only. No guarantees of accuracy or profitability. Please bet responsibly.
