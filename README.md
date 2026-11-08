# explainable-bike-demand

Hourly bike-sharing demand forecasts with LightGBM, explained prediction by prediction with SHAP.

## Quickstart

```bash
uv sync
uv run python -m ipykernel install --user --name explainable-bike-demand --display-name "explainable-bike-demand (3.12)"
uv run python -m bike_demand.train
uv run streamlit run app/streamlit_app.py
```

The notebook `notebooks/explain.ipynb` uses the `explainable-bike-demand (3.12)` kernel and writes all figures to `reports/figures/`.

## Data

[UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), hourly, Washington D.C., 2011–2012. Downloaded on first use. Temperature, humidity and wind are converted back to °C, % and km/h.

## Validation

Time-based split: train 2011-01 to 2012-06, early stopping on 2012-07 to 2012-09, refit on both, test on 2012-10 to 2012-12. Baseline: mean demand per hour × working day.

## Structure

```
src/bike_demand/   data loading, features, training
notebooks/         global + local explanations, PI/PDP comparison, what-if
app/               Streamlit app
model/             trained model + test metrics
reports/figures/   figures used in this README
```
