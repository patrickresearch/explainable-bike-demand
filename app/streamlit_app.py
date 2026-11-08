from datetime import date
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

MODEL = Path(__file__).resolve().parents[1] / "model" / "model.txt"
WEATHER = {1: "Clear", 2: "Mist / clouds", 3: "Light rain / snow", 4: "Heavy rain / snow"}
SEASON_START = [(3, 21), (6, 21), (9, 23), (12, 21)]


@st.cache_resource
def load():
    model = lgb.Booster(model_file=str(MODEL))
    return model, shap.TreeExplainer(model)


def season(d: date) -> int:
    return sum((d.month, d.day) >= s for s in SEASON_START) % 4 + 1


st.set_page_config(page_title="Explainable bike demand", layout="wide")
st.title("Why does the model expect this many rentals?")
model, explainer = load()

with st.sidebar:
    d = st.date_input("Date", date(2012, 6, 15), date(2011, 1, 1), date(2012, 12, 31))
    hr = st.slider("Hour", 0, 23, 17)
    holiday = st.checkbox("Public holiday")
    weathersit = st.selectbox("Weather", WEATHER, format_func=WEATHER.get)
    temp = st.slider("Temperature (°C)", -8.0, 39.0, 24.0, 0.5)
    hum = st.slider("Humidity (%)", 0, 100, 55)
    windspeed = st.slider("Wind (km/h)", 0, 57, 12)

weekday = (d.weekday() + 1) % 7
day = pd.DataFrame(
    {
        "hr": range(24),
        "weekday": weekday,
        "workingday": int(1 <= weekday <= 5 and not holiday),
        "holiday": int(holiday),
        "season": season(d),
        "mnth": d.month,
        "yr": d.year - 2011,
        "weathersit": weathersit,
        "temp": temp,
        "hum": hum,
        "windspeed": windspeed,
    }
)[model.feature_name()]

row = day.iloc[[hr]]
curve = model.predict(day).clip(0)
warmer = max(model.predict(row.assign(temp=temp + 5))[0], 0)
sv = explainer(row)

c1, c2, c3 = st.columns(3)
c1.metric(f"Predicted rentals at {hr:02d}:00", f"{curve[hr]:.0f}")
c2.metric("Average hour (model baseline)", f"{sv.base_values[0]:.0f}")
c3.metric("If it were 5 °C warmer", f"{warmer:.0f}", f"{warmer - curve[hr]:+.0f}")

left, right = st.columns([3, 2])
with left:
    st.subheader("Contribution of each factor")
    shap.plots.waterfall(sv[0], max_display=12, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)
with right:
    st.subheader("Same conditions, every hour")
    st.line_chart(pd.Series(curve, index=pd.RangeIndex(24, name="hour"), name="rentals"))

st.caption(
    "SHAP values explain what the model has learned, not cause and effect in the real world. "
    "Correlated factors such as month, season and temperature share credit with each other."
)
