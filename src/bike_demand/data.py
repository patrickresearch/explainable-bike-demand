from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "hour.csv"
MODEL = ROOT / "model" / "model.txt"

TARGET = "cnt"
FEATURES = [
    "hr",
    "weekday",
    "workingday",
    "holiday",
    "season",
    "mnth",
    "yr",
    "weathersit",
    "temp",
    "hum",
    "windspeed",
]


def load() -> pd.DataFrame:
    if not DATA.exists():
        DATA.parent.mkdir(exist_ok=True)
        with urlopen(URL) as r, ZipFile(BytesIO(r.read())) as z:
            DATA.write_bytes(z.read("hour.csv"))
    df = pd.read_csv(DATA, parse_dates=["dteday"])
    return df.assign(
        temp=(df.temp * 47 - 8).round(1),
        hum=(df.hum * 100).round(),
        windspeed=(df.windspeed * 67).round(1),
    )


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    t = df.dteday
    return (
        df[t < "2012-07-01"],
        df[(t >= "2012-07-01") & (t < "2012-10-01")],
        df[t >= "2012-10-01"],
    )
