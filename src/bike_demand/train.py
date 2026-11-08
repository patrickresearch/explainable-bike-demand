import json

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from bike_demand.data import FEATURES, MODEL, TARGET, load, split

PARAMS = {
    "objective": "regression",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "seed": 0,
    "deterministic": True,
    "force_row_wise": True,
    "verbosity": -1,
}


def scores(y: pd.Series, p) -> dict[str, float]:
    return {
        "rmse": round(root_mean_squared_error(y, p), 2),
        "mae": round(mean_absolute_error(y, p), 2),
        "r2": round(r2_score(y, p), 4),
    }


def main() -> None:
    train, valid, test = split(load())
    dtrain = lgb.Dataset(train[FEATURES], train[TARGET])
    search = lgb.train(
        PARAMS,
        dtrain,
        num_boost_round=3000,
        valid_sets=[lgb.Dataset(valid[FEATURES], valid[TARGET], reference=dtrain)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
    )
    fit = pd.concat([train, valid])
    model = lgb.train(PARAMS, lgb.Dataset(fit[FEATURES], fit[TARGET]), search.best_iteration)

    baseline = fit.groupby(["hr", "workingday"])[TARGET].mean().rename("baseline")
    metrics = {
        "best_iteration": search.best_iteration,
        "test_period": f"{test.dteday.min():%Y-%m-%d}..{test.dteday.max():%Y-%m-%d}",
        "lightgbm": scores(test[TARGET], model.predict(test[FEATURES])),
        "baseline_hour_x_workingday": scores(
            test[TARGET], test.join(baseline, on=["hr", "workingday"])["baseline"]
        ),
    }
    MODEL.parent.mkdir(exist_ok=True)
    model.save_model(MODEL)
    (MODEL.parent / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
