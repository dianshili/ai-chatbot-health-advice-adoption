"""Refit the retained models from scored inputs; no questionnaire/scoring code."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import lightgbm as lgb
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parents[1]


def metrics(y, pred, prefix):
    return {f"{prefix}_R2": float(r2_score(y, pred)), f"{prefix}_RMSE": float(mean_squared_error(y, pred) ** 0.5), f"{prefix}_MAE": float(mean_absolute_error(y, pred))}


def preprocessor(features, categorical, scaled):
    numeric = [c for c in features if c not in categorical]
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    cat_steps = [("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    if scaled:
        num_steps.append(("scaler", StandardScaler()))
        cat_steps.append(("scaler", StandardScaler(with_mean=False)))
    return ColumnTransformer([("num", Pipeline(num_steps), numeric), ("cat", Pipeline(cat_steps), categorical)], remainder="drop", verbose_feature_names_out=True)


def estimator(name, params):
    classes = {"CatBoost": CatBoostRegressor, "Random forest": RandomForestRegressor, "LightGBM": LGBMRegressor, "XGBoost": XGBRegressor, "Support vector machine": SVR, "Elastic Net": ElasticNet, "Linear regression": LinearRegression}
    return classes[name](**params)


def fit(name, params, X, y, config, validation=None):
    pre = preprocessor(config["features"], config["categorical_features"], name in config["scaled_models"])
    encoded = pre.fit_transform(X)
    model = estimator(name, params)
    if validation is not None and name == "CatBoost":
        xv, yv = validation
        model.fit(encoded, y, eval_set=(pre.transform(xv), yv), early_stopping_rounds=100, use_best_model=True)
    elif validation is not None and name == "LightGBM":
        xv, yv = validation
        model.fit(encoded, y, eval_set=[(pre.transform(xv), yv)], eval_metric="rmse", callbacks=[lgb.early_stopping(80, verbose=False)])
    else:
        model.fit(encoded, y)
    return Pipeline([("preprocess", pre), ("model", model)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/ml")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--skip-shap", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config/analysis.json").read_text())
    data = pd.read_csv(ROOT / "data/analysis_scores.csv", float_precision="round_trip").set_index("release_id")
    partitions = pd.read_csv(ROOT / "data/ml_partitions.csv")
    assert len(data) == 2149 and data.index.is_unique
    assert partitions.release_id.is_unique and set(partitions.release_id) == set(data.index)
    ids = {s: partitions.loc[partitions.split.eq(s), "release_id"].tolist() for s in ("train", "validation", "test")}
    X = data[config["features"]]
    y = data[config["outcome"]]
    benchmark = pd.read_csv(ROOT / "benchmarks/model_performance.csv").set_index("Model")
    names = args.models or benchmark.index.tolist()
    rows, prediction_rows, checks = [], [], {}
    selected = None
    for name in names:
        print(f"Refitting {name}", flush=True)
        train, valid, test = ids["train"], ids["validation"], ids["test"]
        validation_model = fit(name, config["model_parameters"][name], X.loc[train], y.loc[train], config, (X.loc[valid], y.loc[valid]))
        final_model = fit(name, config["model_parameters"][name], X.loc[train + valid], y.loc[train + valid], config)
        row = {"Model": name}
        for split_name, model, keys in [("Train", validation_model, train), ("Validation", validation_model, valid), ("Test", final_model, test)]:
            pred = model.predict(X.loc[keys])
            row.update(metrics(y.loc[keys], pred, split_name))
            if split_name == "Test":
                prediction_rows.extend({"release_id": k, "model": name, "observed": float(a), "predicted": float(b)} for k, a, b in zip(keys, y.loc[keys], pred))
        rows.append(row)
        delta = {k: float(row[k] - benchmark.loc[name, k]) for k in row if k != "Model"}
        checks[name] = {"max_abs_metric_difference": max(abs(v) for v in delta.values()), "differences": delta}
        if name == "CatBoost":
            selected = final_model
        pd.DataFrame(rows).to_csv(out / "model_performance.csv", index=False)
        print(f"{name}: maximum metric difference {checks[name]['max_abs_metric_difference']:.3g}", flush=True)
    pd.DataFrame(prediction_rows).to_csv(out / "test_predictions.csv", index=False)
    if selected is not None and not args.skip_shap:
        print("Computing grouped interventional TreeSHAP", flush=True)
        pre = selected.named_steps["preprocess"]
        bg = X.loc[ids["train"]].sample(n=config["shap_background_size"], random_state=config["seed"])
        test = X.loc[ids["test"]]
        explainer = shap.TreeExplainer(selected.named_steps["model"], data=pre.transform(bg), feature_perturbation="interventional")
        values = np.asarray(explainer.shap_values(pre.transform(test), check_additivity=False))
        names_encoded = pre.get_feature_names_out()
        grouped = {}
        for feature in config["features"]:
            indices = [i for i, n in enumerate(names_encoded) if n.split("__", 1)[-1] == feature or n.split("__", 1)[-1].startswith(feature + "_")]
            grouped[feature] = values[:, indices].sum(axis=1)
        gv = pd.DataFrame(grouped, index=test.index)
        gv.index.name = "release_id"
        gv.to_csv(out / "shap_values.csv")
        importance = gv.abs().mean().rename("mean_abs_shap").sort_values(ascending=False).rename_axis("feature").reset_index()
        importance.to_csv(out / "shap_importance.csv", index=False)
        old = pd.read_csv(ROOT / "benchmarks/shap_importance.csv").set_index("feature")
        new = importance.set_index("feature")
        # This CatBoost/SHAP combination returns the tree expectation without
        # CatBoost's global prediction bias. Include that intercept only in the
        # baseline diagnostic; the reported SHAP contributions stay unchanged.
        scale, bias = selected.named_steps["model"].get_scale_and_bias()
        assert scale == 1.0
        prediction_baseline = float(explainer.expected_value) + float(bias)
        checks["SHAP"] = {"max_abs_importance_difference": float((new.mean_abs_shap - old.mean_abs_shap).abs().max()), "top10_order_matches": importance.feature.head(10).tolist() == old.index[:10].tolist(), "prediction_baseline": prediction_baseline, "additivity_max_abs_difference": float(np.max(np.abs(gv.sum(axis=1).to_numpy() + prediction_baseline - selected.predict(test))))}
    checks["scope"] = "Fixed-parameter refit from scored inputs; historical parameter search not rerun."
    checks["model_metrics_match_1e-8"] = all(checks[n]["max_abs_metric_difference"] <= 1e-8 for n in names)
    (out / "verification.json").write_text(json.dumps(checks, indent=2, allow_nan=False) + "\n")
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == "__main__":
    main()
