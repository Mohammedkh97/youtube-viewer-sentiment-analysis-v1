# pyrefly: ignore [missing-import]
import optuna
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    KFold,
    StratifiedKFold,
    cross_val_score,
)
import logging

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory that returns a (possibly tuned) estimator."""

    _MANUAL_PARAMS = {
        "random_forest": dict(n_estimators=800, max_depth=15),
        "xgboost": dict(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
        ),
        "lightgbm": dict(n_estimators=300, learning_rate=0.1),
    }

    @staticmethod
    def _build_base(name: str, random_state: int, **params):
        name = name.lower()
        shared = dict(random_state=random_state, **params)
        if name == "random_forest":
            return RandomForestClassifier(**shared)
        elif name == "xgboost":
            return XGBClassifier(eval_metric="logloss", **shared)
        elif name == "lightgbm":
            return LGBMClassifier(verbose=-1, **shared)
        else:
            raise ValueError(f"Unknown model: {name}")

    @staticmethod
    def _make_cv(strategy: str, n_splits: int, random_state: int):
        strategy = strategy.lower()
        if strategy == "train_test":
            return StratifiedKFold(n_splits=2, shuffle=True, random_state=random_state)
        elif strategy == "kfold":
            return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        elif strategy == "stratified_kfold":
            return StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
        else:
            raise ValueError(f"Unknown cv_strategy: {strategy}")

    @staticmethod
    def _suggest_params(trial, model_name: str):
        if model_name == "random_forest":
            return dict(
                n_estimators=trial.suggest_int("n_estimators", 50, 800),
                max_depth=trial.suggest_int("max_depth", 3, 30),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
                max_features=trial.suggest_categorical(
                    "max_features", ["sqrt", "log2"]
                ),
            )
        elif model_name == "xgboost":
            return dict(
                n_estimators=trial.suggest_int("n_estimators", 50, 400),
                learning_rate=trial.suggest_float(
                    "learning_rate", 0.005, 0.3, log=True
                ),
                max_depth=trial.suggest_int("max_depth", 2, 10),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            )
        elif model_name == "lightgbm":
            return dict(
                n_estimators=trial.suggest_int("n_estimators", 50, 400),
                learning_rate=trial.suggest_float(
                    "learning_rate", 0.005, 0.3, log=True
                ),
                num_leaves=trial.suggest_int("num_leaves", 20, 200),
                max_depth=trial.suggest_int("max_depth", -1, 15),
            )
        else:
            raise ValueError(f"No Optuna sampler defined for model: {model_name}")

    @classmethod
    def get(
        cls,
        name: str,
        param_grids: dict,
        random_state: int = 42,
        search_strategy: str = "manual",
        cv_strategy: str = "stratified_kfold",
        n_splits: int = 5,
        n_iter: int = 20,
        n_trials: int = 30,
        X_train=None,
        y_train=None,
        scoring: str = "f1_macro",
        verbose: int = 1,
    ):
        name_lower = name.lower()

        if search_strategy == "manual":
            params = cls._MANUAL_PARAMS.get(name_lower, {})
            logger.info(f"[Manual] {name} | params: {params}")
            return cls._build_base(name_lower, random_state, **params)

        if X_train is None or y_train is None:
            raise ValueError(
                f"X_train and y_train must be provided for search_strategy='{search_strategy}'"
            )

        cv = cls._make_cv(cv_strategy, n_splits, random_state)

        if search_strategy == "grid":
            base = cls._build_base(name_lower, random_state)
            grid = param_grids.get(name_lower, {})
            gs = GridSearchCV(
                base, grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=verbose
            )
            gs.fit(X_train, y_train)
            logger.info(f"[GridSearch] Best params: {gs.best_params_}")
            return gs.best_estimator_

        if search_strategy == "random":
            base = cls._build_base(name_lower, random_state)
            grid = param_grids.get(name_lower, {})
            rs = RandomizedSearchCV(
                base,
                grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                random_state=random_state,
                n_jobs=-1,
                verbose=verbose,
            )
            rs.fit(X_train, y_train)
            logger.info(f"[RandomSearch] Best params: {rs.best_params_}")
            return rs.best_estimator_

        if search_strategy == "bayesian":

            def objective(trial):
                params = cls._suggest_params(trial, name_lower)
                model = cls._build_base(name_lower, random_state, **params)
                scores = cross_val_score(
                    model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1
                )
                return scores.mean()

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=random_state),
            )
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
            logger.info(f"[Bayesian] Best params: {study.best_params}")
            return cls._build_base(name_lower, random_state, **study.best_params)

        raise ValueError(f"Unknown search_strategy: '{search_strategy}'")
