from sklearn.model_selection import cross_validate, StratifiedKFold, KFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import Dict, Any, List

class CrossValidator:
    """
    Evaluates a model with three CV strategies.
    """
    def __init__(
        self,
        strategy: str = "stratified_kfold",
        n_splits: int = 5,
        test_size: float = 0.2,
        random_state: int = 42,
        scoring: List[str] = None,
    ):
        self.strategy = strategy.lower()
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state
        self.scoring = scoring or ["accuracy", "f1_macro", "precision_macro", "recall_macro"]

    def evaluate(self, model, X, y) -> Dict[str, Any]:
        """Returns a dict with mean ± std for every requested metric."""
        if self.strategy == "train_test":
            return self._train_test_eval(model, X, y)

        cv = self._make_splitter()
        results = cross_validate(
            model, X, y,
            cv=cv,
            scoring=self.scoring,
            n_jobs=-1,
            return_train_score=False
        )

        summary = {}
        for metric in self.scoring:
            scores = results[f"test_{metric}"]
            summary[metric] = {
                "mean": round(scores.mean(), 4),
                "std": round(scores.std(), 4),
                "scores": scores.tolist(),
            }

        self._print_summary(summary)
        return summary

    def _train_test_eval(self, model, X, y) -> Dict[str, Any]:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        summary = {
            "accuracy": {"mean": round(accuracy_score(y_te, y_pred), 4)},
            "f1_macro": {"mean": round(f1_score(y_te, y_pred, average="macro", zero_division=0), 4)},
            "precision_macro": {"mean": round(precision_score(y_te, y_pred, average="macro", zero_division=0), 4)},
            "recall_macro": {"mean": round(recall_score(y_te, y_pred, average="macro", zero_division=0), 4)},
        }
        print(f"\\n[Train/Test Split  test_size={self.test_size}]")
        self._print_summary(summary)
        return summary

    def _make_splitter(self):
        if self.strategy == "kfold":
            print(f"\\n[K-Fold  k={self.n_splits}]")
            return KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        elif self.strategy == "stratified_kfold":
            print(f"\\n[Stratified K-Fold  k={self.n_splits}]")
            return StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    @staticmethod
    def _print_summary(summary: Dict[str, Any]):
        for metric, vals in summary.items():
            std_str = f" ± {vals['std']:.4f}" if "std" in vals else ""
            print(f"   {metric:<22} {vals['mean']:.4f}{std_str}")
