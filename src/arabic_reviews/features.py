"""Feature pipelines for the experiments: one place that turns an experiment spec into an
unfitted ``sklearn`` Pipeline, so that every vocabulary and weight is learned inside the
training fold only and the same text logic (``text.normalize``) is used everywhere."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from . import text

NORMALIZERS = {"none": {}, "light": text.LIGHT, "full": text.FULL}
REPRESENTATIONS = ("word11", "word12", "char25", "word11+char25")


class Normalizer(BaseEstimator, TransformerMixin):
    """Stateless wrapper around ``text.normalize`` (a preset name from ``NORMALIZERS``)."""

    def __init__(self, preset: str = "none"):
        self.preset = preset

    def fit(self, X, y=None):
        if self.preset not in NORMALIZERS:
            raise ValueError(f"unknown normalisation preset: {self.preset}")
        return self

    def transform(self, X):
        options = NORMALIZERS[self.preset]
        return np.array([text.normalize(t, **options) for t in X], dtype=object)


def _vectorizer(kind: str) -> TfidfVectorizer:
    if kind == "word11":
        return TfidfVectorizer(analyzer="word", ngram_range=(1, 1))
    if kind == "word12":
        return TfidfVectorizer(analyzer="word", ngram_range=(1, 2))
    if kind == "char25":
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5))
    raise ValueError(f"unknown representation: {kind}")


def _features(representation: str):
    if representation not in REPRESENTATIONS:
        raise ValueError(f"unknown representation: {representation}")
    parts = representation.split("+")
    if len(parts) == 1:
        return _vectorizer(parts[0])
    return FeatureUnion([(p, _vectorizer(p)) for p in parts])


def build_pipeline(spec: dict) -> Pipeline:
    """Unfitted pipeline for an experiment spec (keys: model, params, normalisation,
    representation, class_weight)."""
    model, params = spec["model"], dict(spec.get("params", {}))
    if model == "dummy_most_frequent":
        return Pipeline([("clf", DummyClassifier(strategy="most_frequent"))])
    if model == "dummy_stratified":
        return Pipeline([("clf", DummyClassifier(strategy="stratified", random_state=0))])
    if model == "logreg":
        clf = LogisticRegression(
            max_iter=2000, class_weight=spec.get("class_weight"), random_state=0, **params
        )
        return Pipeline(
            [
                ("normalize", Normalizer(spec.get("normalisation", "none"))),
                ("features", _features(spec.get("representation", "word11"))),
                ("clf", clf),
            ]
        )
    raise ValueError(f"unknown model: {model}")
