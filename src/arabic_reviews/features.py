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
from sklearn.preprocessing import StandardScaler

from . import text

NORMALIZERS = {"none": {}, "light": text.LIGHT, "full": text.FULL}
REPRESENTATIONS = ("word11", "word12", "char25", "word11+char25")


class Normalizer(BaseEstimator, TransformerMixin):
    """Stateless text step: a ``NORMALIZERS`` preset plus optional symbol/digit handling,
    ``NEG_`` negation marking and stopword removal (all from ``text``)."""

    def __init__(self, preset: str = "none", symbols: str = "keep", digits: str = "keep",
                 negation: bool = False, stopwords: bool = False):
        self.preset = preset
        self.symbols = symbols
        self.digits = digits
        self.negation = negation
        self.stopwords = stopwords

    def fit(self, X, y=None):
        if self.preset not in NORMALIZERS:
            raise ValueError(f"unknown normalisation preset: {self.preset}")
        if self.symbols not in ("keep", "separate", "remove"):
            raise ValueError(f"unknown symbols option: {self.symbols}")
        if self.digits not in ("keep", "ascii", "token"):
            raise ValueError(f"unknown digits option: {self.digits}")
        return self

    def transform(self, X):
        options = dict(NORMALIZERS[self.preset], symbols=self.symbols, digits=self.digits)
        out = []
        for t in X:
            t = text.normalize(t, **options)
            if self.stopwords:
                t = text.remove_stopwords(t)
            if self.negation:
                t = text.mark_negation(t)
            out.append(t)
        return np.array(out, dtype=object)


class TextStats(BaseEstimator, TransformerMixin):
    """Stateless surface statistics of the raw review (see ``text.text_stats``)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([text.text_stats(t) for t in X])


# Word tokens plus single symbols (emoji, punctuation) as tokens of their own.
SYMBOL_TOKEN_PATTERN = r"(?u)\b\w\w+\b|[^\w\s]"


def _vectorizer(kind: str, symbol_tokens: bool = False) -> TfidfVectorizer:
    pattern = {"token_pattern": SYMBOL_TOKEN_PATTERN} if symbol_tokens else {}
    if kind == "word11":
        return TfidfVectorizer(analyzer="word", ngram_range=(1, 1), **pattern)
    if kind == "word12":
        return TfidfVectorizer(analyzer="word", ngram_range=(1, 2), **pattern)
    if kind == "char25":
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5))
    raise ValueError(f"unknown representation: {kind}")


def _features(representation: str, symbol_tokens: bool = False):
    if representation not in REPRESENTATIONS:
        raise ValueError(f"unknown representation: {representation}")
    parts = representation.split("+")
    if len(parts) == 1:
        return _vectorizer(parts[0], symbol_tokens)
    return FeatureUnion([(p, _vectorizer(p, symbol_tokens)) for p in parts])


def _normalizer(spec: dict) -> Normalizer:
    return Normalizer(spec.get("normalisation", "none"), spec.get("symbols", "keep"),
                      spec.get("digits", "keep"), bool(spec.get("negation")), bool(spec.get("stopwords")))


def build_pipeline(spec: dict) -> Pipeline:
    """Unfitted pipeline for an experiment spec (keys: model, params, normalisation,
    representation, class_weight; optional symbols, digits, negation, stopwords, textstats)."""
    model, params = spec["model"], dict(spec.get("params", {}))
    if model == "dummy_most_frequent":
        return Pipeline([("clf", DummyClassifier(strategy="most_frequent"))])
    if model == "dummy_stratified":
        return Pipeline([("clf", DummyClassifier(strategy="stratified", random_state=0))])
    if model == "logreg":
        clf = LogisticRegression(
            max_iter=2000, class_weight=spec.get("class_weight"), random_state=0, **params
        )
        features = _features(spec.get("representation", "word11"), spec.get("symbols") == "separate")
        if spec.get("textstats"):
            # statistics come from the raw text, so they are computed beside the text branch
            text_branch = Pipeline([("normalize", _normalizer(spec)), ("features", features)])
            stats = Pipeline([("stats", TextStats()), ("scale", StandardScaler(with_mean=False))])
            return Pipeline([("features", FeatureUnion([("text", text_branch), ("stats", stats)])),
                             ("clf", clf)])
        return Pipeline([("normalize", _normalizer(spec)), ("features", features), ("clf", clf)])
    raise ValueError(f"unknown model: {model}")
