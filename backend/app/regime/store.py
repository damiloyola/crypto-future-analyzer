"""Persistencia de modelos de régimen en disco.

Se implementa en el Hito 3.

- ``save_model(model, path)`` / ``load_model(path)`` con joblib, en
  ``regime.models_dir`` (``data/models/{symbol}_{timeframe}.joblib``).
- ``needs_retrain(path, every_days=7)``: reentrenamiento semanal.
"""
