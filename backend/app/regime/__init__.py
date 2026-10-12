"""Detector de régimen de mercado con un Gaussian HMM por moneda.

No predice el precio: clasifica el entorno (crash, bajista, neutral, alcista,
euforia) para que el resto del sistema se adapte (el plan de riesgo achica o
bloquea operaciones según el tipo de régimen, la confianza y la estabilidad).

Regla de oro: cero look-ahead. El régimen de cada vela sale del algoritmo
forward (probabilidades filtradas con datos hasta esa vela), nunca de
``predict()`` ni de Viterbi sobre la serie completa.
``tests/test_regime_no_lookahead_guard.py`` revisa el código y falla si
aparecen esos métodos.

Módulos, de abajo hacia arriba:

- ``types``: contrato compartido (``FEATURE_COLUMNS``, ``FeatureScaler``,
  ``StateInfo``, ``RegimeModel``). Las features se calculan en
  ``app.indicators.features`` (velas de 1h o 1d cerradas).
- ``model``: entrenamiento con selección automática del número de estados por
  BIC (de ``regime.n_states_min`` a ``n_states_max``).
- ``labels``: ordena los estados por retorno medio y les pone nombre y
  categoría según cuántos haya (``regime.labels``).
- ``store``: guarda y carga los modelos en disco y decide si hay que
  reentrenar (``regime.retrain_every_days``).
- ``forward``: probabilidades filtradas (el régimen de cada vela).
- ``stability``: confirmación tras ``regime.confirmation_bars`` velas seguidas
  y "régimen inestable" con más de ``instability_max_changes`` cambios en
  ``instability_window`` velas.
- ``service``: todo junto. ``get_regime`` / ``get_market_regimes`` devuelven un
  ``RegimeReport`` (1h para operar y 1d de contexto) con régimen actual,
  confianza, estabilidad, historial y tablas; ``ensure_model`` entrena y guarda
  (lo usa ``scripts/train_regime.py``).
"""
