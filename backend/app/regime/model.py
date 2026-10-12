"""Entrenamiento del HMM con selección automática del número de estados.

Se implementa en el Hito 3.

- ``train_regime_model(features, settings) -> RegimeModel``: prueba de
  ``n_states_min`` a ``n_states_max`` estados (``n_restarts`` inicializaciones
  cada uno) y se queda con el de menor BIC. Nada hardcodeado.
- ``RegimeModel`` guarda el ``GaussianHMM``, el escalador ajustado con datos de
  entrenamiento, el mapeo estado → etiqueta y metadatos (fecha, moneda,
  timeframe, BIC por cantidad de estados).
"""
