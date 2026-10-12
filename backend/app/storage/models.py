"""Tablas.

Se implementa en el Hito 6.

- ``Analysis``: cada análisis generado (fecha, moneda, precio, régimen, score y
  desglose, plan, salida del LLM, snapshot de inputs en JSON).
- ``AnalysisOutcome``: "¿qué pasó después?" (Hito 9).
- ``ManualTrade``, ``WalletState``, ``OpenPosition``: estado cargado a mano
  para los circuit breakers (semilla de la fase 2).
"""
