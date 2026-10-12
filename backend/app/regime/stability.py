"""Filtro de estabilidad del régimen.

Se implementa en el Hito 3.

- ``confirmed_regime(raw, bars=3)``: un régimen nuevo solo se confirma si se
  mantiene 3 velas seguidas; mientras tanto sigue vigente el anterior.
- ``is_unstable(raw, window=20, max_changes=4)``: más de 4 cambios en las
  últimas 20 velas = "régimen inestable" (reduce el riesgo y genera advertencia).
- ``regime_age(raw)``: hace cuántas velas empezó el régimen actual.
"""
