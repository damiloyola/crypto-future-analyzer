"""Comisiones de futuros.

Se implementa en el Hito 2 (con tests).

- Se cobran sobre el TAMAÑO de la posición, no sobre el margen, en la entrada
  y en la salida (taker/maker según config).
- ``round_trip_fees(size_usd, fees)`` en USD.
- ``fees_as_pct_of(risk_usd, target_usd)`` para mostrar el peso de las
  comisiones sobre el riesgo y sobre el objetivo.
- ``funding_estimate(size_usd, funding_rate, hours)``: costo estimado de
  funding si la posición queda abierta hasta el stop por tiempo.
"""
