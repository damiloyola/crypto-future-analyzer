"""Precio de liquidación aproximado (margen aislado).

Se implementa en el Hito 2 (con tests).

Con apalancamiento L, margen de mantenimiento mmr (fracción) y entrada E:

- Long:  ``liq ≈ E × (1 - 1/L + mmr)``
- Short: ``liq ≈ E × (1 + 1/L - mmr)``

Opcionalmente descuenta la comisión de salida del margen disponible. Es una
aproximación conservadora: en Nexo el colateral es toda la billetera de
futuros, así que con más saldo la liquidación real queda más lejos (ver
docs/nexo-api-research.md).

- ``liquidation_distance_ok(entry, stop, liq, min_ratio=2)``: la distancia a la
  liquidación tiene que ser al menos 2 veces la distancia al stop.
"""
