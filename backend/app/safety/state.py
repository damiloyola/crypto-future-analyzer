"""Estado de trading cargado a mano.

Se implementa en el Hito 5.

- Operaciones cerradas (fecha, moneda, dirección, resultado en USD, régimen y
  score del momento si vino de un análisis).
- Saldo de la billetera de futuros y su máximo del mes.
- Posiciones abiertas (moneda, dirección, riesgo en USD).
- Evento de alta volatilidad marcado a mano.

Persistido en SQLite (``storage``). En la fase 4 este módulo se reemplaza por
datos reales de la cuenta, con la misma interfaz.
"""
