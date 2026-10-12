"""Backtest walk-forward.

Se implementa en el Hito 10.

Entrena el HMM con 6 meses, evalúa el mes siguiente (que el modelo no vio) y
avanza mes a mes durante los últimos 12-18 meses. En cada vela de 15m corre el
mismo generador de planes que usa la app (``risk.plan``) y simula el resultado.
Nunca más de una operación abierta por moneda.
"""
