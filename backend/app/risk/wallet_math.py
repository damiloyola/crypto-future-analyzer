"""Matemática de la billetera (panel educativo).

Se implementa en el Hito 9.

- ``breakeven_win_rate(rr, fees_r)``: % de aciertos para no perder plata.
- ``losses_to_ruin(wallet, risk)``: pérdidas seguidas que vacían la billetera.
- ``monte_carlo(win_rate, rr, risk, wallet, sims=200, trades=30)``:
  percentiles 10/50/90 del resultado del mes y % de simulaciones que terminan
  con la billetera en cero.
"""
