"""Generador del plan de operación.

Se implementa en el Hito 4.

``build_trade_plan(analysis, score, regime, safety_state, profile, settings)``:

1. Dirección candidata (long >= +30, short <= -30, si no "No operar").
2. Entrada: precio actual o S/R más cercano en 15m (dos opciones).
3. Stop por estructura (S/R de 15m/1h ± 0,5 × ATR 15m).
4. TP1 en el siguiente S/R, TP2 a 1,5R, TP3 a 2,5R; marca los que superan el
   rango típico del día como "poco probable intradía".
5. Sizing por riesgo en USD (``sizing``), comisiones (``fees``) y liquidación
   (``liquidation``); si la liquidación queda a menos de 2× la distancia al
   stop, propone subir el margen o achicar el tamaño.
6. Stop por tiempo.
7. Ajustes por régimen y reglas de bloqueo.

Los niveles finales se calculan sobre el precio de Nexo (o Binance con aviso).
Es la misma función que usa el backtest walk-forward.
"""
