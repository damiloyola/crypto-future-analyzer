"""Cada componente del score, normalizado a [-1, +1].

Se implementa en el Hito 4.

- ``trend_component(trend)``: alcista +1, bajista -1, lateral 0 (4h, 1h, 15m).
- ``momentum_component(tf_15m, tf_1h)``: RSI respecto de 50 (recortado en
  sobrecompra/sobreventa) y MACD (histograma y cruce).
- ``derivatives_component(funding, oi_change, long_short)``: funding muy
  positivo con OI subiendo = longs sobrecargados = riesgo de caída (resta);
  espejado para shorts.
- ``sentiment_component(fear_greed, news_balance)``: Fear & Greed
  (contrarian en los extremos por defecto) y balance de noticias.

Cada función devuelve el valor y un texto breve que explica el porqué.
"""
