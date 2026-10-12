"""Indicadores técnicos: funciones puras sobre DataFrames OHLCV.

Sin red, sin estado y sin configuración global: reciben datos y parámetros,
devuelven Series/DataFrames. Están implementados a mano (sin pandas-ta, que
quedó sin mantenimiento y choca con numpy 2) y cada uno tiene tests contra
valores de referencia.
"""
