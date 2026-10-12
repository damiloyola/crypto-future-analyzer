"""Construcción del prompt.

Se implementa en el Hito 8.

Arma el input JSON con todo lo calculado (indicadores, niveles, score y su
desglose, derivados, Fear & Greed, régimen, estado de los breakers, plan y los
últimos 10-15 titulares) y las reglas: horizonte de 24 h como máximo,
probabilidades como estimación subjetiva, no inventar datos, "no operar" como
respuesta válida, español rioplatense.
"""
