"""Perfiles de riesgo (conservador, moderado, agresivo).

Se implementa en el Hito 4.

- ``get_profile(name, settings, max_leverage_override=None)``: devuelve el
  perfil, con override de apalancamiento máximo validado (tope duro 50x;
  por encima de 25x se marca ``red_warning = True`` para la advertencia roja
  permanente en la UI).
"""
