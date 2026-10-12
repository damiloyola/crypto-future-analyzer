"""Esquema Pydantic de la respuesta del LLM.

Se implementa en el Hito 8.

``LlmAnalysis``: resumen, sesgo, confianza, escenarios (con
``probabilidad_estimada`` que suma 100), puntos clave (máx. 5), noticias
relevantes con impacto, eventos de las próximas 24 h, riesgos y
``comentario_plan`` (puede recomendar no operar, nunca cambiar números).
"""
