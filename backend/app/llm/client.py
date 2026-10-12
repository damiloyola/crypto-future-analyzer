"""Cliente del LLM con validación.

Se implementa en el Hito 8.

``analyze(input, settings, secrets) -> LlmAnalysis | None``: llama a la API de
Anthropic (modelo de ``llm.model``), valida la respuesta contra
``schema.LlmAnalysis``; si es inválida reintenta una vez y, si vuelve a
fallar, devuelve ``None`` para que el dashboard muestre el análisis sin la
parte del LLM. Sin ``ANTHROPIC_API_KEY`` se desactiva solo.
"""
