"""Caso de uso "Analizar".

Se implementa en el Hito 6 (y suma el LLM en el Hito 8).

``run_analysis(symbol, profile_name, settings) -> FullAnalysis``:

1. snapshot de datos (``data.snapshot``);
2. indicadores por timeframe (``indicators.pipeline``);
3. régimen 1h y diario (``regime.service``);
4. score (``scoring.confluence``);
5. plan (``risk.plan``);
6. veto de seguridad (``safety.breakers`` + lock + correlación);
7. LLM (``llm.client``), opcional;
8. guardado en SQLite (``storage``).
"""
