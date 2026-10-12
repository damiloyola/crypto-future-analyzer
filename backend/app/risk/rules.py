"""Reglas que bloquean la operación y fuerzan "No operar".

Se implementa en el Hito 4. Cada regla devuelve ``RuleResult`` (ok, motivo,
sugerencia de ajuste):

- R:R a TP2 (con comisiones) menor al mínimo;
- 4h y 1h en direcciones opuestas;
- regla de liquidación incumplida y sin ajuste posible;
- apalancamiento resultante mayor al tope del perfil;
- límite de operaciones o de pérdida del día alcanzado;
- evento de alta volatilidad marcado a mano.
"""
