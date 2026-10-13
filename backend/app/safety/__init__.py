"""Capa de seguridad con poder de veto (Hito 5).

Independiente del score, del régimen y del LLM: se evalúa al final y puede
bloquear cualquier plan. En la fase 1 trabaja con el estado que el usuario
carga a mano (resultado de cada operación, saldo de la billetera, posiciones
abiertas y eventos de alta volatilidad); en la fase 4 va a usar datos reales
de la cuenta con la misma interfaz.

Módulos (cada uno puro salvo las funciones de I/O que se nombran):

- ``state``: ``TradingState``, los números del estado de trading en la zona
  del usuario (pérdida neta del día y de la semana, operaciones de hoy, racha
  de pérdidas con su pausa, saldo de la billetera, su máximo del mes y la
  caída). I/O: ``load_trading_state(repo, settings, profile, now)``.
- ``breakers``: un ``BreakerStatus`` por freno (pérdida del día, pausa por
  pérdidas seguidas, pérdida de la semana, caída del mes, evento de alta
  volatilidad y archivo LOCKED) y ``evaluate_breakers(...) -> SafetyVerdict``
  (verde / ámbar / rojo, multiplicador de riesgo y motivos de bloqueo). I/O:
  la lectura del LOCKED dentro de ``evaluate_breakers`` (salvo que se pase
  ``lock``) y ``trigger_lock_if_needed``, que escribe el informe del incidente
  y el LOCKED una sola vez.
- ``correlation``: matriz de correlación de retornos de 1h y riesgo combinado
  de posiciones correlacionadas (``combined_risk_limit``). I/O:
  ``fetch_correlation_matrix``.
- ``lockfile``: el archivo LOCKED (``write_lock`` nunca pisa uno existente; no
  hay función para borrarlo: lo borra el usuario a mano).
- ``incident``: el informe del incidente en markdown.

Flujo típico de un análisis::

    state = load_trading_state(repo, settings, profile, now)
    verdict = evaluate_breakers(state, profile, settings, now)
    trigger_lock_if_needed(verdict, state, settings, now, repo=repo)
    # verdict.blocked -> "No operar"; verdict.risk_multiplier -> achica el riesgo;
    # state.trades_today y state.plan_pnl_today_usd -> PlanInputs.

Esta capa está debajo de ``storage`` y no lo importa: lee los registros por
"duck typing" (ver ``state.TradingStateSource``).
"""
