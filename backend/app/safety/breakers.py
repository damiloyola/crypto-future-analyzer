"""Circuit breakers.

Se implementa en el Hito 5. Cada breaker devuelve ``BreakerStatus`` (umbral,
valor actual, progreso 0-1, acción y motivo):

| Disparador                                   | Acción                          |
|----------------------------------------------|---------------------------------|
| Pérdida del día >= 50 % del límite diario    | Riesgo × 0,5 el resto del día   |
| Pérdida del día >= límite diario             | Bloqueo hasta el día siguiente  |
| 3 pérdidas seguidas                          | Bloqueo de 4 h (enfriamiento)   |
| Pérdida semanal >= 60 % de la billetera      | Riesgo × 0,5 hasta fin de semana|
| Billetera del mes cae >= 70 % desde su máximo| Bloqueo total con lock file     |

``evaluate_breakers(state, profile, settings, now) -> SafetyVerdict`` combina
todo en verde / ámbar / rojo para el banner.
"""
