# Crypto Future Analyzer

Aplicación web local para analizar criptomonedas antes de abrir operaciones de
futuros intradiarias (long/short). Corre en la Mac y se puede usar desde el
celular en la misma red wifi.

> **Herramienta de análisis personal. No es asesoramiento financiero.**
> La app no predice el futuro y **no ejecuta operaciones**: reúne datos
> objetivos, calcula niveles y riesgo con reglas fijas y presenta escenarios
> con probabilidades **estimadas**.

## Principios

1. Las reglas de riesgo las aplica **el código**, nunca el LLM. El LLM
   interpreta y redacta; no decide tamaños ni puede saltarse reglas.
2. **Riesgo en dólares primero, apalancamiento después**: el apalancamiento
   sale del cálculo, no se elige a mano.
3. **Cero look-ahead**: el detector de régimen usa solo datos hasta cada vela
   (algoritmo forward), con un test que lo verifica.
4. Los circuit breakers tienen **poder de veto** sobre cualquier plan.
5. Datos, indicadores, scoring y riesgo son **módulos reutilizables**, sin
   lógica en la UI, para montar encima las fases 2 a 4.

## Fases

| Fase | Qué es | Estado |
|---|---|---|
| 1 | Analizador visual (sin órdenes) | En construcción |
| 2 | Diario de operaciones con estadísticas | Pendiente |
| 3 | Bot en simulación con backtesting | Pendiente |
| 4 | Bot real conectado por API a Nexo | Pendiente |

### Hitos de la fase 1

| # | Hito | Estado |
|---|---|---|
| 0 | Esqueleto del proyecto | ✅ |
| 1 | Investigación de la Nexo Pro API ([informe](docs/nexo-api-research.md)) | ✅ |
| 2 | Datos e indicadores + script de consola ([fuentes](docs/data-sources.md)) | ✅ (falta correrlo en vivo en la Mac) |
| 3 | Detector de régimen (HMM) + `make train-regime` | ✅ (falta entrenarlo con datos reales en la Mac) |
| 4 | Score, plan de riesgo, ajustes por régimen y perfiles | ✅ (se ve en la consola; falta probarlo con datos reales en la Mac) |
| 5 | Circuit breakers, correlación y carga a mano del estado (`make trade`, `make safety`) | ✅ (se ve en la consola; falta probarlo con tus operaciones reales en la Mac) |
| 6 | API FastAPI | ⏳ |
| 7 | Dashboard: régimen, KPIs y gráfico principal | ⏳ |
| 8 | Integración del LLM | ⏳ |
| 9 | Simulador, matemática de la billetera, paneles e historial | ⏳ |
| 10 | Validación histórica walk-forward | ⏳ |
| 11 | README y documentación completa | ⏳ |

## Requisitos

- macOS (probado para Apple Silicon) o Linux.
- **Python 3.11 a 3.13** (recomendado 3.12). Lo más simple: `brew install uv`;
  el `Makefile` usa `uv` si está instalado y se baja el Python correcto solo.
  Sin `uv` hace falta `python3` 3.11+ (`brew install python@3.12`).
- **Node.js 20.19+ o 22+** (`brew install node`).

## Uso

```bash
make setup   # instala backend y frontend; crea .env a partir de .env.example
# completá a mano ANTHROPIC_API_KEY (y opcional CRYPTOPANIC_API_KEY) en .env
make dev     # levanta backend y frontend juntos
```

- Dashboard: http://localhost:5173
- Desde el celular: `http://<ip-de-la-mac>:5173` (`make dev` muestra la IP).
  La primera vez, macOS puede pedir permiso para aceptar conexiones entrantes.
- API y documentación interactiva: http://localhost:8000/docs

Otros comandos: `make help`.

```bash
make test    # tests del backend (pytest) y del frontend (typecheck + vitest)
make lint    # ruff
```

### Script de consola (Hito 2)

Imprime todo lo que la app sabe de una moneda: precios, indicadores por
timeframe, soportes y resistencias, derivados, sentimiento, noticias, el score
de confluencia y el plan de operación (ver Hito 4).

```bash
make cli SYMBOL=BTC                          # datos reales (Binance, o Bybit si falla)
make cli SYMBOL=SOL ARGS="--nexo-price 187,45"   # con el precio que ves en Nexo
make cli SYMBOL=ETH ARGS="--profile moderado"
make cli SYMBOL=BTC ARGS="--offline"         # datos sintéticos, sin red (para probar)
make cli SYMBOL=BTC ARGS="--json"            # salida en JSON
```

Si Binance y Bybit no responden, termina con un mensaje claro (código 1). Si
falla una fuente secundaria (noticias, Fear & Greed), sigue y lo avisa.

### Detector de régimen (Hito 3)

Antes de cualquier señal, la app clasifica qué tipo de mercado hay (crash,
bajista, neutral, alcista, euforia) con un modelo estadístico (Gaussian HMM),
uno por moneda. **No predice el precio**: detecta el entorno. Usa velas de 1
hora (el régimen para operar) y velas diarias (el de contexto), siempre
cerradas y sin mirar el futuro: el régimen de cada vela se calcula solo con
las velas hasta ella (algoritmo forward; hay tests que lo verifican).

Mira el **resumen de las últimas 24 velas** (cuánto subió o bajó, rango y
volumen promedio, más la volatilidad), no cada vela suelta: así una vela
grande no alcanza para declarar un "Crash" y, en el régimen de 1 hora, cada
régimen dura del orden de un día (decisión tuya;
`regime.features.smoothing_window`, 1 = vela por vela). Ojo: el ajuste es uno
solo para los dos timeframes, así que en el régimen diario de contexto las 24
velas son **24 días**: mira la tendencia de las últimas tres semanas y media,
cada régimen dura de 2 a 4 semanas y hacen falta 190 días de calentamiento
(así lo elegiste: el diario da la foto grande y es solo informativo).

```bash
make train-regime                              # todas las favoritas, 1 hora y diario
make train-regime SYMBOL=SOL                   # una sola moneda
make train-regime SYMBOL=SOL ARGS="--force"    # reentrena aunque esté al día
make train-regime SYMBOL=BTC ARGS="--offline"  # datos sintéticos, sin red (para probar)
make cli SYMBOL=BTC                            # la consola ya muestra el régimen
make cli SYMBOL=BTC ARGS="--no-regime"         # sin régimen (más rápido)
```

- La cantidad de estados (de 3 a 5: Crash, Bajista, Neutral, Alcista y
  Euforia) la elige sola el BIC. Los modelos se guardan en `data/models/`
  (los de prueba, en `data/models/sintetico/`, y nunca se mezclan) y se
  reentrenan solos cuando tienen 7 días (`regime.auto_retrain`). Entrenar una
  moneda tarda alrededor de 1 minuto y cuarto (casi todo es el modelo de 1
  hora); si hace falta, la consola avisa antes.
- Qué muestra: régimen actual y si está confirmado (se mantuvo 3 velas
  seguidas) o hay un cambio en espera; confianza (la probabilidad del estado
  más probable en la última vela, "baja" si es menos de 60 %); "régimen
  inestable" si el tipo más probable cambió más de 4 veces en las últimas 20
  velas; hace cuánto empezó; y una tabla por régimen con retorno medio por
  vela, volatilidad, duración típica y % del tiempo. Avisa si el modelo es
  viejo, si se entrenó con poca historia o si las velas vinieron de Bybit.
- Resumen de los últimos 30 días: cuántas veces cambió el régimen confirmado,
  cuánto duraron en promedio los tramos completos (los que empezaron y
  terminaron dentro de los 30 días; con 0 o 1 cambio no hay ninguno y no se
  muestra duración), % del tiempo "inestable" y % con confianza baja. Con
  datos de prueba: un cambio cada ~33 h, inestable menos del 1 % del tiempo.
  Si con datos reales sale muy distinto, avisame.
- La confirmación y la inestabilidad cuentan cambios de tipo (Neutral →
  Bajista; `regime.change_level: category`). Con hasta 5 estados cada tipo es
  un solo estado, así que es lo mismo que contar cambios de estado (los
  nombres para 6 y 7 estados de `regime.labels` hoy no se usan).
- `make train-regime` termina con código 1 si alguna moneda no se pudo
  entrenar o reentrenar (en ese caso queda el modelo anterior y lo dice), y
  con 130 si lo cortás con Ctrl+C (sale enseguida y no guarda nada a medias).

### Score y plan de operación (Hito 4)

La consola termina con el **score de confluencia** (de -100 a +100, con lo que
suma o resta cada parte) y el **plan de operación**: un cartel grande LONG,
SHORT o NO OPERAR y, abajo, entrada, stop, objetivos, margen, tamaño,
apalancamiento, liquidación, comisiones y stop por tiempo. Es la misma función
que va a usar la app (`risk.plan.build_trade_plan`). Nunca opera: solo calcula.

```bash
make cli SYMBOL=BTC                                         # plan con el perfil por omisión (agresivo)
make cli SYMBOL=SOL ARGS="--nexo-price 187,45"              # niveles sobre el precio que ves en Nexo
make cli SYMBOL=ETH ARGS="--max-leverage 30"                # tope a mano, de 1x a 50x (más de 25x: advertencia roja)
make cli SYMBOL=BTC ARGS='--high-vol-event "Dato de CPI"'   # evento solo para esta corrida: No operar
make cli SYMBOL=BTC ARGS="--trades-today 2 --pnl-today -30" # números del día a mano, para probar (ver Hito 5)
make cli SYMBOL=BTC ARGS="--json"                           # el plan va en "plan" y el score en "score"
```

Cómo se arma, en pocas palabras (todos los números están en `config.yaml`):

- **Dirección**: long si el score es +30 o más, short si es -30 o menos; en el
  medio, No operar.
- **Stop**: debajo del soporte (o arriba de la resistencia) más cercano de 15m
  o 1h, con un colchón de media vela (0,5 ATR de 15m). Los niveles a menos de
  una vela normal (1 ATR) se saltean.
- **Objetivos**: TP1 en el siguiente soporte o resistencia (también salteando
  los que están a menos de 1 ATR; si ese nivel queda en el TP2 o más lejos, no
  hay TP1 y el primer objetivo es el TP2); TP2 en 1,5 veces lo que arriesgás,
  corrido lo justo para que, pagadas las comisiones, ganes al menos lo que pide
  tu perfil o el régimen (1,5 o 2 por cada 1 que arriesgás); TP3, un "R" más
  allá. Si el TP2 queda más lejos de lo que se mueve el precio en un día normal
  (ATR diario), No operar; los objetivos así se marcan "poco probable
  intradía". Si faltan las velas diarias para saberlo, también No operar.
- **Tamaño**: primero el riesgo en dólares (50 USD en el perfil agresivo, con
  las comisiones adentro), de ahí el tamaño, y el apalancamiento sale solo
  (tamaño / margen). Si pasa tu tope o la liquidación queda cerca del stop, la
  app sube el margen; si no alcanza, achica la posición; si el riesgo queda
  debajo del mínimo, No operar. Si la posición queda más chica que el margen
  (menos de 1x, pasa con stops muy lejos), el margen baja solo a 1x. Cada
  ajuste (también los del régimen) se muestra con su motivo. En Euforia, si no
  hay dato de funding, se toma como muy alto (riesgo × 0,5 para los longs).
- **Límite de pérdida del día**: el riesgo se achica a lo que te queda del
  límite (por ejemplo, si perdiste 80 de tus 100 USD, la operación arriesga 20
  en vez de 50), así un stop no te hace pasar el límite. Si lo que queda es
  menos que el mínimo operable (15 USD en el perfil agresivo), No operar hasta
  mañana. El tope vale también para los niveles de referencia de un plan
  bloqueado (por ejemplo, si ya hiciste todas las operaciones del día).
- **No operar** también si: 4h y 1h van en direcciones opuestas, el régimen lo
  bloquea (por ejemplo, long en Crash), el score no llega al mínimo que pide el
  régimen (por ejemplo, +50 para un long con el régimen Bajista), llegaste al
  límite de operaciones o de pérdida del día, marcaste un evento de alta
  volatilidad, o el precio de Nexo que cargaste difiere más de 2 % del de
  Binance (¿error de tipeo?: se muestran los dos precios y hay que corregirlo).
  Se listan todas las reglas que fallan, cada una con una sugerencia, y los
  niveles se muestran igual como "referencia, no operar" para que veas por qué.
- **Entrada alternativa**: una orden límite en el soporte (long) o la
  resistencia (short) más cercano de 15m, salteando los que están a menos de
  1 ATR del precio (pegados al precio sería casi lo mismo que entrar a
  mercado), con su propio stop, tamaño, ajustes, liquidación y comisiones.
- **Stop por tiempo**: si en 24 h no tocó el TP1 (o el TP2, si no hay TP1) ni
  el stop, cerrá (la hora se muestra en tu zona horaria). También estima el
  funding de esas 24 h (aparte del riesgo).

### Frenos y carga de operaciones (Hito 5)

En la fase 1 la app no ve tu cuenta: **vos le contás** cómo te va. Con eso, una
capa de seguridad con **poder de veto** (independiente del score, del régimen y
de la IA) mira tu día, tu semana y tu mes antes de cada plan, y puede achicar el
riesgo o bloquear todo.

```bash
make trade ARGS="add --symbol BTC --side long --pnl -32,5"       # una operación cerrada (resultado neto, comisiones incluidas)
make trade ARGS='add --symbol ETH --side short --pnl 18 --opened "ayer 22:00" --closed "ayer 23:40"'
make trade ARGS="open --symbol SOL --side long --risk 40"         # una posición que dejaste abierta
make trade ARGS="update 1 --risk 0"                               # moviste el stop a la entrada (el riesgo previsto queda anotado)
make trade ARGS="close 1 --pnl -12"                               # la cerrás y queda como operación
make trade ARGS="balance --amount 300"                            # el saldo que ves en la billetera de futuros
make trade ARGS="deposit --amount 100"                            # también withdraw --amount 50
make trade ARGS='event --note "Dato de CPI" --until "mañana 11:00"'  # evento de alta volatilidad
make trade ARGS="delete trade 3"                                  # borrar algo cargado por error (pregunta antes)
make trades                                                       # lista lo cargado
make safety                                                       # cómo están los frenos
```

- **Fechas y horas** en tu zona (`app.user_timezone`): `"2026-10-13 14:30"`,
  `"13/10 14:30"`, `"14:30"` (hoy), `"ayer 22:00"`, `"mañana 11:00"`. Montos con
  coma o punto (`-32,5`). Sin `--closed` es ahora; sin `--opened`, la misma hora
  del cierre. Una fecha sin año es de este año. Si así queda en el futuro:
  - a más de medio año, es del año pasado (el 1 de enero, `"31/12 23:00"` es de
    ayer);
  - si no, es un error de tipeo y te lo dice con el año a la vista (el 13/10,
    `"14/10 10:00"` da "La hora de cierre (14/10/2026 10:00) está en el
    futuro"), en vez de guardarla hace un año sin avisar.

  En un evento, la más cercana. `"29/02"` sin año, cuando ni este año ni los de
  al lado son bisiestos, pide el año ("La fecha 29/02 no existe en estos años:
  poné el año"). Si escribís una hora que todavía no llegó (`"22:00"` a la
  01:00), te avisa y te sugiere `"ayer 22:00"`. Una hora hasta 5 minutos
  adelante del reloj de la compu se guarda como "ahora" (así los frenos la
  cuentan ya) y te lo dice.
- **Análisis ligado**: cada vez que analizás (`make cli`, con datos reales) el
  análisis se guarda con un número. Al cargar la operación, la app le adjunta
  sola el último análisis de esa moneda de las 24 h antes de abrir (régimen,
  score y plan del momento, para el informe si algo sale mal); `--analysis 7`
  elige uno (tiene que ser anterior a la apertura) y `--analysis none`, ninguno.
  Un análisis **de prueba** (con `--offline` y `--db`, o con `--pnl-today` /
  `--trades-today`) queda marcado "(datos sintéticos)" o "(números de prueba)":
  no se liga solo (con `--analysis 7` sí, con un aviso) y en el informe del
  incidente sale marcado y no cuenta para "el plan decía No operar" ni "contra
  el régimen".
- **Stop movido**: `update N --risk X` cambia lo que perdés si toca el stop sin
  cerrar la posición (la hora de apertura y las operaciones del día no
  cambian). Con el stop en la entrada o en ganancia poné `--risk 0`: ya no
  cuenta para el límite del día ni para el tope de posiciones correlacionadas.
  El **riesgo previsto** al abrir queda anotado igual y la salida lo dice: si
  algo sale mal, el informe del incidente lo usa para decir "moviste el stop a
  la entrada" o "agrandaste el stop". `make trades` muestra los dos ("0,00 USD
  (previsto 40,00 USD)"). También `--entry` y `--note` (`--note ""` la borra).
  `open --risk 0` también vale.
- Después de cargar algo muestra en una línea cómo quedaron los frenos.
- **Confirmación con `--yes`** (casi siempre es un error de tipeo; los números
  en `app.manual_input`). No se guardan sin `--yes`:
  - un resultado (`add`, `close`) más grande que tu saldo o que 3 veces tu
    riesgo por operación (150 USD en el agresivo);
  - una posición abierta (`open`, `update --risk`) con un riesgo más grande que
    tu saldo o que esos 150 USD (un 400 en vez de 40 te dejaría en "No operar"
    todo el día, porque el riesgo abierto cuenta en el límite del día);
  - un retiro más grande que el saldo que tenías en esa fecha (un cero de más
    deja el saldo en negativo y desarma las cuentas del freno del mes);
  - un evento de más de 7 días.

  Por ejemplo, `make trade ARGS="add --symbol BTC --side long --pnl -325"`
  explica "¿No será un error de tipeo?" y no guarda nada; si de verdad perdiste
  325, repetilo con `--yes` al final. Si el comando que falla era el primero de
  una `--db` nueva, tampoco queda la base creada.

Los frenos (todos los números en `config.yaml`, `circuit_breakers`, y en tu perfil):

| Freno | Cuándo salta | Qué hace |
|---|---|---|
| Pérdida del día | perdiste el 50 % de tu límite diario (50 de 100 USD) | riesgo × 0,5 hasta las 00:00 |
| Límite del día | perdiste el 100 % del límite diario | no hay planes hasta las 00:00 |
| Operaciones del día | llegaste al máximo de operaciones por día de tu perfil (3 en el agresivo) | estado ROJO: no hay planes hasta las 00:00 ("ya hiciste 3 de 3 operaciones permitidas hoy") |
| Pérdidas seguidas | 3 operaciones perdidas seguidas | pausa de 4 h; después la cuenta arranca de cero |
| Pérdida de la semana | perdiste el 60 % de la billetera mensual (180 de 300 USD) | riesgo × 0,5 hasta el lunes |
| Caída del mes | la billetera cayó 70 % desde su máximo del mes | **bloqueo total** (ver abajo) |
| Evento de alta volatilidad | hay un evento cargado activo | no hay planes hasta su hora de fin |

- La pérdida del día y de la semana es el **neto** de lo que cerraste en ese
  período (ganancias menos pérdidas); el día y la semana (desde el lunes) son
  los de tu zona horaria, y el mes es el mes calendario.
- **Un freno del día o de la semana que saltó sigue hasta que termina el
  período**, aunque después ganes (`latch_until_period_end`). Ejemplo: perdés
  60 USD (riesgo × 0,5) y después ganás 30: el neto es -30, pero el freno sigue
  hasta las 00:00, y se muestra "Hoy llegaste a perder 60 USD (ahora vas -30
  USD)". El de la semana sigue hasta el lunes.
- **Pausa por pérdidas seguidas**: si perdés otra vez durante la pausa, se
  estira 4 h desde esa pérdida (`extend_cooldown_on_loss`), aunque en el medio
  hayas ganado alguna. Ejemplo: tercera pérdida a las 14:00, pausa hasta las
  18:00; otra pérdida a las 16:00 (una posición que tenías abierta), pausa
  hasta las 20:00. El mensaje dice qué racha empezó la pausa y qué pérdida la
  estiró.
- **Retiros y depósitos en la caída del mes**:
  - Un **retiro** baja el máximo en la misma proporción que el saldo, así
    sacar plata no cambia el % de caída. Ejemplo: arrancaste el mes con 300,
    perdiste 100 (saldo 200, caída 33 %) y retirás 100: el saldo queda en 100,
    el máximo baja a 150 y la caída sigue en 33 %.
  - Un **depósito** primero "devuelve" lo que retiraste en el mes, sin cambiar
    el % de caída (`deposit_refills_withdrawals`): retirar y volver a depositar
    no borra la caída. Ejemplo: máximo 300 y saldo 100 (66,67 % abajo); retirás
    100 y después depositás 100: sigue en 66,67 %. Lo que pase de lo retirado
    sube el máximo en lo mismo que depositaste: la pérdida de antes sigue
    contando y depositar no esquiva el bloqueo. Ejemplo: máximo 300, saldo 100,
    sin retiros, y depositás 200: saldo 300, máximo 500, caída 40 %.
  - Si en el mes primero depositás y después retirás, se aplican las dos
    reglas en ese orden. Ejemplo: máximo 300 y saldo 100; depositás 90 (no
    había retiros: saldo 190, máximo 390, caída 51,28 %) y después retirás 90
    (saldo 100, el máximo baja en la misma proporción, a 205,26, y la caída
    sigue en 51,28 %).
- Si nunca anotaste el saldo, la app supone que empezaste el mes con la
  billetera mensual del perfil (300 USD) y te avisa: anotalo con `balance`.
- Los multiplicadores se suman a los del régimen (día × semana = × 0,25) y van
  antes del tope por lo que queda del límite del día; si el riesgo queda debajo
  del mínimo operable, No operar. Cada ajuste se muestra con su motivo.
- **Límite del día con posiciones abiertas**: "lo que queda del límite de
  pérdida del día" también descuenta lo que perderías si saltan los stops de
  tus posiciones abiertas (`daily_cap_includes_open_risk`). Ejemplo
  (agresivo, límite 100 USD): perdiste 30 hoy y tenés una posición abierta con
  riesgo 40: el plan nuevo puede arriesgar como mucho 30 (100 − 30 − 40). Si eso
  queda debajo del mínimo operable (15 USD), No operar. Una posición con el
  stop en ganancia cuenta 0 (cargala con `update N --risk 0`); una cobertura en
  la misma moneda cuenta (pueden saltar los dos stops). La consola lo dice en el
  plan: "Límite del día: hoy perdiste 30 USD y tenés 40 USD en juego en
  posiciones abiertas (lo que perdés si saltan sus stops): te quedan 30 USD de
  100 USD". Si ya llegaste al límite del día, el plan no muestra esa línea (ya
  dice "llegaste a tu límite"); si después ganaste algo, aclara "antes llegaste
  a perder X": eso mantiene trabados los frenos que ya saltaron, pero lo que
  queda se cuenta desde lo que perdés ahora.
- **Bloqueo total**: la app escribe un informe del incidente en
  `data/incidents/` (qué operaciones, en qué régimen, qué decía el score y qué
  falló) y el archivo `data/LOCKED`. Mientras exista, la consola muestra un
  cartel rojo y **no arma ningún plan** (el resto del análisis sale igual).
  Cargar operaciones no lo escribe (así podés terminar de cargar, por ejemplo,
  un depósito que faltaba): la línea de los frenos avisa que "al analizar o con
  make safety la app se va a bloquear"; lo escriben `make safety` y el análisis.
  Si no se puede escribir el archivo LOCKED (permisos de la carpeta, disco
  lleno), `make safety` y `make cli` dicen "No se pudo escribir el archivo
  LOCKED (...): la app no arma planes hasta que se pueda escribir" y salen con
  error; el informe queda guardado una sola vez, aunque lo intentes varias
  veces.
- Una pérdida que cargás tarde pero que cerró **antes** del bloqueo es parte de
  ese incidente: no cuenta como una caída nueva después de desbloquear.
- **Para salir del bloqueo**: leé el informe y borrá `data/LOCKED` a mano (la
  app no lo borra nunca). Desde ahí la caída se vuelve a medir desde el saldo
  del momento del bloqueo, como si empezara un mes nuevo
  (`restart_after_unlock`). Ejemplo: la billetera bajó de 300 a 90 y se
  bloqueó; borrás LOCKED: se vuelve a bloquear solo si cae otro 70 % desde 90
  (a 27 USD). La sección de los frenos lo muestra: "la caída se mide de nuevo
  desde el bloqueo del 12/10 11:37 (que ya desbloqueaste)", y abajo, en gris,
  la caída de todo el mes, que ya no cuenta para el bloqueo. Si se vuelve a
  bloquear en el mismo mes, el cartel, el motivo, la fila del freno y la de la
  billetera muestran la misma caída (la medida desde el bloqueo anterior, por
  ejemplo "85 USD → 20 USD, 76,47 %"); la de todo el mes va aparte, en gris.
- El bloqueo queda anotado en la base. Si hay un LOCKED de este mes que no
  quedó anotado (lo escribió una versión vieja de la app o la base falló), la
  próxima corrida lo anota sola con su fecha y lo dice
  (`backfill_unrecorded_lock`): así, cuando lo borres, la caída se mide desde
  ese bloqueo. Mientras la base siga fallando, la consola lo avisa en cada
  corrida.
- Una caída de más de 100 % solo pasa si el saldo calculado da negativo (casi
  siempre un dato mal cargado, por ejemplo un resultado o un retiro con un cero
  de más): se muestra "la billetera cayó más del 100 %" con un aviso de revisar
  lo cargado, en los frenos, en las dos consolas, en el LOCKED y en el
  informe.
- **Correlación**: la consola muestra la matriz de correlación de los
  retornos de 1 h de los últimos 30 días entre la moneda, tus favoritas y las de
  tus posiciones abiertas (`*` = 0,8 o más: se mueven casi igual). Si tenés una
  posición abierta y el plan nuevo va en la **misma dirección** en una moneda
  muy correlacionada (o la misma), es casi la misma apuesta: entre todas no
  pueden arriesgar más de 1,5 × tu riesgo por operación (75 USD en el agresivo).
  El plan nuevo se achica a lo que entra; si queda debajo del mínimo operable,
  No operar. Una posición en la misma moneda pero al revés es cobertura: no
  suma al tope de posiciones correlacionadas (1,5 × tu riesgo), pero su riesgo
  sí cuenta en lo que te queda del límite del día, porque pueden saltar los dos
  stops (el aviso es solo para la misma moneda).
- El margen de un plan nunca pasa del saldo de tu billetera que muestran los
  frenos (el anotado o, si nunca anotaste uno, el supuesto: la billetera
  mensual del perfil al empezar el mes más tus resultados), y la liquidación
  "modelo Nexo" usa ese saldo (con "saldo supuesto" si lo es). Si el saldo es 0
  o menos: No operar ("no hay plata para poner de margen"); los niveles quedan
  como referencia, pero las filas de margen, apalancamiento y liquidación dicen
  "sin saldo" (no inventan una billetera).
- `--trades-today` y `--pnl-today` siguen existiendo para probar: el plan usa
  esos números y lo avisa; los frenos siguen mirando lo cargado.
- `make cli ARGS="--offline"` usa otra base de prueba (al lado de los modelos
  sintéticos, con su propio LOCKED): las pruebas nunca tocan tu estado real.
  `make trade ARGS="... --offline"` carga en esa base. `--db RUTA` (en las dos
  consolas) usa otra base, con **su propio** LOCKED y su carpeta de informes al
  lado, con el nombre de la base: `--db data/prueba.db` usa `data/prueba.LOCKED`
  y `data/prueba_incidents/`. Así una base de prueba en `data/` nunca bloquea
  (ni desbloquea) la real. Con la ruta de la base real (`data/analyzer.db`) usa
  `data/LOCKED` y `data/incidents/` de siempre.
- **`--db` con una ruta relativa** en `make cli`, `make trade`, `make trades` y
  `make safety`: es relativa a la carpeta desde donde corriste `make` (la raíz,
  `backend/` o `frontend/`), no a `backend/`. Con una base que no existe,
  `make cli`, `make trades`, `make safety`, `close`, `update` y `delete` dicen
  "No existe la base ..." y no crean nada (¿ruta mal escrita?); el primer
  `make trade` que carga algo la crea y lo avisa (si ese comando falla, por
  ejemplo porque falta `--yes`, no queda la base creada).
- Las bases (`*.db`), los archivos `LOCKED` y `*.LOCKED` y las carpetas
  `incidents/` y `*_incidents/` están en `.gitignore` en cualquier carpeta del
  proyecto: tus datos nunca se suben a git.
- En JSON (`make cli ARGS="--json"`), los frenos van en `"safety"` (cada freno
  con umbral, valor actual y progreso, para el panel de riesgo del Hito 9) y la
  matriz con la exposición en `"correlation"`; si la app está bloqueada,
  `"plan": null` y `"locked": true`.

## Configuración

- **`config.yaml`**: todas las reglas, umbrales y pesos (perfiles de riesgo,
  ajustes por régimen, circuit breakers, comisiones, etc.), con comentarios. Se
  valida al arrancar: si algo no cierra, la app no levanta y dice por qué.
- **`.env`**: solo secretos (`ANTHROPIC_API_KEY`, `CRYPTOPANIC_API_KEY`). Está
  en `.gitignore`. En la fase 1 **no** se usan API keys de exchanges.

## Estructura

```
├── backend/
│   ├── app/
│   │   ├── config.py      # carga y valida config.yaml y .env
│   │   ├── main.py        # app FastAPI
│   │   ├── exchanges/     # interfaz común + Binance, Bybit (fallback), Nexo
│   │   ├── data/          # caché, Fear & Greed, CoinGecko, noticias, snapshot
│   │   ├── indicators/    # EMA, VWAP, RSI, MACD, ATR, Bollinger, S/R, tendencia, features
│   │   ├── regime/        # HMM: BIC, forward filtering, etiquetas, estabilidad, disco
│   │   ├── scoring/       # score de confluencia
│   │   ├── risk/          # plan, sizing, comisiones, liquidación, perfiles, simuladores
│   │   ├── safety/        # circuit breakers, lock file, incidentes, correlación
│   │   ├── validation/    # walk-forward, benchmarks, métricas, stress test
│   │   ├── llm/           # prompt, esquema y validación
│   │   ├── storage/       # SQLite
│   │   ├── services/      # orquesta un análisis completo
│   │   └── api/           # endpoints (sin lógica de negocio)
│   ├── scripts/           # CLI, carga del estado (trades), entrenamiento del régimen, walk-forward
│   └── tests/
├── frontend/              # React + Vite + TypeScript + Tailwind
│   └── src/ (components, charts, hooks, pages, lib)
├── docs/                  # investigación y documentación
├── data/                  # SQLite, modelos, lock e informes (no se versiona)
├── config.yaml
├── .env.example
└── Makefile
```

Cada módulo tiene un docstring que explica qué hace y en qué hito se
implementa.

## Fuentes de datos

Públicas, sin cuenta de exchange: Binance (velas, funding, open interest, ratio
long/short), Bybit como fallback, alternative.me (Fear & Greed), CoinGecko
(dominancia), CryptoPanic o RSS de CoinDesk/Cointelegraph (noticias). El precio
de Nexo no tiene acceso público confirmado; ver
[docs/nexo-api-research.md](docs/nexo-api-research.md).
