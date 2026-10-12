"""Cada componente del score de confluencia, normalizado a [-1, +1].

Funciones puras (sin I/O): reciben valores sueltos y la parte de
``config.yaml > scoring`` que les toca, y devuelven un ``ComponentReading``
(valor, si había datos, y textos en castellano que explican el porqué). El peso
de cada componente lo aplica ``confluence.compute_score``.

Componentes
    - **Tendencia** (``trend_component``): ``"alcista"`` +1, ``"bajista"`` -1,
      ``"lateral"`` 0. Se usa para 4h, 1h y 15m.
    - **Momentum** (``momentum_component``): por cada timeframe de
      ``scoring.momentum.timeframes`` hay dos sub-puntajes, RSI y MACD; el valor
      es el promedio de los que estén disponibles.

      * RSI (``rsi_score``), con ``n = rsi_neutral``, ``ob = rsi_overbought`` y
        ``os = rsi_oversold``: entre ``os`` y ``ob`` es lineal,
        ``(rsi - n) / (ob - n)`` arriba de ``n`` y ``(rsi - n) / (n - os)``
        debajo (vale +1 en ``ob``, 0 en ``n`` y -1 en ``os``). En sobrecompra el
        aporte alcista "se recorta": baja en línea recta de +1 en ``ob`` a 0 en
        100. En sobreventa, espejado: de -1 en ``os`` a 0 en 0.
      * MACD (``macd_score``): ``0,5 × signo(macd - señal) + 0,5 × signo(macd)``.
    - **Derivados** (``derivatives_component``): suma de tres partes, recortada
      a [-1, +1]. ``f`` es el funding en % **por 8 h** (``funding_pct_per_8h``
      lo normaliza con el intervalo que informa el exchange: los umbrales de la
      config están en "% cada 8 h").

      * Funding (``funding_part``): ``f >= funding_very_high_pct`` -1;
        ``f >= funding_high_pct`` -0,5; ``f <= funding_low_pct`` +0,5; si no, 0.
      * Interés abierto (``open_interest_part``, decisión del usuario): solo
        cuenta con el OI subiendo (``>= oi_rising_pct``) y un funding que ya
        marca un lado sobrecargado. ``f >= funding_high_pct``: longs
        sobrecargados con el OI subiendo, riesgo de caída (-0,5);
        ``f <= funding_low_pct``: shorts sobrecargados con el OI subiendo
        (+0,5). Con el funding normal (por ejemplo +0,01 %) la suba del OI no
        cuenta (0). Sin funding no se puede interpretar.
      * Ratio long/short de cuentas (``long_short_part``):
        ``>= long_short_crowded_long`` -0,5; ``<= long_short_crowded_short``
        +0,5.
    - **Sentimiento** (``sentiment_component``): Fear & Greed (``fear_greed_part``)
      en modo ``"contrarian"`` (el miedo suma al long, la codicia resta, a pleno
      desde los extremos ``fear_greed_extreme_fear`` / ``extreme_greed``) o
      ``"trend"`` (al revés). Las noticias pesan ``news_share`` del componente
      si hay un balance de noticias clasificado; con ``news_share: 0``
      (decisión del usuario) no mueven el score.

Datos faltantes
    Un componente sin ningún dato queda con valor 0 y ``available=False``. Una
    parte que falta suma 0 (en derivados) o no entra al promedio (en momentum) y
    se nombra en ``missing_parts``. Los pesos no se reparten: que falten datos
    empuja hacia "no operar", que es el lado conservador.

Los números de los textos van con coma decimal ("RSI 15m en 62,3").
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from app.config import DerivativesScoringConfig, MomentumScoringConfig, SentimentScoringConfig

# Escala del RSI (no son umbrales: son los límites del indicador).
RSI_MIN = 0.0
RSI_MAX = 100.0
# Escala de Fear & Greed (0 a 100) y su punto medio.
FEAR_GREED_MIN = 0.0
FEAR_GREED_MAX = 100.0
FEAR_GREED_NEUTRAL = 50.0
# Unidad de los umbrales de funding en config.yaml: "% por período de 8 h".
FUNDING_REFERENCE_HOURS = 8.0

TREND_VALUES: dict[str, float] = {"alcista": 1.0, "bajista": -1.0, "lateral": 0.0}


@dataclass(frozen=True)
class ComponentReading:
    """Valor de un componente antes de aplicarle el peso."""

    value: float  # en [-1, +1]; 0 si no hay datos
    available: bool  # False = no había ningún dato para este componente
    details: tuple[str, ...]  # explicación en castellano, una línea por parte
    missing_parts: tuple[str, ...] = ()  # partes sin datos ("RSI 1h", "ratio long/short")


# --- utilidades ----------------------------------------------------------------


def clean_number(x: object) -> float | None:
    """``float`` finito o ``None`` (``None``, ``NaN``, infinitos y textos dan ``None``)."""
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _clip(x: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, x))


def _sign(x: float) -> float:
    return 1.0 if x > 0 else -1.0 if x < 0 else 0.0


def fmt_num(x: float, decimals: int = 2, *, signed: bool = False) -> str:
    """Número con coma decimal y sin ceros de más: ``0.5`` → ``"0,5"``.

    Con ``signed=True`` los positivos llevan ``+`` (``"+0,5"``); el cero va sin
    signo. Nunca devuelve ``"-0"``.
    """
    text = f"{x:.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in ("-0", ""):
        text = "0"
    if signed and not text.startswith("-") and text != "0":
        text = "+" + text
    return text.replace(".", ",")


def score_beside_threshold(value: float, threshold: float, decimals: int = 1) -> tuple[float, int]:
    """``(valor, decimales)`` para mostrar un score sin que el redondeo lo pegue al umbral.

    Si con ``decimals`` decimales el score se ve igual que el umbral pero no lo
    es (``49.96`` contra ``50`` se vería "+50"), se muestra con dos decimales
    más y redondeado hacia el lado en que está: hacia abajo si está debajo
    (``49.96`` → "+49,96"; ``29.999999999`` → "+29,999") y hacia arriba si está
    arriba. Así un mensaje nunca dice "tiene que ser +50 y está en +50". Si no
    hay riesgo de confusión devuelve ``(value, decimals)`` tal cual.
    """
    if value == threshold or f"{value:.{decimals}f}" != f"{threshold:.{decimals}f}":
        return value, decimals
    more = decimals + 2
    factor = 10.0**more
    scaled = round(value * factor, 6)  # limpia el ruido de float antes de truncar
    shown = (math.floor(scaled) if value < threshold else math.ceil(scaled)) / factor
    if shown == threshold:  # pegado al umbral aun con más decimales: un paso al costado
        shown = threshold - 1.0 / factor if value < threshold else threshold + 1.0 / factor
    return shown, more


def _contrib(value: float) -> str:
    """Aporte de una parte entre paréntesis: ``(+0,5)``, ``(-1)``, ``(0)``."""
    return f"({fmt_num(value, 2, signed=True)})"


# --- tendencia -------------------------------------------------------------------


def trend_value(trend: str | None) -> float | None:
    """+1 / -1 / 0 según la etiqueta; ``None`` si no hay tendencia.

    Lanza ``ValueError`` si la etiqueta no es ``"alcista"``, ``"bajista"`` ni
    ``"lateral"``.
    """
    if trend is None:
        return None
    if trend not in TREND_VALUES:
        raise ValueError(f"Tendencia desconocida: {trend!r} (alcista, bajista o lateral)")
    return TREND_VALUES[trend]


def trend_component(trend: str | None, timeframe: str) -> ComponentReading:
    """Componente de tendencia de un timeframe (``None`` = sin datos)."""
    value = trend_value(trend)
    if value is None:
        return ComponentReading(
            0.0,
            False,
            (f"Sin tendencia de {timeframe} (faltan velas o EMAs): cuenta 0.",),
            (f"tendencia {timeframe}",),
        )
    if value > 0:
        text = f"Tendencia alcista en {timeframe}: suma {_contrib(value)}."
    elif value < 0:
        text = f"Tendencia bajista en {timeframe}: resta {_contrib(value)}."
    else:
        text = f"Tendencia lateral en {timeframe}: ni suma ni resta {_contrib(value)}."
    return ComponentReading(value, True, (text,))


# --- momentum --------------------------------------------------------------------


def rsi_score(rsi: float, neutral: float, overbought: float, oversold: float) -> float:
    """Sub-puntaje del RSI en [-1, +1] (fórmula en el docstring del módulo).

    El RSI se acota a [0, 100]. Lanza ``ValueError`` si no se cumple
    ``oversold < neutral < overbought``.
    """
    if not oversold < neutral < overbought:
        raise ValueError("RSI: tiene que cumplirse oversold < neutral < overbought")
    r = min(RSI_MAX, max(RSI_MIN, float(rsi)))
    if r > overbought:
        # Sobrecompra: el aporte alcista se recorta hasta 0 en 100.
        return _clip((RSI_MAX - r) / (RSI_MAX - overbought), 0.0, 1.0)
    if r < oversold:
        # Sobreventa: el aporte bajista se recorta hasta 0 en 0.
        return -_clip((r - RSI_MIN) / (oversold - RSI_MIN), 0.0, 1.0) + 0.0  # sin "-0.0"
    if r >= neutral:
        return (r - neutral) / (overbought - neutral)
    return (r - neutral) / (neutral - oversold)


def macd_score(macd: float, signal: float) -> float:
    """``0,5 × signo(macd - señal) + 0,5 × signo(macd)``: -1, -0,5, 0, +0,5 o +1."""
    return 0.5 * _sign(macd - signal) + 0.5 * _sign(macd)


def _rsi_text(tf: str, rsi: float, score: float, cfg: MomentumScoringConfig) -> str:
    head = f"RSI {tf} en {fmt_num(rsi, 1)}"
    if rsi > cfg.rsi_overbought:
        why = "sobrecompra, el aporte alcista se recorta"
    elif rsi < cfg.rsi_oversold:
        why = "sobreventa, el aporte bajista se recorta"
    elif rsi > cfg.rsi_neutral:
        why = f"arriba de {fmt_num(cfg.rsi_neutral)}, a favor de la suba"
    elif rsi < cfg.rsi_neutral:
        why = f"debajo de {fmt_num(cfg.rsi_neutral)}, a favor de la baja"
    else:
        why = "neutral"
    return f"{head}: {why} {_contrib(score)}."


def _macd_text(tf: str, macd: float, signal: float, score: float) -> str:
    vs_signal = (
        "arriba de su señal"
        if macd > signal
        else "debajo de su señal"
        if macd < signal
        else "igual a su señal"
    )
    vs_zero = "arriba de 0" if macd > 0 else "debajo de 0" if macd < 0 else "en 0"
    return f"MACD {tf} {vs_signal} y {vs_zero} {_contrib(score)}."


def momentum_component(
    rsi_by_tf: Mapping[str, float | None],
    macd_by_tf: Mapping[str, float | None],
    signal_by_tf: Mapping[str, float | None],
    cfg: MomentumScoringConfig,
) -> ComponentReading:
    """Promedio de los sub-puntajes de RSI y MACD disponibles en ``cfg.timeframes``.

    El MACD necesita la línea y la señal; si falta alguna, ese MACD no cuenta.
    Sin ningún sub-puntaje el componente queda sin datos (valor 0).
    """
    scores: list[float] = []
    details: list[str] = []
    missing: list[str] = []
    for tf in cfg.timeframes:
        rsi = clean_number(rsi_by_tf.get(tf))
        if rsi is None:
            missing.append(f"RSI {tf}")
        else:
            s = rsi_score(rsi, cfg.rsi_neutral, cfg.rsi_overbought, cfg.rsi_oversold)
            scores.append(s)
            details.append(_rsi_text(tf, rsi, s, cfg))
        macd = clean_number(macd_by_tf.get(tf))
        signal = clean_number(signal_by_tf.get(tf))
        if macd is None or signal is None:
            missing.append(f"MACD {tf}")
        else:
            s = macd_score(macd, signal)
            scores.append(s)
            details.append(_macd_text(tf, macd, signal, s))
    if not scores:
        tfs = " y ".join(cfg.timeframes) if cfg.timeframes else "ningún timeframe"
        return ComponentReading(
            0.0, False, (f"Sin RSI ni MACD en {tfs}: el momentum cuenta 0.",), tuple(missing)
        )
    value = sum(scores) / len(scores)
    if missing:
        verb = "Falta" if len(missing) == 1 else "Faltan"
        details.append(f"{verb} {', '.join(missing)}: se promedia con lo que hay.")
    return ComponentReading(value, True, tuple(details), tuple(missing))


# --- derivados -------------------------------------------------------------------


def funding_pct_per_8h(rate: float, interval_hours: float) -> float:
    """Funding como fracción por intervalo → % por 8 h (la unidad de la config).

    Ej.: 0,0003 cada 4 h = 0,03 % cada 4 h = 0,06 % cada 8 h. Se redondea a
    10 decimales para que un funding justo en un umbral (0,05 %) no quede
    0,0499999... por el redondeo de float. Lanza ``ValueError`` si el
    intervalo no es positivo.
    """
    if not interval_hours > 0:
        raise ValueError(f"El intervalo de funding tiene que ser > 0 (llegó {interval_hours!r})")
    return round(rate * 100.0 * FUNDING_REFERENCE_HOURS / interval_hours, 10)


def funding_part(f_pct_8h: float, cfg: DerivativesScoringConfig) -> float:
    """-1 si el funding es muy alto, -0,5 si es alto, +0,5 si es negativo, si no 0."""
    if f_pct_8h >= cfg.funding_very_high_pct:
        return -1.0
    if f_pct_8h >= cfg.funding_high_pct:
        return -0.5
    if f_pct_8h <= cfg.funding_low_pct:
        return 0.5
    return 0.0


def open_interest_part(
    oi_change_pct: float, f_pct_8h: float, cfg: DerivativesScoringConfig
) -> float:
    """Aporte del OI: solo cuenta si sube y el funding ya marca un lado sobrecargado.

    Con ``oi_change_pct >= oi_rising_pct``: -0,5 si ``f_pct_8h >=
    funding_high_pct`` (longs sobrecargados), +0,5 si ``f_pct_8h <=
    funding_low_pct`` (shorts sobrecargados) y 0 con el funding normal. Con el
    OI quieto o bajando, 0. ``f_pct_8h`` es el funding en % por 8 h.
    """
    if oi_change_pct < cfg.oi_rising_pct:
        return 0.0
    if f_pct_8h >= cfg.funding_high_pct:
        return -0.5
    if f_pct_8h <= cfg.funding_low_pct:
        return 0.5
    return 0.0


def long_short_part(ratio: float, cfg: DerivativesScoringConfig) -> float:
    """-0,5 con exceso de cuentas en long; +0,5 con exceso en short; si no, 0."""
    if ratio >= cfg.long_short_crowded_long:
        return -0.5
    if ratio <= cfg.long_short_crowded_short:
        return 0.5
    return 0.0


def _funding_text(rate: float, interval: float, f: float, part: float) -> str:
    per_8h = f"{fmt_num(f, 4)} % cada {fmt_num(FUNDING_REFERENCE_HOURS)} h"
    if math.isclose(interval, FUNDING_REFERENCE_HOURS):
        head = f"Funding {per_8h}"
    else:
        head = f"Funding {fmt_num(rate * 100.0, 4)} % cada {fmt_num(interval)} h (= {per_8h})"
    if part <= -1.0:
        why = "muy alto, demasiados longs pagando por mantenerse, riesgo de caída"
    elif part < 0:
        why = "alto, longs sobrecargados, riesgo de caída"
    elif part > 0:
        why = "negativo, shorts sobrecargados, puede rebotar"
    else:
        why = "normal"
    return f"{head}: {why} {_contrib(part)}."


def _oi_text(oi: float, part: float, cfg: DerivativesScoringConfig) -> str:
    head = f"Interés abierto {fmt_num(oi, 1, signed=True)} % en {cfg.oi_lookback_hours} h"
    if oi < cfg.oi_rising_pct:
        return (
            f"{head}: no sube {fmt_num(cfg.oi_rising_pct, 1)} % o más, no indica nada "
            f"{_contrib(part)}."
        )
    if part < 0:
        text = "con funding alto: longs sobrecargados con el OI subiendo, riesgo de caída"
    elif part > 0:
        text = "con funding negativo: shorts sobrecargados con el OI subiendo, riesgo de suba"
    else:
        text = "con funding normal: ningún lado está sobrecargado, así que la suba del OI no cuenta"
    return f"{head} {text} {_contrib(part)}."


def _long_short_text(ratio: float, part: float) -> str:
    head = f"Ratio long/short de cuentas {fmt_num(ratio, 2)}"
    if part < 0:
        why = "demasiadas cuentas en long, riesgo de caída"
    elif part > 0:
        why = "demasiadas cuentas en short, riesgo de suba"
    else:
        why = "equilibrado"
    return f"{head}: {why} {_contrib(part)}."


def derivatives_component(
    funding_rate: float | None,
    funding_interval_hours: float | None,
    oi_change_pct: float | None,
    long_short_ratio: float | None,
    cfg: DerivativesScoringConfig,
    *,
    default_interval_hours: float,
) -> ComponentReading:
    """Funding + interés abierto + ratio long/short, recortado a [-1, +1].

    ``funding_rate`` es la tasa como fracción por intervalo (como la manda el
    exchange) y ``funding_interval_hours`` el intervalo que informa el exchange;
    si no viene (o no es positivo) se usa ``default_interval_hours``
    (``fees.funding_interval_hours``). La parte del OI necesita el funding para
    saber quién está sobrecargado, y con el funding normal no cuenta
    (``open_interest_part``). Un ratio long/short que no es positivo se toma
    como dato faltante.
    """
    rate = clean_number(funding_rate)
    interval = clean_number(funding_interval_hours)
    if interval is None or interval <= 0:
        interval = default_interval_hours
    oi = clean_number(oi_change_pct)
    ratio = clean_number(long_short_ratio)
    if ratio is not None and ratio <= 0:
        ratio = None

    parts: list[float] = []
    details: list[str] = []
    missing: list[str] = []
    f: float | None = None
    if rate is None:
        missing.append("funding")
    else:
        f = funding_pct_per_8h(rate, interval)
        p = funding_part(f, cfg)
        parts.append(p)
        details.append(_funding_text(rate, interval, f, p))
    if oi is None:
        missing.append("interés abierto")
    elif f is None:
        missing.append("interés abierto")
        details.append(
            f"Interés abierto {fmt_num(oi, 1, signed=True)} % en {cfg.oi_lookback_hours} h: "
            "sin funding no se puede saber quién está sobrecargado (cuenta 0)."
        )
    else:
        p = open_interest_part(oi, f, cfg)
        parts.append(p)
        details.append(_oi_text(oi, p, cfg))
    if ratio is None:
        missing.append("ratio long/short")
    else:
        p = long_short_part(ratio, cfg)
        parts.append(p)
        details.append(_long_short_text(ratio, p))

    if not parts:
        return ComponentReading(
            0.0,
            False,
            (
                *details,
                "Sin funding, interés abierto ni ratio long/short: los derivados cuentan 0.",
            ),
            tuple(missing),
        )
    raw = sum(parts)
    value = _clip(raw)
    if value != raw:
        details.append(
            f"Las partes suman {fmt_num(raw, 2, signed=True)}; "
            f"se recorta a {fmt_num(value, 2, signed=True)}."
        )
    if missing:
        tail = "esa parte cuenta 0" if len(missing) == 1 else "esas partes cuentan 0"
        details.append(f"Sin {', '.join(missing)}: {tail}.")
    return ComponentReading(value, True, tuple(details), tuple(missing))


# --- sentimiento -----------------------------------------------------------------


def _check_fear_greed_cfg(cfg: SentimentScoringConfig) -> None:
    fear, greed = cfg.fear_greed_extreme_fear, cfg.fear_greed_extreme_greed
    if not FEAR_GREED_MIN <= fear < FEAR_GREED_NEUTRAL < greed <= FEAR_GREED_MAX:
        raise ValueError(
            "scoring.sentiment: tiene que cumplirse "
            "0 <= fear_greed_extreme_fear < 50 < fear_greed_extreme_greed <= 100"
        )


def fear_greed_part(value: float, cfg: SentimentScoringConfig) -> float:
    """Aporte de Fear & Greed en [-1, +1] según ``fear_greed_mode``.

    ``"contrarian"``: con ``g <= 50`` vale ``(50 - g) / (50 - miedo extremo)``
    (acotado a [0, 1]: +1 desde el miedo extremo para abajo); con ``g > 50``,
    ``-(g - 50) / (codicia extrema - 50)`` (acotado: -1 desde la codicia
    extrema). ``"trend"`` es lo mismo con el signo cambiado.
    """
    _check_fear_greed_cfg(cfg)
    g = min(FEAR_GREED_MAX, max(FEAR_GREED_MIN, float(value)))
    n = FEAR_GREED_NEUTRAL
    if g <= n:
        contrarian = _clip((n - g) / (n - cfg.fear_greed_extreme_fear), 0.0, 1.0)
    else:
        contrarian = -_clip((g - n) / (cfg.fear_greed_extreme_greed - n), 0.0, 1.0)
    value = contrarian if cfg.fear_greed_mode == "contrarian" else -contrarian
    return value + 0.0  # sin "-0.0"


def fear_greed_label(value: float, cfg: SentimentScoringConfig) -> str:
    """ "miedo extremo", "miedo", "neutral", "codicia" o "codicia extrema" según la config."""
    if value <= cfg.fear_greed_extreme_fear:
        return "miedo extremo"
    if value >= cfg.fear_greed_extreme_greed:
        return "codicia extrema"
    if value < FEAR_GREED_NEUTRAL:
        return "miedo"
    if value > FEAR_GREED_NEUTRAL:
        return "codicia"
    return "neutral"


def sentiment_component(
    fear_greed: float | None, news_balance: float | None, cfg: SentimentScoringConfig
) -> ComponentReading:
    """Fear & Greed y, si ``news_share > 0`` y hay un balance de noticias, noticias.

    ``news_balance`` (de -1 = todas malas a +1 = todas buenas) es opcional:
    con ``news_balance`` y ``news_share > 0`` el valor es
    ``(1 - news_share) × F&G + news_share × noticias``; sin balance, o con
    ``news_share: 0`` (decisión del usuario: las noticias las comenta la IA y
    no mueven el score), cuenta solo Fear & Greed. Si falta Fear & Greed su
    parte vale 0 (no se reparte su peso). Lanza ``ValueError`` si
    ``news_balance`` está fuera de [-1, +1].
    """
    g = clean_number(fear_greed)
    news = clean_number(news_balance)
    if news is not None and not -1.0 <= news <= 1.0:
        raise ValueError(f"news_balance tiene que estar entre -1 y +1 (llegó {news_balance!r})")
    share = cfg.news_share

    details: list[str] = []
    missing: list[str] = []
    fg_part: float | None = None
    if g is None:
        missing.append("Fear & Greed")
        details.append("Sin Fear & Greed: esa parte cuenta 0.")
    else:
        g = min(FEAR_GREED_MAX, max(FEAR_GREED_MIN, g))
        fg_part = fear_greed_part(g, cfg)
        if cfg.fear_greed_mode == "contrarian":
            mode = "modo contrarian: el miedo suma al long y la codicia resta"
        else:
            mode = "modo tendencia: la codicia suma al long y el miedo resta"
        details.append(
            f"Fear & Greed en {fmt_num(g, 0)} ({fear_greed_label(g, cfg)}), {mode} "
            f"{_contrib(fg_part)}."
        )

    if share <= 0:
        details.append("Las noticias no mueven el score: las comenta la IA.")
    elif news is None:
        details.append("Sin balance de noticias: no se cuentan, se usa solo Fear & Greed.")
    else:
        details.append(
            f"Balance de noticias {fmt_num(news, 2, signed=True)} "
            f"(pesa {fmt_num(share * 100.0, 0)} % del sentimiento)."
        )

    if news is not None and share > 0:
        fg_value = 0.0 if fg_part is None else fg_part
        value = (1.0 - share) * fg_value + share * news
    elif fg_part is not None:
        value = fg_part
    else:
        return ComponentReading(0.0, False, tuple(details), tuple(missing))
    return ComponentReading(_clip(value), True, tuple(details), tuple(missing))


__all__ = [
    "FEAR_GREED_NEUTRAL",
    "FUNDING_REFERENCE_HOURS",
    "TREND_VALUES",
    "ComponentReading",
    "clean_number",
    "derivatives_component",
    "fear_greed_label",
    "fear_greed_part",
    "fmt_num",
    "funding_part",
    "funding_pct_per_8h",
    "long_short_part",
    "macd_score",
    "momentum_component",
    "open_interest_part",
    "rsi_score",
    "sentiment_component",
    "trend_component",
    "trend_value",
]
