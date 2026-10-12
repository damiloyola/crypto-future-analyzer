"""Filtro de estabilidad del régimen.

Funciona con cualquier etiqueta comparable con ``==`` (índices de estado o
categorías como ``"bajista"``); el servicio decide cuál usar según
``regime.change_level``. Todas las funciones son causales: lo que dicen de la
vela *t* depende solo de las velas 0..t, así que agregar velas nuevas nunca
cambia lo ya calculado. Los valores ``None`` en la serie cruda significan
"sin dato": cortan las rachas, no se confirman y no cuentan como cambio.

Los umbrales salen de ``settings.regime`` (``confirmation_bars``,
``instability_window``, ``instability_max_changes``); el servicio los pasa.
Los valores por omisión de ``bars`` y ``window`` son los mismos de
``config.yaml`` y están solo por comodidad.

- ``confirmed_series(raw, bars)``: régimen vigente en cada vela. Al principio
  no hay ninguno (``None``) hasta que un valor se mantiene ``bars`` velas
  seguidas; después, un valor nuevo reemplaza al confirmado recién cuando se
  mantuvo ``bars`` velas seguidas (en esa vela pasa a estar confirmado).
  Mientras tanto sigue vigente el anterior.
- ``changes_in_window(raw, window)``: cambios entre velas consecutivas dentro
  de las últimas ``window`` velas (son ``window - 1`` comparaciones; el paso
  de la vela anterior a la ventana a la primera de la ventana no cuenta).
- ``unstable_series(raw, window, max_changes)`` / ``is_unstable(...)``: "régimen
  inestable" cuando hubo **más de** ``max_changes`` cambios en la ventana.
- ``regime_age(confirmed)``: cuántas velas lleva vigente el régimen confirmado
  actual, contando la vela actual (0 si todavía no hay ninguno). Cuenta desde
  que quedó confirmado; el valor crudo había arrancado ``bars - 1`` velas antes.
- ``pending(raw, confirmed)``: si la última vela cruda difiere del régimen
  confirmado, el candidato y cuántas velas seguidas lleva.
- ``summarize_stability(raw, bars, window, max_changes)``: todo lo anterior
  para la última vela, en un ``StabilitySummary``.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


def _plain(value: Any) -> Any:
    """Convierte escalares de numpy a tipos de Python (para que se serialicen a JSON)."""
    return value.item() if isinstance(value, np.generic) else value


def _check_positive(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} tiene que ser >= 1 (llegó {value})")


def confirmed_series(raw: Sequence[Hashable | None], bars: int = 3) -> list[Any]:
    """Régimen confirmado (vigente) en cada vela; ``None`` mientras no haya ninguno.

    Un valor queda confirmado en la vela en que completa ``bars`` velas
    seguidas. Con ``bars=1`` el confirmado es igual al crudo.
    """
    _check_positive("bars", bars)
    out: list[Any] = []
    confirmed: Any = None
    run_value: Any = None
    run_len = 0
    for value in raw:
        value = _plain(value)
        if value is None:
            run_value, run_len = None, 0
        elif run_len and value == run_value:
            run_len += 1
        else:
            run_value, run_len = value, 1
        if run_value is not None and run_len >= bars:
            confirmed = run_value
        out.append(confirmed)
    return out


def _change_flags(raw: Sequence[Hashable | None]) -> np.ndarray:
    """``flags[t] = 1`` si la vela *t* cambió respecto de la *t-1* (ambas con dato)."""
    values = [_plain(v) for v in raw]
    flags = np.zeros(len(values), dtype=np.int64)
    for t in range(1, len(values)):
        prev, cur = values[t - 1], values[t]
        if prev is not None and cur is not None and cur != prev:
            flags[t] = 1
    return flags


def changes_in_window(raw: Sequence[Hashable | None], window: int = 20) -> int:
    """Cantidad de cambios de régimen entre velas consecutivas de las últimas ``window`` velas.

    Son ``window - 1`` comparaciones. Con menos velas que ``window`` se usan
    todas las que hay.
    """
    _check_positive("window", window)
    tail = list(raw)[-window:]
    return int(_change_flags(tail).sum())


def unstable_series(raw: Sequence[Hashable | None], window: int, max_changes: int) -> list[bool]:
    """Para cada vela: ``True`` si en la ventana que termina en ella hubo más de ``max_changes``."""
    _check_positive("window", window)
    flags = _change_flags(raw)
    n = len(flags)
    if n == 0:
        return []
    csum = np.concatenate(([0], np.cumsum(flags)))
    t = np.arange(n)
    # La ventana de la vela t va de t-window+1 a t; los cambios que caen adentro
    # son los de las velas t-window+2 .. t (el de la primera vela mira afuera).
    first = np.maximum(t - window + 2, 0)
    counts = csum[t + 1] - csum[first]
    return [bool(c > max_changes) for c in counts]


def is_unstable(raw: Sequence[Hashable | None], window: int, max_changes: int) -> bool:
    """``True`` si en las últimas ``window`` velas hubo más de ``max_changes`` cambios."""
    return changes_in_window(raw, window) > max_changes


def regime_age(confirmed: Sequence[Hashable | None]) -> int:
    """Velas que lleva vigente el régimen confirmado actual (incluye la actual; 0 si no hay)."""
    values = [_plain(v) for v in confirmed]
    if not values or values[-1] is None:
        return 0
    return _trailing_run(values)


def _trailing_run(values: Sequence[Any]) -> int:
    """Largo de la racha final (velas seguidas iguales a la última)."""
    last = values[-1]
    run = 0
    for value in reversed(values):
        if value is None or value != last:
            break
        run += 1
    return run


def pending(
    raw: Sequence[Hashable | None], confirmed: Sequence[Hashable | None]
) -> tuple[Any, int] | None:
    """``(candidato, velas_seguidas)`` si la última vela cruda no es el régimen confirmado.

    Devuelve ``None`` si coinciden, si no hay velas o si la última vela cruda no
    tiene dato. Mientras no hay nada confirmado (arranque), cualquier valor
    crudo es candidato.
    """
    values = [_plain(v) for v in raw]
    conf = [_plain(v) for v in confirmed]
    if len(values) != len(conf):
        raise ValueError("raw y confirmed tienen que tener el mismo largo")
    if not values or values[-1] is None:
        return None
    if conf[-1] is not None and values[-1] == conf[-1]:
        return None
    return values[-1], _trailing_run(values)


@dataclass(frozen=True)
class StabilitySummary:
    """Estado del filtro de estabilidad en la última vela."""

    raw: Any  # régimen crudo de la última vela (estado más probable)
    confirmed: Any  # régimen vigente (None si todavía no se confirmó ninguno)
    is_confirmed: bool  # la última vela cruda coincide con el régimen vigente
    age_bars: int  # velas que lleva vigente el régimen confirmado
    pending: tuple[Any, int] | None  # (candidato, velas seguidas) si hay uno en espera
    changes_in_window: int  # cambios en las últimas `window` velas
    unstable: bool  # más de `max_changes` cambios en la ventana
    window: int
    max_changes: int


def summarize_stability(
    raw: Sequence[Hashable | None], *, bars: int, window: int, max_changes: int
) -> StabilitySummary:
    """Resumen del filtro de estabilidad para la última vela de ``raw``."""
    conf = confirmed_series(raw, bars)
    values = [_plain(v) for v in raw]
    last_raw = values[-1] if values else None
    last_conf = conf[-1] if conf else None
    return StabilitySummary(
        raw=last_raw,
        confirmed=last_conf,
        is_confirmed=last_conf is not None and last_raw == last_conf,
        age_bars=regime_age(conf),
        pending=pending(values, conf),
        changes_in_window=changes_in_window(values, window),
        unstable=is_unstable(values, window, max_changes),
        window=window,
        max_changes=max_changes,
    )
