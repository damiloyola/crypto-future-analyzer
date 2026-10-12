"""Guardia permanente contra el look-ahead (regla de CLAUDE.md).

Revisa el código fuente de todos los módulos de ``backend/app/`` (y
``backend/scripts/``; los tests no, porque usan ``predict_proba`` como oráculo)
y falla si alguno llama a métodos de hmmlearn que miran el futuro:

- ``predict``, ``predict_proba``, ``score_samples`` (forward-backward) y
  ``decode`` (Viterbi o MAP sobre toda la serie);
- sus versiones privadas (``_decode_viterbi``, ``_decode_map``,
  ``_compute_posteriors_*``, ``_do_estep``, ``_score*`` con posteriores) y el
  núcleo en C ``hmmlearn._hmmc`` (``backward_*``, ``viterbi``, ``*_xi_sum``);
- también ``getattr(obj, "predict")`` y similares.

El régimen de cada vela sale únicamente de ``app.regime.forward`` (algoritmo
forward). ``hmm.score``, ``hmm.bic``/``aic`` y ``hmm.fit`` están permitidos: el
primero es solo el forward, los otros son entrenamiento y selección de modelo,
no clasifican velas.

Se analiza el árbol sintáctico (``ast``), no el texto: los docstrings y
comentarios pueden nombrar estos métodos sin problema. ``bytes.decode()`` o
``.decode("utf-8")`` no cuentan (no tienen argumentos o el argumento es un
texto); ``decode(X=...)`` sí cuenta. Si alguna vez hiciera falta un caso
legítimo (por ejemplo un modelo de scikit-learn que no es el HMM), se marca la
línea con ``# lookahead-ok: <motivo>``; sin motivo la marca no vale.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
SCAN_DIRS = (BACKEND / "app", BACKEND / "scripts")

# Atributos prohibidos siempre (accedidos o llamados).
FORBIDDEN_ATTRS: dict[str, str] = {
    "predict": "Viterbi sobre toda la serie",
    "predict_proba": "forward-backward (probabilidades suavizadas)",
    "predict_log_proba": "probabilidades suavizadas",
    "score_samples": "forward-backward (probabilidades suavizadas)",
    "_decode_viterbi": "Viterbi",
    "_decode_map": "forward-backward",
    "_compute_posteriors_log": "forward-backward",
    "_compute_posteriors_scaling": "forward-backward",
    "_do_estep": "forward-backward",
    "_score_log": "puede calcular posteriores suavizadas",
    "_score_scaling": "puede calcular posteriores suavizadas",
    "backward_log": "algoritmo backward",
    "backward_scaling": "algoritmo backward",
    "viterbi": "Viterbi",
    "compute_log_xi_sum": "forward-backward",
    "compute_scaling_xi_sum": "forward-backward",
}
# ``decode`` choca con ``bytes.decode``: solo se marca si parece la del HMM
# (algún argumento que no es un texto, o los nombres de argumento del HMM).
_DECODE_HMM_KWARGS = {"X", "lengths", "algorithm"}
_FORBIDDEN_MODULES = ("hmmlearn._hmmc",)
_MARKER = re.compile(r"#\s*lookahead-ok:\s*\S")


def _is_str_constant(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _looks_like_hmm_decode(call: ast.Call) -> bool:
    if any(kw.arg in _DECODE_HMM_KWARGS for kw in call.keywords):
        return True
    return any(not _is_str_constant(arg) for arg in call.args)


def find_lookahead_calls(source: str, filename: str = "<src>") -> list[str]:
    """Usos prohibidos en ``source``, como ``"archivo:línea: explicación"``."""
    tree = ast.parse(source, filename=filename)
    lines = source.splitlines()
    found: list[tuple[ast.AST, str]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            found.append((node, f".{node.attr} ({FORBIDDEN_ATTRS[node.attr]})"))
        elif isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "decode"
                and _looks_like_hmm_decode(node)
            ):
                found.append((node, ".decode(...) (Viterbi/MAP sobre toda la serie)"))
            elif (
                isinstance(func, ast.Attribute)
                and func.attr == "_score"
                and any(
                    kw.arg == "compute_posteriors"
                    and not (isinstance(kw.value, ast.Constant) and kw.value.value is False)
                    for kw in node.keywords
                )
            ):
                found.append((node, "._score(compute_posteriors=...) (posteriores suavizadas)"))
            elif (
                isinstance(func, ast.Name)
                and func.id == "getattr"
                and len(node.args) >= 2
                and _is_str_constant(node.args[1])
                and (node.args[1].value in FORBIDDEN_ATTRS or node.args[1].value == "decode")
            ):
                found.append((node, f'getattr(..., "{node.args[1].value}")'))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(_FORBIDDEN_MODULES):
                    found.append((node, f"import {alias.name} (núcleo en C de hmmlearn)"))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = [a.name for a in node.names]
            if module.startswith(_FORBIDDEN_MODULES) or (module == "hmmlearn" and "_hmmc" in names):
                found.append((node, f"from {module} import ... (núcleo en C de hmmlearn)"))
            for name in names:
                if name in FORBIDDEN_ATTRS:
                    found.append((node, f"from {module} import {name}"))

    out = []
    for node, why in found:
        start = node.lineno
        end = getattr(node, "end_lineno", None) or start
        if any(_MARKER.search(lines[i - 1]) for i in range(start, end + 1)):
            continue
        out.append(f"{filename}:{start}: {why}")
    return sorted(set(out))


def _source_files() -> list[Path]:
    files: list[Path] = []
    for d in SCAN_DIRS:
        if d.is_dir():
            files.extend(p for p in d.rglob("*.py") if "__pycache__" not in p.parts)
    return sorted(files)


# --- la guardia ----------------------------------------------------------------------


def test_app_source_never_uses_lookahead_hmm_methods():
    violations: list[str] = []
    for path in _source_files():
        rel = path.relative_to(BACKEND).as_posix()
        violations.extend(find_lookahead_calls(path.read_text(encoding="utf-8"), rel))
    assert not violations, (
        "Estas líneas usan métodos de hmmlearn que miran el futuro (look-ahead). "
        "El régimen por vela tiene que salir de app.regime.forward:\n" + "\n".join(violations)
    )


def test_scan_covers_the_whole_app():
    rels = {p.relative_to(BACKEND).as_posix() for p in _source_files()}
    for must in (
        "app/regime/forward.py",
        "app/regime/model.py",
        "app/regime/service.py",
        "app/regime/stability.py",
    ):
        assert must in rels
    assert len(rels) >= 20
    assert not any(r.startswith("tests/") for r in rels)


# --- el detector funciona --------------------------------------------------------------


@pytest.mark.parametrize(
    "snippet",
    [
        "states = hmm.predict(Z)",
        "p = model.hmm.predict_proba(Z)",
        "p = self.hmm.predict_proba(Z, lengths=[10])",
        "s = hmm.score_samples(Z)",
        "ll, path = hmm.decode(Z)",
        "ll, path = hmm.decode(Z, algorithm='viterbi')",
        "ll, path = hmm.decode(X=Z)",
        "f = hmm.predict\nf(Z)",
        "f = getattr(hmm, 'predict_proba')",
        "f = getattr(hmm, 'decode')",
        "path = hmm._decode_viterbi(Z)",
        "ll, post = hmm._score(Z, compute_posteriors=True)",
        "from hmmlearn._hmmc import backward_log",
        "from hmmlearn import _hmmc",
        "import hmmlearn._hmmc as core",
        "x = core.viterbi(a, b, c)",
        "p = hmm.predict(Z)  # lookahead-ok",  # marca sin motivo: no vale
        "p = hmm.predict_proba(\n    Z\n)",
    ],
)
def test_detector_catches_forbidden_patterns(snippet):
    assert find_lookahead_calls(snippet), snippet


@pytest.mark.parametrize(
    "snippet",
    [
        '"""Nunca usar hmm.predict(Z) ni hmm.decode(Z): miran el futuro."""',
        "# hmm.predict_proba(Z) está prohibido",
        "text = raw.decode()",
        "text = raw.decode('utf-8')",
        "text = raw.decode(encoding='latin-1', errors='ignore')",
        "ll = hmm.score(Z)",
        "b = hmm.bic(Z)",
        "hmm.fit(Z)",
        "ll = hmm._score(Z, compute_posteriors=False)",
        "res = filtered_probabilities(Z, a, b, c, d)",
        "p = clf.predict(X)  # lookahead-ok: regresión de sklearn, no es el HMM",
    ],
)
def test_detector_ignores_safe_code(snippet):
    assert find_lookahead_calls(snippet) == [], snippet
