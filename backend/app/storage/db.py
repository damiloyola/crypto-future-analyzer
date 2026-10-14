"""Conexión a SQLite.

- ``default_db_path(settings)``: dónde vive la base (``<app.data_dir>/analyzer.db``,
  relativo a la raíz del repo).
- ``make_engine(path)``: motor de SQLAlchemy para un archivo SQLite. Crea las
  carpetas que falten y permite usar la conexión desde varios hilos (FastAPI
  atiende pedidos en un pool de hilos). ``":memory:"`` = base en memoria.
- ``init_db(engine)``: crea las tablas que falten y controla la versión del
  esquema (``PRAGMA user_version``). Si la base fue creada por una versión
  distinta de la app, frena con un mensaje claro en vez de leer columnas que
  no existen.
- ``open_db(path)``: ``make_engine`` + ``init_db`` en un paso.
- ``session_scope(engine)``: sesión que confirma al salir bien y deshace todo
  si hay un error.

``create_all`` no modifica tablas que ya existen: si en un hito futuro cambia
una tabla, hay que subir ``SCHEMA_VERSION`` y escribir la migración.

Historial del esquema (``PRAGMA user_version``)
    1. Hito 5, tanda 1: ``manual_trades``, ``open_positions``,
       ``wallet_events`` y ``analyses``.
    2. Hito 5, tanda 2: agrega ``high_volatility_events`` (evento de alta
       volatilidad con nota, inicio y hora de fin). El cambio es **aditivo**: no
       toca ninguna tabla existente. Una base de la versión 1 se migra sola al
       abrirla (``MIGRATIONS[1]`` crea la tabla nueva y marca la versión 2), sin
       perder datos.
    3. Hito 5, segunda ronda: agrega ``safety_locks`` (cada bloqueo total que
       escribió la app, para medir de nuevo la caída del mes después de que el
       usuario borra el LOCKED). También aditivo: ``MIGRATIONS[2]`` crea la
       tabla. Una base de la versión 1 pasa por 1 → 2 → 3.
    4. Hito 5, tercera ronda: agrega la columna ``initial_risk_usd`` (el
       riesgo previsto al abrir) a ``open_positions`` y ``manual_trades``, así
       mover el stop con ``update_position`` no pisa el riesgo previsto. Aditivo:
       ``MIGRATIONS[3]`` agrega la columna (vacía en las filas viejas, que
       siguen usando ``risk_usd`` como previsto). 1 → 2 → 3 → 4.

Una base de una versión **más nueva** que la app (por ejemplo, abierta con un
checkout viejo) se rechaza: la app vieja no sabe qué tablas o columnas espera.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from app.config import AppSettings
from app.storage import models as _models  # noqa: F401  (registra las tablas)

#: Nombre del archivo de la base dentro de ``app.data_dir``.
DEFAULT_DB_FILENAME = "analyzer.db"

#: Ruta especial para una base en memoria (pruebas).
IN_MEMORY = ":memory:"

#: Versión del esquema. Se guarda en ``PRAGMA user_version`` al crear la base.
#: Ver el historial en el docstring del módulo.
SCHEMA_VERSION = 4


def default_db_path(settings: AppSettings) -> Path:
    """Ruta absoluta de la base: ``<app.data_dir>/analyzer.db``.

    ``app.data_dir`` es relativo a la raíz del repo (``settings.resolve_path``).
    """
    return settings.resolve_path(settings.app.data_dir) / DEFAULT_DB_FILENAME


def _set_sqlite_pragmas(dbapi_connection: Any, _connection_record: Any) -> None:
    # Espera hasta 5 s si otro proceso (la consola y el servidor a la vez)
    # tiene la base tomada, en vez de fallar enseguida con "database is locked".
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA busy_timeout = 5000")
    finally:
        cursor.close()


def make_engine(path: Path | str, *, echo: bool = False) -> Engine:
    """Motor de SQLAlchemy para el archivo SQLite ``path``.

    Crea las carpetas padre si no existen. ``check_same_thread=False`` porque
    la API usa la misma conexión desde distintos hilos (cada operación del
    repositorio abre y cierra su propia sesión).

    ``":memory:"`` da una base en memoria (una sola conexión compartida, se
    pierde al cerrar el proceso): sirve para pruebas rápidas.
    """
    if str(path) == IN_MEMORY:
        engine = create_engine(
            "sqlite://",
            echo=echo,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    else:
        db_path = Path(path).expanduser().resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(
            f"sqlite:///{db_path}",
            echo=echo,
            connect_args={"check_same_thread": False},
        )
    event.listen(engine, "connect", _set_sqlite_pragmas)
    return engine


def _user_version(engine: Engine) -> int:
    with engine.connect() as conn:
        return int(conn.exec_driver_sql("PRAGMA user_version").scalar() or 0)


def _table_names(engine: Engine) -> set[str]:
    with engine.connect() as conn:
        rows = conn.exec_driver_sql(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ).all()
    return {str(r[0]) for r in rows}


def _migrate_1_to_2(engine: Engine) -> None:
    """Versión 1 → 2: crea ``high_volatility_events`` (aditivo, no toca datos)."""
    table = SQLModel.metadata.tables[_models.HighVolatilityEvent.__tablename__]
    table.create(engine, checkfirst=True)


def _migrate_2_to_3(engine: Engine) -> None:
    """Versión 2 → 3: crea ``safety_locks`` (aditivo, no toca datos)."""
    table = SQLModel.metadata.tables[_models.SafetyLock.__tablename__]
    table.create(engine, checkfirst=True)


def _columns(engine: Engine, table: str) -> set[str]:
    with engine.connect() as conn:
        rows = conn.exec_driver_sql(f'PRAGMA table_info("{table}")').all()
    return {str(r[1]) for r in rows}


def _migrate_3_to_4(engine: Engine) -> None:
    """Versión 3 → 4: agrega ``initial_risk_usd`` a posiciones y operaciones (aditivo).

    Las filas que ya estaban quedan con la columna vacía (``NULL``): para
    ellas el riesgo previsto es el ``risk_usd`` guardado, como antes. Si la
    columna ya existe (una migración cortada a la mitad), no hace nada.
    """
    for model in (_models.OpenPosition, _models.ManualTrade):
        table = model.__tablename__
        if "initial_risk_usd" not in _columns(engine, table):
            with engine.begin() as conn:
                conn.exec_driver_sql(f'ALTER TABLE "{table}" ADD COLUMN initial_risk_usd FLOAT')


#: Migraciones: ``MIGRATIONS[n]`` lleva una base de la versión ``n`` a la ``n + 1``.
MIGRATIONS: dict[int, Callable[[Engine], None]] = {
    1: _migrate_1_to_2,
    2: _migrate_2_to_3,
    3: _migrate_3_to_4,
}


def _set_user_version(engine: Engine, version: int) -> None:
    with engine.begin() as conn:
        conn.exec_driver_sql(f"PRAGMA user_version = {int(version)}")


def init_db(engine: Engine) -> None:
    """Crea las tablas que falten, migra versiones viejas y marca la versión del esquema.

    - Base nueva (versión 0): crea todo y guarda ``SCHEMA_VERSION``. Si tiene
      solo tablas de la app (una creación que se cortó a la mitad), la completa.
    - Base de una versión anterior (de la 1 a la 3): aplica las migraciones
      de ``MIGRATIONS`` en orden (todas aditivas) y marca ``SCHEMA_VERSION``.
    - Base con la misma versión: crea solo las tablas que falten (no toca datos).
    - Base de una versión más nueva que la app, o un archivo SQLite con tablas
      ajenas a la app: ``RuntimeError`` explicando qué hacer, para no leer ni
      escribir con un esquema que no coincide.
    """
    version = _user_version(engine)
    if version < 0 or version > SCHEMA_VERSION:
        raise RuntimeError(
            f"La base de datos {engine.url.database} es de otra versión de la app "
            f"(esquema {version}, se esperaba {SCHEMA_VERSION}). Hacé una copia del "
            "archivo y migrala, o borrala si no tiene datos que te importen."
        )
    if version == 0:
        foreign = _table_names(engine) - set(SQLModel.metadata.tables)
        if foreign:
            raise RuntimeError(
                f"El archivo {engine.url.database} no parece ser la base de la app: "
                f"tiene otras tablas ({', '.join(sorted(foreign))}). Usá otro archivo."
            )
        SQLModel.metadata.create_all(engine)
        _set_user_version(engine, SCHEMA_VERSION)
        return
    for step in range(version, SCHEMA_VERSION):
        MIGRATIONS[step](engine)
        _set_user_version(engine, step + 1)
    SQLModel.metadata.create_all(engine)


def open_db(path: Path | str) -> Engine:
    """``make_engine(path)`` + ``init_db``: la base lista para usar."""
    engine = make_engine(path)
    init_db(engine)
    return engine


@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    """Sesión transaccional: ``commit`` si el bloque termina bien, ``rollback`` si falla.

    ``expire_on_commit=False``: los objetos leídos siguen teniendo sus valores
    después del ``commit`` (no vuelven a consultar la base).
    """
    session = Session(engine, expire_on_commit=False)
    try:
        yield session
        session.commit()
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()


__all__ = [
    "DEFAULT_DB_FILENAME",
    "IN_MEMORY",
    "MIGRATIONS",
    "SCHEMA_VERSION",
    "default_db_path",
    "init_db",
    "make_engine",
    "open_db",
    "session_scope",
]
