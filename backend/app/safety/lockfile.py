"""Archivo de lock.

Se implementa en el Hito 5.

- ``write_lock(path, report_path)``: crea ``data/LOCKED``.
- ``is_locked(path)``: mientras exista, la app no genera planes.
- No hay función para borrarlo: es deliberado. El usuario lo borra a mano
  después de revisar el informe del incidente.
"""
