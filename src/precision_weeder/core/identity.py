"""Identidad de contenido de los artefactos versionados (A1, orden A1/E2 §3.1).

Mapas, perfiles, caminos, solicitudes y resultados llevan un SHA-256 de su
contenido canonico. Un cambio de geometria, mapa o tolerancia cambia el hash e
invalida cualquier informe ligado al hash anterior.

El JSON canonico ordena claves, no admite NaN/inf y serializa los reales con
`repr` (ida y vuelta exacta en `float64`), de modo que el hash es estable entre
ejecuciones y plataformas con la misma aritmetica IEEE-754.

Propiedad de datos (A1E2-R1): `freeze` convierte un contenido JSON en estructuras
inmutables propias (`MappingProxyType` sobre un diccionario nuevo y tuplas), y
`thaw` produce una copia profunda independiente en `dict`/`list`. Un certificado
terminado guarda la version congelada y exporta siempre una copia: modificar una
exportacion o la entrada original no altera el artefacto construido.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from .errors import SchemaValidationError

__all__ = ["canonical_json", "content_hash", "freeze", "is_sha256", "thaw"]


def freeze(value: Any) -> Any:
    """Copia profunda inmutable de un contenido JSON (dict, list/tuple, escalares).

    Raises:
        SchemaValidationError: si contiene tipos no JSON o reales no finitos.
    """
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SchemaValidationError("contenido congelado: real no finito")
        return value
    raise SchemaValidationError(f"contenido congelado: tipo no JSON {type(value).__name__}")


def thaw(value: Any) -> Any:
    """Copia profunda independiente en tipos JSON nativos (`dict`, `list`)."""
    if isinstance(value, Mapping):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [thaw(item) for item in value]
    return value


def is_sha256(value: Any) -> bool:
    """`True` si `value` es un SHA-256 hexadecimal en minusculas."""
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def canonical_json(payload: Any) -> str:
    """JSON canonico: claves ordenadas, sin espacios y sin valores no finitos.

    Raises:
        SchemaValidationError: si el contenido no es serializable o contiene NaN/inf.
    """
    try:
        return json.dumps(
            thaw(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise SchemaValidationError(f"contenido no serializable de forma canonica: {exc}") from exc


def content_hash(payload: Any) -> str:
    """SHA-256 hexadecimal del JSON canonico de `payload`."""
    return hashlib.sha256(canonical_json(payload).encode("ascii")).hexdigest()
