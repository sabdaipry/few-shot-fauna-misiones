"""
Persistencia de sesión entre ejecuciones de SAREKO.

SessionManager.save(events, batch_summary) — serializa a JSON en un hilo background
SessionManager.load()                      — devuelve (events, batch_summary) o None
SessionManager.clear()                     — borra el archivo de sesión
"""

import dataclasses
import json
import types
from datetime import datetime, timezone
from pathlib import Path

from PySide6.QtCore import QThread

_SESSION_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "last_session.json"


class _SaveWorker(QThread):
    """Escribe una cadena JSON a disco sin bloquear el hilo principal."""

    def __init__(self, path: Path, content: str) -> None:
        super().__init__()
        self._path    = path
        self._content = content

    def run(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(self._content, encoding="utf-8")
        except Exception:
            pass


def _event_to_dict(event) -> dict:
    if dataclasses.is_dataclass(event) and not isinstance(event, type):
        return dataclasses.asdict(event)
    if isinstance(event, types.SimpleNamespace):
        return vars(event)
    return {}


def _event_from_dict(d: dict) -> types.SimpleNamespace:
    return types.SimpleNamespace(**d)


class SessionManager:
    """Gestiona la sesión persistente en 04-app/data/last_session.json."""

    _worker: "_SaveWorker | None" = None

    @classmethod
    def save(cls, events: list[dict], batch_summary: dict) -> None:
        """Serializa la sesión completa a JSON y la escribe en background.

        events       — lista de records de ValidacionTab
        batch_summary — resumen del lote de AnalisisTab
        """
        payload = {
            "version":       1,
            "saved_at":      datetime.now(timezone.utc).isoformat(),
            "events":        [
                {
                    "filename":   r["filename"],
                    "filepath":   str(r["filepath"]) if r.get("filepath") else None,
                    "event":      _event_to_dict(r["event"]),
                    "validation": r.get("validation", {
                        "state": "pending", "category": None, "custom_species": None,
                    }),
                }
                for r in events
            ],
            "batch_summary": batch_summary,
        }
        try:
            content = json.dumps(payload, ensure_ascii=False, default=str)
        except Exception:
            return

        # Si hay una escritura anterior en curso, esperar hasta 500 ms
        if cls._worker and cls._worker.isRunning():
            cls._worker.wait(500)

        cls._worker = _SaveWorker(_SESSION_PATH, content)
        cls._worker.start()

    @classmethod
    def load(cls) -> "tuple[list[dict], dict] | None":
        """Carga la sesión guardada. Devuelve (events, batch_summary) o None."""
        if not _SESSION_PATH.exists():
            return None
        try:
            data = json.loads(_SESSION_PATH.read_text(encoding="utf-8"))
            if data.get("version") != 1:
                return None

            events: list[dict] = []
            for r in data.get("events", []):
                fp_str = r.get("filepath")
                events.append({
                    "filename":   r["filename"],
                    "filepath":   Path(fp_str) if fp_str else None,
                    "event":      _event_from_dict(r.get("event", {})),
                    "validation": r.get("validation", {
                        "state": "pending", "category": None, "custom_species": None,
                    }),
                })

            batch_summary = data.get("batch_summary", {})
            if not events and not batch_summary.get("files"):
                return None
            return events, batch_summary

        except Exception:
            return None

    @classmethod
    def clear(cls) -> None:
        """Elimina el archivo de sesión guardado."""
        try:
            _SESSION_PATH.unlink(missing_ok=True)
        except Exception:
            pass
