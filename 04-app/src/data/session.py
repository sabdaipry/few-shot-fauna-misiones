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

_SESSION_PATH  = Path(__file__).resolve().parent.parent.parent / "data" / "last_session.json"
_HISTORY_PATH  = Path(__file__).resolve().parent.parent.parent / "data" / "history.json"


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
    """Gestiona la sesión persistente y el historial acumulado en 04-app/data/."""

    _worker:         "_SaveWorker | None" = None
    _history_worker: "_SaveWorker | None" = None

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
                    "filename":     r["filename"],
                    "filepath":     str(r["filepath"]) if r.get("filepath") else None,
                    "event":        _event_to_dict(r["event"]),
                    "validation":   r.get("validation", {
                        "state": "pending", "category": None, "custom_species": None,
                    }),
                    "extra_species": r.get("extra_species", []),
                    "multi_species": r.get("multi_species", False),
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
                    "filename":     r["filename"],
                    "filepath":     Path(fp_str) if fp_str else None,
                    "event":        _event_from_dict(r.get("event", {})),
                    "validation":   r.get("validation", {
                        "state": "pending", "category": None, "custom_species": None,
                    }),
                    "extra_species": r.get("extra_species", []),
                    "multi_species": r.get("multi_species", False),
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

    @classmethod
    def clear_history(cls) -> None:
        """Elimina el archivo de historial acumulado."""
        try:
            _HISTORY_PATH.unlink(missing_ok=True)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Historial acumulado (history.json)

    @classmethod
    def load_history(cls) -> dict:
        """Carga history.json. Devuelve dict vacío si no existe o está corrupto."""
        if not _HISTORY_PATH.exists():
            return {}
        try:
            data = json.loads(_HISTORY_PATH.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return {}
            return data
        except Exception:
            return {}

    @classmethod
    def save_history(cls, history: dict) -> None:
        """Escribe history.json en background."""
        try:
            content = json.dumps(history, ensure_ascii=False, default=str)
        except Exception:
            return
        if cls._history_worker and cls._history_worker.isRunning():
            cls._history_worker.wait(500)
        cls._history_worker = _SaveWorker(_HISTORY_PATH, content)
        cls._history_worker.start()

    @classmethod
    def append_history(cls, batch_summary: dict, records: list[dict]) -> dict:
        """Acumula los datos de una corrida completada en history.json.

        Devuelve el historial actualizado (ya incluye la corrida actual)
        para que el llamador pueda usarlo sin esperar la escritura a disco.
        """
        history = cls.load_history()

        history.setdefault("total_runs",        0)
        history.setdefault("total_frames",       0)
        history.setdefault("total_records",      0)
        history.setdefault("total_validations",  0)
        history.setdefault("runs",               [])
        history.setdefault("species_counts",     {})
        history.setdefault("confidence_counts",  {"alta": 0, "baja": 0, "ambiguo": 0})
        history.setdefault("latency_records",    [])

        history["total_runs"]       += 1
        history["total_frames"]     += batch_summary.get("total_frames", 0)
        history["total_records"]    += len(records)
        history["total_validations"] += sum(
            1 for r in records
            if r.get("validation", {}).get("state") == "validated"
        )

        files     = batch_summary.get("files", [])
        has_error = any(f.get("state") == "error" for f in files)
        history["runs"].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode":      batch_summary.get("mode", "—"),
            "files":     [f.get("name", "?") for f in files],
            "state":     "error" if has_error else "completado",
        })

        sp_counts   = history["species_counts"]
        conf_counts = history["confidence_counts"]
        for r in records:
            event = r.get("event")
            sp    = getattr(event, "species", "") if event is not None else ""
            if sp:
                sp_counts[sp] = sp_counts.get(sp, 0) + 1
            if event is None:
                continue
            if getattr(event, "ambiguous", False):
                conf_counts["ambiguo"] = conf_counts.get("ambiguo", 0) + 1
            elif getattr(event, "confidence_level", "") == "alta":
                conf_counts["alta"] = conf_counts.get("alta", 0) + 1
            else:
                conf_counts["baja"] = conf_counts.get("baja", 0) + 1

        # Acumular latency records por cada archivo completado de esta corrida
        mode           = batch_summary.get("mode", "—")
        consensus_mode = batch_summary.get("consensus_mode", "static")
        for f in files:
            if f.get("state") != "completado":
                continue
            proc_sec = f.get("processing_sec")
            if proc_sec is None:
                continue
            dur_sec = f.get("duration_sec")
            history["latency_records"].append({
                "filename":         f.get("name", ""),
                "type":             "video" if dur_sec is not None else "imagen",
                "duration_sec":     dur_sec,
                "mode":             mode,
                "consensus_mode":   consensus_mode,
                "frames_processed": f.get("frames_processed"),
                "processing_sec":   float(proc_sec),
            })

        cls.save_history(history)
        return history
