"""
Pestaña Validación de SAREKO.

Estructura:
    ValidacionTab
      ├── _HeaderCard          — breadcrumb + título grande
      ├── _SearchCard          — filtro por especie / archivo
      └── _ContentRow (QHBoxLayout)
            ├── _RegistrosSection — tabla paginada de eventos
            └── _SidePanel        — panel deslizable de detalle
"""

import csv
import os
import pickle
from pathlib import Path
from typing import Optional

from PySide6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    Qt,
    QSize,
    QTimer,
    Signal,
)
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QCompleter,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    import matplotlib.cm as _mpl_cm
    import numpy as _np_mpl
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

from ..styles import (
    ACCENT,
    NEUTRAL,
    SUCCESS,
    TEXT_PRIMARY,
    badge_qss,
    badge_qss_for,
    body_qss,
    card_qss,
    icon_eye_btn,
    icon_trash_btn,
    magnifier_icon,
    section_label_qss,
    title_qss,
    validation_badge_qss,
)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

_APP_DIR    = Path(__file__).resolve().parent.parent.parent.parent
_UMAP_CACHE = _APP_DIR / "data" / "umap_2d_catalog.pkl"
_CLIPS_DIR  = _APP_DIR / "SAREKO_clips"

PAGE_SIZE    = 20
PANEL_WIDTH  = 420

_SPECIAL_OPTS = ["Desconocida", "Vacío / Ruido", "Ingreso manual..."]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_completer_data(species_catalog: list) -> tuple[list[str], dict[str, str]]:
    """
    Construye (display_strings, display_to_species) para el QCompleter.

    display_strings: "nombre_comun_es_ar (Nombre científico)" o solo el nombre
    científico cuando no hay nombre común.
    display_to_species: mapeo display → nombre científico, para rellenar
    el QLineEdit con la especie correcta al seleccionar una sugerencia.
    """
    display_strings: list[str] = []
    display_to_species: dict[str, str] = {}
    seen: set[str] = set()
    for entry in species_catalog:
        sp = entry.get("species", "")
        if not sp or sp in seen:
            continue
        seen.add(sp)
        common = entry.get("nombre_comun_es_ar", "")
        if common and str(common).lower() not in ("", "nan", "none"):
            display = f"{common} ({sp})"
        else:
            display = sp
        display_strings.append(display)
        display_to_species[display] = sp
    return display_strings, display_to_species


def _completer_popup_qss() -> str:
    return """
        QAbstractItemView {
            background: #1a1a1a;
            color: #edefec;
            selection-background-color: #1f2c1d;
            border: 1px solid #99e17a;
            font-size: 12px;
            padding: 2px;
        }
    """


def _make_completer(species_catalog: list, parent) -> "tuple[QCompleter | None, dict[str, str]]":
    """Crea un QCompleter con MatchContains + CaseInsensitive para el catálogo."""
    if not species_catalog:
        return None, {}
    display_strings, display_to_species = _build_completer_data(species_catalog)
    completer = QCompleter(display_strings, parent)
    completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
    completer.setFilterMode(Qt.MatchFlag.MatchContains)
    completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
    completer.popup().setStyleSheet(_completer_popup_qss())
    return completer, display_to_species


def _fmt_time(sec: float) -> str:
    s = max(0, int(sec))
    return f"{s // 60}:{s % 60:02d}"


def _sep() -> QFrame:
    sep = QFrame()
    sep.setFrameShape(QFrame.Shape.HLine)
    sep.setStyleSheet("border: none; border-top: 1px solid rgba(153,225,122,50);")
    sep.setFixedHeight(1)
    return sep


def _get_common_name(event) -> str:
    for attr in ("nombre_comun_es_ar", "nombre_comun_en"):
        val = getattr(event, attr, "")
        if val and str(val).lower() not in ("", "nan", "none"):
            return str(val)
    return getattr(event, "species", "—")


def _get_confidence_level(event) -> str:
    if getattr(event, "ambiguous", False):
        return "ambiguo"
    return getattr(event, "confidence_level", "baja")


def _get_decisor(event) -> str:
    if getattr(event, "ambiguous", False):
        return "Consenso"
    if getattr(event, "confidence_level", "") == "alta":
        return "BioCLIP"
    return "KNN"


_METHOD_TEXTS = {
    "BioCLIP":  "Clasificación directa — distancia coseno al centroide ≤ 0.1866",
    "KNN":      "Árbitro KNN — distancia coseno al centroide > 0.1866, voto ponderado por 5 vecinos",
    "Consenso": "Evento ambiguo — quórum no alcanzado en ventana de 10 frames",
}


def _load_frame(filepath: Path, frame_idx: int) -> Optional[QPixmap]:
    """Extrae un frame de video y lo devuelve como QPixmap, o None si falla."""
    try:
        import cv2
        cap = cv2.VideoCapture(str(filepath))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(img)
    except Exception:
        return None


def _export_btn(label: str) -> QPushButton:
    btn = QPushButton(label)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setFixedHeight(28)
    btn.setStyleSheet(f"""
        QPushButton {{
            background: rgba(153,225,122,20);
            color: {TEXT_PRIMARY};
            border: 1px solid rgba(153,225,122,80);
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
            padding: 0 12px;
        }}
        QPushButton:hover {{ background: rgba(153,225,122,40); }}
        QPushButton:disabled {{
            background: rgba(74,82,72,30);
            color: rgba(237,239,236,60);
            border-color: rgba(74,82,72,60);
        }}
    """)
    return btn


def _action_btn(label: str, danger: bool = False) -> QPushButton:
    bg_hover = "rgba(224,92,92,50)" if danger else "rgba(153,225,122,40)"
    border   = "rgba(224,92,92,100)" if danger else "rgba(153,225,122,80)"
    bg       = "rgba(224,92,92,20)"  if danger else "rgba(153,225,122,20)"
    btn = QPushButton(label)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setFixedHeight(26)
    btn.setStyleSheet(f"""
        QPushButton {{
            background: {bg};
            color: {TEXT_PRIMARY};
            border: 1px solid {border};
            border-radius: 5px;
            font-size: 11px;
            font-weight: 600;
            padding: 0 10px;
        }}
        QPushButton:hover {{ background: {bg_hover}; }}
    """)
    return btn


def _combo_qss() -> str:
    return f"""
        QComboBox {{
            background: rgba(0,0,0,120);
            color: {TEXT_PRIMARY};
            border: 1px solid rgba(153,225,122,80);
            border-radius: 5px;
            padding: 2px 8px;
            font-size: 11px;
        }}
        QComboBox::drop-down {{ border: none; width: 16px; }}
        QComboBox QAbstractItemView {{
            background: #1a1a1a;
            color: {TEXT_PRIMARY};
            selection-background-color: #1f2c1d;
            border: 1px solid rgba(153,225,122,80);
            font-size: 11px;
        }}
    """


def _table_qss() -> str:
    return f"""
        QTableWidget {{
            background: transparent;
            border: none;
            outline: none;
            color: {TEXT_PRIMARY};
            font-size: 12px;
        }}
        QTableWidget::item {{
            padding: 4px 8px;
            border-bottom: 1px solid rgba(255,255,255,15);
            background: transparent;
        }}
        QTableWidget::item:hover    {{ background: rgba(153,225,122,12); }}
        QTableWidget::item:selected {{ background: rgba(31,44,29,160); border: none; }}
        QHeaderView::section {{
            background: transparent;
            color: {ACCENT};
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 1px;
            border: none;
            border-bottom: 1px solid rgba(153,225,122,89);
            padding: 4px 8px;
        }}
        QScrollBar:vertical {{
            background: rgba(255,255,255,15);
            width: 6px;
            margin: 0;
            border-radius: 3px;
        }}
        QScrollBar::handle:vertical {{
            background: rgba(153,225,122,100);
            border-radius: 3px;
            min-height: 20px;
        }}
        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {{ height: 0; }}
        QScrollBar:horizontal {{
            background: rgba(255,255,255,15);
            height: 6px;
            margin: 0;
            border-radius: 3px;
        }}
        QScrollBar::handle:horizontal {{
            background: rgba(153,225,122,100);
            border-radius: 3px;
            min-width: 20px;
        }}
        QScrollBar::add-line:horizontal,
        QScrollBar::sub-line:horizontal {{ width: 0; }}
    """


# ---------------------------------------------------------------------------
# Card izquierda del header
# ---------------------------------------------------------------------------

class _HeaderCard(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("valheadercard")
        self.setStyleSheet(card_qss("valheadercard"))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 22, 28, 22)
        layout.setSpacing(8)

        bc = QLabel("SAREKO / VALIDACIÓN")
        bc.setStyleSheet(section_label_qss())

        title = QLabel("Detalle de detecciones y revisión humana")
        title.setStyleSheet(title_qss(26))
        title.setWordWrap(True)

        layout.addWidget(bc)
        layout.addWidget(title)


# ---------------------------------------------------------------------------
# Card derecha del header — buscador
# ---------------------------------------------------------------------------

class _SearchCard(QFrame):
    search_requested = Signal(str)
    cleared          = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("valsearchcard")
        self.setStyleSheet(card_qss("valsearchcard"))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 22, 28, 22)
        layout.setSpacing(10)

        lbl = QLabel("FILTRAR POR ESPECIE O ARCHIVO")
        lbl.setStyleSheet(section_label_qss())
        layout.addWidget(lbl)

        row = QHBoxLayout()
        row.setSpacing(8)

        self._field = QLineEdit()
        self._field.setPlaceholderText("Nombre común, científico o archivo…")
        self._field.setStyleSheet(f"""
            QLineEdit {{
                background: rgba(0,0,0,100);
                color: {TEXT_PRIMARY};
                border: 1px solid rgba(153,225,122,80);
                border-radius: 6px;
                padding: 6px 10px;
                font-size: 12px;
            }}
            QLineEdit:focus {{ border-color: {ACCENT}; }}
        """)
        self._field.returnPressed.connect(self._on_search)

        btn_search = QPushButton("Buscar")
        btn_search.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_search.setFixedHeight(32)
        btn_search.setStyleSheet(f"""
            QPushButton {{
                background: rgba(153,225,122,30);
                color: {TEXT_PRIMARY};
                border: 1px solid rgba(153,225,122,100);
                border-radius: 6px;
                font-size: 12px;
                font-weight: 600;
                padding: 0 14px;
            }}
            QPushButton:hover {{ background: rgba(153,225,122,55); }}
        """)
        btn_search.clicked.connect(self._on_search)

        btn_clear = QPushButton("Limpiar")
        btn_clear.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_clear.setFixedHeight(32)
        btn_clear.setStyleSheet(f"""
            QPushButton {{
                background: rgba(74,82,72,40);
                color: {TEXT_PRIMARY};
                border: 1px solid rgba(74,82,72,100);
                border-radius: 6px;
                font-size: 12px;
                font-weight: 600;
                padding: 0 14px;
            }}
            QPushButton:hover {{ background: rgba(74,82,72,70); }}
        """)
        btn_clear.clicked.connect(self._on_clear)

        row.addWidget(self._field, 1)
        row.addWidget(btn_search)
        row.addWidget(btn_clear)
        layout.addLayout(row)

    def _on_search(self) -> None:
        self.search_requested.emit(self._field.text().strip())

    def _on_clear(self) -> None:
        self._field.clear()
        self.cleared.emit()

    @property
    def query(self) -> str:
        return self._field.text().strip()


# ---------------------------------------------------------------------------
# Widget UMAP / fallback de distancias
# ---------------------------------------------------------------------------

class _UMAPWidget(QFrame):
    """
    Muestra un UMAP contextual si hay coordenadas pre-calculadas en
    04-app/data/umap_2d_catalog.pkl; si no, muestra barras de las
    distancias del top-5.
    """

    _umap_cache: Optional[dict] = None   # cargado una vez, compartido entre instancias
    _cache_tried: bool = False

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("umapwidget")
        self.setStyleSheet("""
            QFrame#umapwidget {
                background: rgba(0,0,0,80);
                border: 1px solid rgba(153,225,122,40);
                border-radius: 8px;
            }
        """)
        self.setMinimumHeight(220)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if _MPL_AVAILABLE:
            self._fig    = Figure(figsize=(4, 2.8), facecolor="#050505")
            self._canvas = FigureCanvasQTAgg(self._fig)
            self._canvas.setStyleSheet("background: transparent;")
            layout.addWidget(self._canvas)
            self._ax = self._fig.add_subplot(111)
            self._style_ax()
            self._ensure_umap_cache()
        else:
            lbl = QLabel("matplotlib no disponible")
            lbl.setStyleSheet(body_qss(0.4))
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(lbl)

    @classmethod
    def _ensure_umap_cache(cls) -> None:
        if cls._cache_tried:
            return
        cls._cache_tried = True
        if _UMAP_CACHE.exists():
            try:
                with open(_UMAP_CACHE, "rb") as fh:
                    cls._umap_cache = pickle.load(fh)
            except Exception:
                cls._umap_cache = None

    def _style_ax(self) -> None:
        self._ax.set_facecolor("#0a0a0a")
        self._ax.tick_params(colors="#3a4238", labelsize=6)
        for spine in self._ax.spines.values():
            spine.set_edgecolor("#1a241a")
        self._ax.set_xlabel("UMAP 1", color="#4a5248", fontsize=6)
        self._ax.set_ylabel("UMAP 2", color="#4a5248", fontsize=6)
        self._fig.tight_layout(pad=0.5)

    def set_event(self, top5_candidates: list) -> None:
        if not _MPL_AVAILABLE:
            return
        self._ax.cla()
        self._style_ax()

        if self._umap_cache is not None:
            self._draw_full_umap(top5_candidates)
        else:
            self._draw_top5_bars(top5_candidates)

        self._canvas.draw()

    def _draw_full_umap(self, top5: list) -> None:
        data   = self._umap_cache
        coords = data.get("coords")
        labels = data.get("labels")
        if coords is None or labels is None:
            self._draw_top5_bars(top5)
            return

        import numpy as np
        labels_arr = _np_mpl.asarray(labels)
        unique_sp  = sorted(set(labels_arr))
        cmap       = _mpl_cm.get_cmap("tab20", max(len(unique_sp), 1))
        color_map  = {sp: cmap(i / len(unique_sp)) for i, sp in enumerate(unique_sp)}

        top5_species = {c.get("species", "") for c in top5}
        predicted    = top5[0].get("species", "") if top5 else ""

        # Fondo: todas las especies no top-5
        for sp in unique_sp:
            if sp in top5_species:
                continue
            mask = labels_arr == sp
            pts  = coords[mask]
            self._ax.scatter(
                pts[:, 0], pts[:, 1],
                c=[color_map[sp]], s=3, alpha=0.12, linewidths=0,
            )

        # Top-5 resaltado
        for i, cand in enumerate(top5):
            sp   = cand.get("species", "")
            dist = cand.get("cosine_distance", 0.0)
            mask = labels_arr == sp
            pts  = coords[mask]
            c    = color_map.get(sp, (0.6, 0.6, 0.6, 1.0))
            size = 30 if sp == predicted else 14
            alpha = 0.9 if sp == predicted else 0.55
            label = f"{sp.split()[-1]} (d={dist:.3f})"
            self._ax.scatter(
                pts[:, 0], pts[:, 1],
                c=[c], s=size, alpha=alpha, linewidths=0, label=label,
                marker="*" if sp == predicted else "o",
            )

        self._ax.legend(
            fontsize=5.5, framealpha=0.25, facecolor="#0a0a0a",
            edgecolor="#1f2c1d", labelcolor="#c0c8be",
            loc="best", markerscale=1.2,
        )
        self._ax.set_title("Espacio latente BioCLIP v2", fontsize=6.5,
                           color=ACCENT, pad=3)
        self._fig.tight_layout(pad=0.5)

    def _draw_top5_bars(self, top5: list) -> None:
        if not top5:
            self._ax.text(0.5, 0.5, "Sin candidatos", ha="center", va="center",
                         color="#4a5248", transform=self._ax.transAxes, fontsize=8)
            return

        species   = [c.get("species", "?").split()[-1] for c in top5]
        distances = [c.get("cosine_distance", 0.0) for c in top5]
        colors    = [SUCCESS if i == 0 else "#4a5248" for i in range(len(distances))]

        self._ax.barh(range(len(species)), distances, color=colors, alpha=0.8)
        self._ax.set_yticks(range(len(species)))
        self._ax.set_yticklabels(species, fontsize=6.5, color=TEXT_PRIMARY)
        self._ax.set_xlabel("Distancia coseno", fontsize=6, color="#4a5248")
        self._ax.invert_yaxis()
        self._ax.set_title("Top 5 candidatos", fontsize=6.5, color=ACCENT, pad=3)
        self._fig.tight_layout(pad=0.5)

    def clear(self) -> None:
        if not _MPL_AVAILABLE:
            return
        self._ax.cla()
        self._style_ax()
        self._canvas.draw()


# ---------------------------------------------------------------------------
# Celda de validación (combo + botón Validar / badge + editar)
# ---------------------------------------------------------------------------

class _ValidationCell(QWidget):
    """
    Muestra un combo con los top-5 candidatos + opciones especiales.
    Tras validar, reemplaza el combo por un badge "✓ Categoría" + botón editar.
    """

    validated = Signal(str, str)  # category, custom_species (or "")

    def __init__(self, record: dict, species_catalog: list | None = None, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        self._record          = record
        self._species_catalog = species_catalog or []

        hl = QHBoxLayout(self)
        hl.setContentsMargins(4, 2, 4, 2)
        hl.setSpacing(4)

        # ── estado pendiente ──────────────────────────────────────────
        self._combo = QComboBox()
        self._combo.setStyleSheet(_combo_qss())
        self._combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._populate_combo()

        self._btn_val = QPushButton("Validar")
        self._btn_val.setFixedHeight(24)
        self._btn_val.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_val.setStyleSheet(f"""
            QPushButton {{
                background: {ACCENT};
                color: #0a1a08;
                border: none;
                border-radius: 4px;
                font-size: 11px;
                font-weight: 700;
                padding: 0 10px;
            }}
            QPushButton:hover {{ background: #b0f08a; }}
        """)
        self._btn_val.clicked.connect(self._on_validate)

        # ── estado validado ───────────────────────────────────────────
        self._badge = QLabel()
        self._badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._badge.setFixedHeight(22)

        self._btn_edit = QPushButton("✎")
        self._btn_edit.setFixedSize(22, 22)
        self._btn_edit.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_edit.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: rgba(237,239,236,120);
                border: none;
                font-size: 12px;
            }}
            QPushButton:hover {{ color: {TEXT_PRIMARY}; }}
        """)
        self._btn_edit.clicked.connect(self._on_edit)

        hl.addWidget(self._combo, 1)
        hl.addWidget(self._btn_val)
        hl.addWidget(self._badge)
        hl.addWidget(self._btn_edit)

        # Restaurar estado si ya fue validado
        if record["validation"]["state"] == "validated":
            self._show_validated()
        else:
            self._show_pending()

    def _populate_combo(self) -> None:
        self._combo.clear()
        self._combo.addItem("— Seleccionar —")

        top5 = getattr(self._record["event"], "top5_candidates", [])
        for cand in top5:
            sp   = cand.get("species", "")
            dist = cand.get("cosine_distance", 0.0)
            self._combo.addItem(f"{sp} (d={dist:.3f})")

        for opt in _SPECIAL_OPTS:
            self._combo.addItem(opt)

    def _on_validate(self) -> None:
        idx = self._combo.currentIndex()
        text = self._combo.currentText()

        if text == "— Seleccionar —":
            return

        top5 = getattr(self._record["event"], "top5_candidates", [])
        custom = None

        if text == "Ingreso manual...":
            dlg = QDialog(self)
            dlg.setWindowTitle("Ingresar especie manualmente")
            dlg.setMinimumWidth(380)
            dlg.setStyleSheet("""
                QDialog { background-color: #1a1a1a; color: #edefec; }
                QLabel { color: #edefec; }
                QLineEdit {
                    background-color: #0d0d0d;
                    color: #edefec;
                    border: 1px solid #99e17a;
                    border-radius: 6px;
                    padding: 4px 8px;
                    font-size: 12px;
                }
                QPushButton {
                    background-color: #1f2c1d;
                    color: #99e17a;
                    border: 1px solid #99e17a;
                    border-radius: 6px;
                    padding: 4px 12px;
                    min-width: 60px;
                }
                QPushButton:hover { background-color: #2d3f2a; }
            """)
            dlg_layout = QVBoxLayout(dlg)
            dlg_layout.setSpacing(8)
            lbl_hint = QLabel("Nombre científico de la especie:")
            lbl_hint.setStyleSheet("color: #edefec; font-size: 12px;")
            dlg_layout.addWidget(lbl_hint)

            le = QLineEdit()
            le.setPlaceholderText("Escribe para buscar en el catálogo…")
            completer, display_to_species = _make_completer(self._species_catalog, le)
            if completer is not None:
                completer.activated[str].connect(
                    lambda t, _le=le, _m=display_to_species: QTimer.singleShot(
                        0, lambda: _le.setText(_m.get(t, t))
                    )
                )
                le.setCompleter(completer)
            dlg_layout.addWidget(le)

            btns = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
            )
            btns.accepted.connect(dlg.accept)
            btns.rejected.connect(dlg.reject)
            dlg_layout.addWidget(btns)
            if dlg.exec() != QDialog.DialogCode.Accepted or not le.text().strip():
                return
            custom = le.text().strip()
            category = f"Conocida — {custom}"
        elif 1 <= idx <= len(top5):
            # idx 1 → top-1 = "Correcta"; idx 2-5 → "Top 5"
            category = "Correcta" if idx == 1 else "Top 5"
        else:
            category = text  # "Desconocida" o "Vacío / Ruido"

        # Especie que el usuario confirmó como correcta (para mostrar en panel)
        if custom:
            validated_sp: "str | None" = custom
        elif 1 <= idx <= len(top5):
            validated_sp = top5[idx - 1].get("species", "")
        else:
            validated_sp = text  # "Desconocida" o "Vacío / Ruido"

        self._record["validation"] = {
            "state":             "validated",
            "category":          category,
            "custom_species":    custom,
            "validated_species": validated_sp,
        }
        self._show_validated()
        self.validated.emit(category, custom or "")

    def _on_edit(self) -> None:
        self._record["validation"]["state"] = "pending"
        self._show_pending()

    def _show_validated(self) -> None:
        cat = self._record["validation"].get("category", "Validado")
        short = cat if len(cat) <= 20 else cat[:18] + "…"
        self._badge.setText(f"✓ {short}")
        self._badge.setStyleSheet(validation_badge_qss(cat))
        self._combo.setVisible(False)
        self._btn_val.setVisible(False)
        self._badge.setVisible(True)
        self._btn_edit.setVisible(True)

    def _show_pending(self) -> None:
        self._populate_combo()
        self._combo.setVisible(True)
        self._btn_val.setVisible(True)
        self._badge.setVisible(False)
        self._btn_edit.setVisible(False)

    def set_validated(self, category: str, species: str = "") -> None:
        """Sincroniza el estado validado sin re-emitir la señal."""
        self._record["validation"] = {
            "state":          "validated",
            "category":       category,
            "custom_species": species if species else None,
        }
        self._show_validated()


# ---------------------------------------------------------------------------
# Panel lateral deslizable
# ---------------------------------------------------------------------------

class _SidePanel(QFrame):
    """Panel de detalle que desliza desde la derecha."""

    closed                = Signal()
    validation_saved      = Signal(int, str, str)   # row_idx, category, custom_species
    multi_species_changed = Signal(int, list)        # global_idx, species_list

    def __init__(self, species_catalog: list | None = None, parent=None):
        super().__init__(parent)
        self._species_catalog = species_catalog or []
        self.setObjectName("sidepanel")
        self.setStyleSheet(card_qss("sidepanel"))
        self.setMinimumWidth(0)
        self.setMaximumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        self._anim: Optional[QPropertyAnimation] = None
        self._anim = QPropertyAnimation(self, b"maximumWidth")
        self._anim.setDuration(260)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim_connected: bool = False
        self._icon_magnifier = magnifier_icon(TEXT_PRIMARY, 14)

        self._current_record: Optional[dict] = None
        self._current_row_idx: int = -1

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Scroll interno para el contenido
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("""
            QScrollArea { background: transparent; border: none; }
            QScrollBar:vertical {
                background: rgba(255,255,255,15); width: 5px; margin: 0; border-radius: 2px;
            }
            QScrollBar::handle:vertical {
                background: rgba(153,225,122,100); border-radius: 2px; min-height: 20px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        """)
        scroll.viewport().setStyleSheet("background: transparent;")

        content = QWidget()
        content.setStyleSheet("background: transparent;")
        self._layout = QVBoxLayout(content)
        self._layout.setContentsMargins(16, 14, 16, 16)
        self._layout.setSpacing(12)
        scroll.setWidget(content)
        outer.addWidget(scroll)

        # ── Título + botón cerrar ────────────────────────────────────────
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)

        self._title_lbl = QLabel("DETALLE DEL EVENTO")
        self._title_lbl.setStyleSheet(section_label_qss())

        btn_close = QPushButton("✕")
        btn_close.setFixedSize(22, 22)
        btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_close.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: rgba(237,239,236,100);
                border: none;
                font-size: 12px;
            }}
            QPushButton:hover {{ color: {TEXT_PRIMARY}; }}
        """)
        btn_close.clicked.connect(self.close_panel)
        title_row.addWidget(self._title_lbl)
        title_row.addStretch()
        title_row.addWidget(btn_close)
        self._layout.addLayout(title_row)
        self._layout.addWidget(_sep())

        # ── Nombre común y científico (especie asignada por pipeline) ────
        self._lbl_common_name = QLabel("—")
        self._lbl_common_name.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 17px; font-weight: 700;"
            " background: transparent;"
        )
        self._lbl_common_name.setWordWrap(True)
        self._layout.addWidget(self._lbl_common_name)

        self._lbl_scientific_name = QLabel("—")
        self._lbl_scientific_name.setStyleSheet(
            f"color: {ACCENT}; font-size: 12px; font-style: italic;"
            " background: transparent;"
        )
        self._lbl_scientific_name.setWordWrap(True)
        self._layout.addWidget(self._lbl_scientific_name)

        # ── Frame representativo ─────────────────────────────────────────
        self._frame_lbl = QLabel()
        self._frame_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._frame_lbl.setMinimumHeight(140)
        self._frame_lbl.setStyleSheet("""
            QLabel {
                background: rgba(0,0,0,80);
                border: 1px solid rgba(153,225,122,40);
                border-radius: 6px;
                color: rgba(237,239,236,60);
                font-size: 11px;
            }
        """)
        self._frame_lbl.setText("Frame no disponible")
        self._layout.addWidget(self._frame_lbl)

        # ── Botón ver clip ───────────────────────────────────────────────
        self._btn_clip = QPushButton("▶ Ver clip completo")
        self._btn_clip.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_clip.setFixedHeight(30)
        self._btn_clip.setStyleSheet(f"""
            QPushButton {{
                background: rgba(153,225,122,25);
                color: {TEXT_PRIMARY};
                border: 1px solid rgba(153,225,122,80);
                border-radius: 6px;
                font-size: 12px;
                font-weight: 600;
            }}
            QPushButton:hover {{ background: rgba(153,225,122,50); }}
            QPushButton:disabled {{
                background: rgba(74,82,72,20);
                color: rgba(237,239,236,40);
                border-color: rgba(74,82,72,50);
            }}
        """)
        self._btn_clip.clicked.connect(self._open_clip)
        self._layout.addWidget(self._btn_clip)

        self._layout.addWidget(_sep())

        # ── Distancia coseno ─────────────────────────────────────────────
        dist_row = QHBoxLayout()
        lbl_d = QLabel("DISTANCIA COSENO")
        lbl_d.setStyleSheet(section_label_qss())
        self._lbl_dist = QLabel("—")
        self._lbl_dist.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 16px; font-weight: 700;"
            " background: transparent;"
        )
        self._lbl_dist.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        dist_row.addWidget(lbl_d)
        dist_row.addStretch()
        dist_row.addWidget(self._lbl_dist)
        self._layout.addLayout(dist_row)

        # Confianza badge
        self._badge_conf = QLabel()
        self._badge_conf.setFixedHeight(22)
        self._badge_conf.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._layout.addWidget(self._badge_conf)

        # ── Umbral aplicado ──────────────────────────────────────────────
        umbral_row = QHBoxLayout()
        lbl_umb = QLabel("UMBRAL APLICADO")
        lbl_umb.setStyleSheet(section_label_qss())
        self._lbl_umbral_val = QLabel("0.1866")
        self._lbl_umbral_val.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 13px; font-weight: 700;"
            " background: transparent;"
        )
        self._lbl_umbral_val.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        umbral_row.addWidget(lbl_umb)
        umbral_row.addStretch()
        umbral_row.addWidget(self._lbl_umbral_val)
        self._layout.addLayout(umbral_row)

        # ── Método de asignación ─────────────────────────────────────────
        lbl_met_header = QLabel("MÉTODO DE ASIGNACIÓN")
        lbl_met_header.setStyleSheet(section_label_qss())
        self._layout.addWidget(lbl_met_header)

        self._lbl_metodo = QLabel("—")
        self._lbl_metodo.setStyleSheet(body_qss(0.75))
        self._lbl_metodo.setWordWrap(True)
        self._layout.addWidget(self._lbl_metodo)

        self._layout.addWidget(_sep())

        # ── Validación inline ────────────────────────────────────────────
        lbl_val = QLabel("VALIDACIÓN")
        lbl_val.setStyleSheet(section_label_qss())
        self._layout.addWidget(lbl_val)

        self._val_cell_container = QWidget()
        self._val_cell_container.setStyleSheet("background: transparent;")
        vc_lay = QVBoxLayout(self._val_cell_container)
        vc_lay.setContentsMargins(0, 0, 0, 0)
        self._layout.addWidget(self._val_cell_container)

        self._lbl_validated_species = QLabel("")
        self._lbl_validated_species.setStyleSheet(
            f"color: {ACCENT}; font-size: 11px; font-weight: 600;"
            " background: transparent; padding: 1px 0;"
        )
        self._lbl_validated_species.setWordWrap(True)
        self._lbl_validated_species.hide()
        self._layout.addWidget(self._lbl_validated_species)

        self._layout.addWidget(_sep())

        # ── Especies adicionales ─────────────────────────────────────────
        lbl_extra = QLabel("ESPECIES ADICIONALES")
        lbl_extra.setStyleSheet(section_label_qss())
        self._layout.addWidget(lbl_extra)

        self._extra_species = _ExtraSpeciesSection(self._species_catalog)
        self._extra_species.species_changed.connect(self._on_extra_species_changed)
        self._layout.addWidget(self._extra_species)

        self._layout.addWidget(_sep())

        # ── Top-5 candidatos (tabla pequeña) ────────────────────────────
        lbl_top5 = QLabel("TOP 5 CANDIDATOS")
        lbl_top5.setStyleSheet(section_label_qss())
        self._layout.addWidget(lbl_top5)

        self._top5_table = QTableWidget(0, 2)
        self._top5_table.setHorizontalHeaderLabels(["ESPECIE", "DISTANCIA"])
        hh = self._top5_table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self._top5_table.setColumnWidth(1, 74)
        self._top5_table.verticalHeader().setVisible(False)
        self._top5_table.setShowGrid(False)
        self._top5_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._top5_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._top5_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._top5_table.setStyleSheet(self._top5_qss())
        self._top5_table.setMaximumHeight(160)
        self._layout.addWidget(self._top5_table)

        self._layout.addWidget(_sep())

        # ── UMAP ─────────────────────────────────────────────────────────
        lbl_umap = QLabel("ESPACIO LATENTE")
        lbl_umap.setStyleSheet(section_label_qss())
        self._layout.addWidget(lbl_umap)

        self._umap = _UMAPWidget()
        self._layout.addWidget(self._umap)

        self._layout.addStretch()

    def _top5_qss(self) -> str:
        return f"""
            QTableWidget {{
                background: transparent;
                border: none;
                color: {TEXT_PRIMARY};
                font-size: 11px;
            }}
            QTableWidget::item {{
                padding: 3px 6px;
                border-bottom: 1px solid rgba(255,255,255,12);
                background: transparent;
            }}
            QHeaderView::section {{
                background: transparent;
                color: {ACCENT};
                font-size: 9px;
                font-weight: 700;
                letter-spacing: 1px;
                border: none;
                border-bottom: 1px solid rgba(153,225,122,70);
                padding: 3px 6px;
            }}
        """

    # ------------------------------------------------------------------

    def open_panel(self) -> None:
        if self._anim is None:
            return
        self._anim.stop()
        self._anim.setStartValue(self.maximumWidth())
        self._anim.setEndValue(PANEL_WIDTH)
        if self._anim_connected:
            try:
                self._anim.finished.disconnect()
            except (RuntimeError, TypeError):
                pass
        self._anim.finished.connect(self._on_open_finished)
        self._anim_connected = True
        self._anim.start()

    def close_panel(self) -> None:
        if self._anim is None:
            self.closed.emit()
            return
        self._anim.stop()
        self._anim.setStartValue(self.maximumWidth())
        self._anim.setEndValue(0)
        if self._anim_connected:
            try:
                self._anim.finished.disconnect()
            except (RuntimeError, TypeError):
                pass
            self._anim_connected = False
        self._anim.finished.connect(self._on_close_finished)
        self._anim_connected = True
        self._anim.start()
        self._current_record = None

    def _on_open_finished(self) -> None:
        pass

    def _on_close_finished(self) -> None:
        self.closed.emit()

    def load_record(self, record: dict, row_idx: int = -1) -> None:
        self._current_record  = record
        self._current_row_idx = row_idx
        event    = record["event"]
        filepath = record.get("filepath")

        # Título
        self._title_lbl.setText(record["filename"].upper())

        # Nombre común y científico
        self._lbl_common_name.setText(_get_common_name(event))
        self._lbl_scientific_name.setText(getattr(event, "species", "—"))

        # Frame representativo
        pixmap = None
        frame_idx = getattr(event, "representative_frame_idx", None)
        if filepath and frame_idx is not None:
            pixmap = _load_frame(filepath, frame_idx)
        if pixmap is None and filepath and filepath.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            pixmap = QPixmap(str(filepath))

        if pixmap and not pixmap.isNull():
            scaled = pixmap.scaled(
                PANEL_WIDTH - 40, 160,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._frame_lbl.setPixmap(scaled)
            self._frame_lbl.setText("")
        else:
            self._frame_lbl.clear()
            ts = getattr(event, "representative_timestamp", None)
            txt = f"Frame {frame_idx}"
            if ts is not None:
                txt += f" — {_fmt_time(ts)}"
            self._frame_lbl.setText(txt)

        # Botón clip o imagen
        _is_image = filepath is not None and filepath.suffix.lower() in {".jpg", ".jpeg", ".png"}
        _is_video = filepath is not None and filepath.suffix.lower() in {".mp4", ".avi", ".mov"}
        if _is_image and self._icon_magnifier:
            self._btn_clip.setIcon(self._icon_magnifier)
            self._btn_clip.setIconSize(QSize(14, 14))
            self._btn_clip.setText("Abrir imagen")
        else:
            from PySide6.QtGui import QIcon as _QIcon
            self._btn_clip.setIcon(_QIcon())
            self._btn_clip.setText("▶ Ver clip completo")
        self._btn_clip.setEnabled(_is_image or _is_video)

        # Distancia coseno
        dist = getattr(event, "cosine_distance", None)
        self._lbl_dist.setText(f"{dist:.2f}" if dist is not None else "—")

        # Badge confianza
        conf = _get_confidence_level(event)
        self._badge_conf.setText(conf)
        self._badge_conf.setStyleSheet(badge_qss_for(conf))

        # Método de asignación
        decisor = _get_decisor(event)
        self._lbl_metodo.setText(_METHOD_TEXTS.get(decisor, "—"))

        # Validación inline
        for i in reversed(range(self._val_cell_container.layout().count())):
            w = self._val_cell_container.layout().itemAt(i).widget()
            if w:
                w.deleteLater()
        cell = _ValidationCell(record, self._species_catalog)
        cell.validated.connect(
            lambda cat, sp: self.validation_saved.emit(self._current_row_idx, cat, sp)
        )
        cell.validated.connect(lambda _c, _s: self._refresh_validated_species_label())
        self._val_cell_container.layout().addWidget(cell)
        self._refresh_validated_species_label()

        # Especies adicionales
        self._extra_species.set_data(record.get("extra_species", []))

        # Top-5 tabla
        top5 = getattr(event, "top5_candidates", [])
        self._top5_table.setRowCount(0)
        for i, cand in enumerate(top5):
            sp   = cand.get("species", "—")
            dist = cand.get("cosine_distance", 0.0)
            self._top5_table.insertRow(i)
            self._top5_table.setRowHeight(i, 28)

            it_sp = QTableWidgetItem(sp)
            it_sp.setFont(self._italic_font() if i == 0 else it_sp.font())
            it_d = QTableWidgetItem(f"{dist:.4f}")
            it_d.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            if i == 0:
                for it in (it_sp, it_d):
                    it.setForeground(QColor(SUCCESS))
            self._top5_table.setItem(i, 0, it_sp)
            self._top5_table.setItem(i, 1, it_d)

        # UMAP
        self._umap.set_event(top5)

    def _italic_font(self):
        from PySide6.QtGui import QFont
        f = QFont()
        f.setItalic(True)
        return f

    def sync_validation(self, category: str, species: str = "") -> None:
        """Actualiza la celda inline del panel cuando la tabla valida la misma fila."""
        lay = self._val_cell_container.layout()
        for i in range(lay.count()):
            w = lay.itemAt(i).widget()
            if isinstance(w, _ValidationCell):
                w.set_validated(category, species)
                break
        self._refresh_validated_species_label()

    def _refresh_validated_species_label(self) -> None:
        """Muestra el label con la especie confirmada por el usuario, si el evento está validado."""
        if not self._current_record:
            self._lbl_validated_species.hide()
            return
        val = self._current_record.get("validation", {})
        if val.get("state") != "validated":
            self._lbl_validated_species.hide()
            return

        sp = val.get("validated_species")
        if sp is None:
            # Compatibilidad con sesiones guardadas antes de agregar validated_species
            cat = val.get("category", "")
            if cat == "Correcta":
                sp = getattr(self._current_record["event"], "species", None)
            elif cat.startswith("Conocida — "):
                sp = val.get("custom_species")
            elif cat in ("Desconocida", "Vacío / Ruido"):
                sp = cat

        if not sp or sp in ("Desconocida", "Vacío / Ruido"):
            self._lbl_validated_species.hide()
            return

        common = ""
        for entry in self._species_catalog:
            if entry.get("species") == sp:
                c = entry.get("nombre_comun_es_ar", "")
                if c and str(c).lower() not in ("", "nan", "none"):
                    common = str(c)
                break

        text = (
            f"Especie validada: {common} ({sp})" if common
            else f"Especie validada: {sp}"
        )
        self._lbl_validated_species.setText(text)
        self._lbl_validated_species.show()

    def _on_extra_species_changed(self, species_list: list) -> None:
        if self._current_record is not None:
            self._current_record["extra_species"] = species_list
            self._current_record["multi_species"] = bool(species_list)
        self.multi_species_changed.emit(self._current_row_idx, species_list)

    def _open_clip(self) -> None:
        record = self._current_record
        if not record:
            return
        filepath: Optional[Path] = record.get("filepath")
        if not filepath:
            return

        # Para imágenes: abrir directamente sin extraer clip
        if filepath.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            try:
                os.startfile(str(filepath))
            except Exception:
                pass
            return

        event     = record["event"]
        start_sec = getattr(event, "start_time", None)
        end_sec   = getattr(event, "end_time", None)

        if start_sec is None or end_sec is None:
            try:
                os.startfile(str(filepath))
            except Exception:
                pass
            return

        _CLIPS_DIR.mkdir(parents=True, exist_ok=True)
        clip_name = f"{filepath.stem}_{int(start_sec)}s-{int(end_sec)}s.mp4"
        clip_path = _CLIPS_DIR / clip_name

        try:
            import cv2
            cap = cv2.VideoCapture(str(filepath))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            start_frame = int(start_sec * fps)
            end_frame   = int(end_sec   * fps)
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(clip_path), fourcc, fps, (width, height))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            current = start_frame
            while current <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                current += 1
            cap.release()
            out.release()
            if not clip_path.exists() or clip_path.stat().st_size == 0:
                raise RuntimeError("clip vacío")
            os.startfile(str(clip_path))
        except Exception:
            _msg = QMessageBox(self)
            _msg.setWindowTitle("Aviso")
            _msg.setText("No se pudo recortar el clip. Se abrirá el video completo.")
            _msg.setIcon(QMessageBox.Icon.Warning)
            _msg.setStyleSheet("""
                QMessageBox { background-color: #1a1a1a; color: #edefec; }
                QMessageBox QLabel { color: #edefec; }
                QPushButton {
                    background-color: #1f2c1d; color: #99e17a;
                    border: 1px solid #99e17a; border-radius: 6px;
                    padding: 4px 12px; min-width: 60px;
                }
                QPushButton:hover { background-color: #2d3f2a; }
            """)
            _msg.exec()
            try:
                os.startfile(str(filepath))
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Sección de especies adicionales (panel lateral)
# ---------------------------------------------------------------------------

class _ExtraSpeciesSection(QWidget):
    """
    Lista dinámica de especies adicionales con campo inline para agregar.
    Emite species_changed(list) al agregar o eliminar una especie.
    """

    species_changed = Signal(list)

    def __init__(self, species_catalog: list | None = None, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        self._catalog = species_catalog or []
        self._species: list[str] = []
        self._display_to_species: dict[str, str] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        # Lista dinámica
        self._list_container = QWidget()
        self._list_container.setStyleSheet("background: transparent;")
        self._list_layout = QVBoxLayout(self._list_container)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setSpacing(2)
        root.addWidget(self._list_container)

        # Botón "+ Agregar especie"
        self._btn_add = QPushButton("+ Agregar especie")
        self._btn_add.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_add.setFixedHeight(26)
        self._btn_add.setStyleSheet(f"""
            QPushButton {{
                background: rgba(153,225,122,15);
                color: {ACCENT};
                border: 1px solid rgba(153,225,122,70);
                border-radius: 5px;
                font-size: 11px;
                font-weight: 600;
                padding: 0 10px;
            }}
            QPushButton:hover {{ background: rgba(153,225,122,35); }}
        """)
        self._btn_add.clicked.connect(self._show_input)
        root.addWidget(self._btn_add)

        # Fila de input inline (oculta por defecto)
        self._input_row = QWidget()
        self._input_row.setStyleSheet("background: transparent;")
        in_lay = QHBoxLayout(self._input_row)
        in_lay.setContentsMargins(0, 0, 0, 0)
        in_lay.setSpacing(4)

        self._le = QLineEdit()
        self._le.setPlaceholderText("Escribe para buscar en el catálogo…")
        self._le.setStyleSheet(f"""
            QLineEdit {{
                background: rgba(0,0,0,100);
                color: {TEXT_PRIMARY};
                border: 1px solid rgba(153,225,122,80);
                border-radius: 5px;
                padding: 3px 8px;
                font-size: 11px;
            }}
            QLineEdit:focus {{ border-color: {ACCENT}; }}
        """)
        self._le.returnPressed.connect(self._confirm_add)

        completer, self._display_to_species = _make_completer(self._catalog, self._le)
        if completer is not None:
            completer.activated[str].connect(
                lambda t, _m=self._display_to_species: QTimer.singleShot(
                    0, lambda: self._le.setText(_m.get(t, t))
                )
            )
            self._le.setCompleter(completer)

        _btn_confirm = QPushButton("Confirmar")
        _btn_confirm.setCursor(Qt.CursorShape.PointingHandCursor)
        _btn_confirm.setFixedHeight(26)
        _btn_confirm.setStyleSheet(f"""
            QPushButton {{
                background: rgba(153,225,122,30);
                color: {TEXT_PRIMARY};
                border: 1px solid rgba(153,225,122,100);
                border-radius: 5px;
                font-size: 11px;
                font-weight: 600;
                padding: 0 8px;
            }}
            QPushButton:hover {{ background: rgba(153,225,122,55); }}
        """)
        _btn_confirm.clicked.connect(self._confirm_add)

        _btn_cancel = QPushButton("✕")
        _btn_cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        _btn_cancel.setFixedSize(26, 26)
        _btn_cancel.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: rgba(237,239,236,80);
                border: none;
                font-size: 12px;
            }}
            QPushButton:hover {{ color: {TEXT_PRIMARY}; }}
        """)
        _btn_cancel.clicked.connect(self._hide_input)

        in_lay.addWidget(self._le, 1)
        in_lay.addWidget(_btn_confirm)
        in_lay.addWidget(_btn_cancel)
        root.addWidget(self._input_row)
        self._input_row.hide()

    # ------------------------------------------------------------------

    def set_data(self, species_list: list) -> None:
        self._species = list(species_list)
        self._hide_input()
        self._rebuild_list()

    def get_data(self) -> list:
        return list(self._species)

    def _rebuild_list(self) -> None:
        for i in reversed(range(self._list_layout.count())):
            item = self._list_layout.itemAt(i)
            if item and item.widget():
                item.widget().deleteLater()
        for sp in self._species:
            self._list_layout.addWidget(self._make_row(sp))

    def _make_row(self, species: str) -> QWidget:
        common = ""
        for entry in self._catalog:
            if entry.get("species") == species:
                c = entry.get("nombre_comun_es_ar", "")
                if c and str(c).lower() not in ("", "nan", "none"):
                    common = str(c)
                break
        display = f"{species} — {common}" if common else species

        row = QWidget()
        row.setStyleSheet("background: transparent;")
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(6)

        lbl = QLabel(display)
        lbl.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 11px; font-style: italic;"
            " background: transparent;"
        )
        lbl.setWordWrap(True)

        btn_del = QPushButton("✕")
        btn_del.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_del.setFixedSize(18, 18)
        btn_del.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: rgba(224,92,92,160);
                border: none;
                font-size: 10px;
            }}
            QPushButton:hover {{ color: #e05c5c; }}
        """)
        btn_del.clicked.connect(lambda _, s=species: self._remove(s))

        lay.addWidget(lbl, 1)
        lay.addWidget(btn_del)
        return row

    def _show_input(self) -> None:
        self._le.clear()
        self._btn_add.hide()
        self._input_row.show()
        self._le.setFocus()

    def _hide_input(self) -> None:
        self._input_row.hide()
        self._btn_add.show()

    def _confirm_add(self) -> None:
        sp = self._le.text().strip()
        if not sp or sp in self._species:
            self._hide_input()
            return
        self._species.append(sp)
        self._rebuild_list()
        self._hide_input()
        self.species_changed.emit(list(self._species))

    def _remove(self, species: str) -> None:
        if species in self._species:
            self._species.remove(species)
            self._rebuild_list()
            self.species_changed.emit(list(self._species))


# ---------------------------------------------------------------------------
# Sección de registros: tabla paginada + controles
# ---------------------------------------------------------------------------

class _RegistrosSection(QFrame):
    """Card de ancho completo con tabla de registros y paginación."""

    detail_requested    = Signal(dict, int)      # record dict, global_idx
    delete_requested    = Signal(int)            # índice global en _filtered
    validation_happened = Signal(int, str, str)  # global_idx, category, custom_species

    def __init__(self, species_catalog: list | None = None, parent=None):
        super().__init__(parent)
        self._species_catalog = species_catalog or []
        self.setObjectName("registroscard")
        self.setStyleSheet(card_qss("registroscard"))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 20, 28, 20)
        layout.setSpacing(12)

        # ── Header ────────────────────────────────────────────────────────
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)

        sec = QLabel("SEGUIMIENTO")
        sec.setStyleSheet(section_label_qss())
        header.addWidget(sec)
        header.addStretch()

        # Paginación
        self._btn_prev = self._nav_btn("◀")
        self._lbl_page = QLabel("—")
        self._lbl_page.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 12px; background: transparent;"
        )
        self._lbl_page.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_page.setMinimumWidth(80)
        self._btn_next = self._nav_btn("▶")

        self._btn_prev.clicked.connect(self._on_prev)
        self._btn_next.clicked.connect(self._on_next)

        header.addWidget(self._btn_prev)
        header.addWidget(self._lbl_page)
        header.addWidget(self._btn_next)
        header.addSpacing(12)

        # Exportar
        self._btn_csv  = _export_btn("Exportar CSV")
        self._btn_xlsx = _export_btn("Exportar XLSX")
        self._btn_csv.setEnabled(False)
        self._btn_xlsx.setEnabled(False)
        self._btn_csv.clicked.connect(self._export_csv)
        self._btn_xlsx.clicked.connect(self._export_xlsx)
        header.addWidget(self._btn_csv)
        header.addSpacing(6)
        header.addWidget(self._btn_xlsx)

        layout.addLayout(header)
        layout.addWidget(_sep())

        # ── Tabla ─────────────────────────────────────────────────────────
        self._table = QTableWidget(0, 8)
        self._table.setHorizontalHeaderLabels([
            "ARCHIVO", "INTERVALO", "ESPECIE (COMÚN)",
            "ESPECIE (CIENTÍFICO)", "CONFIANZA", "DECISOR", "VALIDACIÓN", "ACCIONES",
        ])
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(6, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(7, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(0, 140)
        self._table.setColumnWidth(1, 90)
        self._table.setColumnWidth(2, 140)
        self._table.setColumnWidth(3, 160)
        self._table.setColumnWidth(4, 110)
        self._table.setColumnWidth(5, 80)
        self._table.setColumnWidth(6, 180)
        self._table.setColumnWidth(7, 70)
        self._table.verticalHeader().setVisible(False)
        self._table.setShowGrid(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._table.setStyleSheet(_table_qss())
        self._table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._table.setMinimumHeight(120)
        layout.addWidget(self._table, 1)

        # Estado interno (recibe datos del ValidacionTab)
        self._filtered: list[dict] = []
        self._page     = 0
        self._total_pages = 0
        self._open_detail_record: Optional[dict] = None
        self._val_cells: dict[int, _ValidationCell] = {}
        self._active_global_idx: int = -1

    # ── Navigation helpers ────────────────────────────────────────────────

    def _nav_btn(self, text: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setFixedSize(26, 26)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet(f"""
            QPushButton {{
                background: rgba(153,225,122,20);
                color: {TEXT_PRIMARY};
                border: 1px solid rgba(153,225,122,60);
                border-radius: 5px;
                font-size: 11px;
                font-weight: 700;
            }}
            QPushButton:hover    {{ background: rgba(153,225,122,45); }}
            QPushButton:disabled {{
                background: rgba(74,82,72,20);
                color: rgba(237,239,236,40);
                border-color: rgba(74,82,72,40);
            }}
        """)
        return btn

    def _on_prev(self) -> None:
        if self._page > 0:
            self._page -= 1
            self._rebuild_table()

    def _on_next(self) -> None:
        if self._page < self._total_pages - 1:
            self._page += 1
            self._rebuild_table()

    # ── API pública ───────────────────────────────────────────────────────

    def set_data(self, filtered: list[dict], page: int = 0) -> None:
        self._filtered    = filtered
        self._page        = page
        self._total_pages = max(1, -(-len(filtered) // PAGE_SIZE))  # ceil division
        self._rebuild_table()
        has_data = bool(filtered)
        self._btn_csv.setEnabled(has_data)
        self._btn_xlsx.setEnabled(has_data)

    # ── Construcción de la tabla ──────────────────────────────────────────

    def _rebuild_table(self) -> None:
        self._table.setRowCount(0)
        self._val_cells.clear()

        start = self._page * PAGE_SIZE
        page_records = self._filtered[start : start + PAGE_SIZE]

        for local_idx, record in enumerate(page_records):
            global_idx = start + local_idx
            event      = record["event"]
            self._table.insertRow(local_idx)
            self._table.setRowHeight(local_idx, 46)

            # Col 0 — Archivo
            it0 = QTableWidgetItem(record["filename"])
            it0.setToolTip(record["filename"])
            self._table.setItem(local_idx, 0, it0)

            # Col 1 — Intervalo
            st = getattr(event, "start_time", None)
            et = getattr(event, "end_time", None)
            if st is not None and et is not None:
                ivl = f"{_fmt_time(st)} – {_fmt_time(et)}"
            else:
                ivl = "—"
            it1 = QTableWidgetItem(ivl)
            it1.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(local_idx, 1, it1)

            # Col 2 — Especie (común)
            it2 = QTableWidgetItem(_get_common_name(event))
            self._table.setItem(local_idx, 2, it2)

            # Col 3 — Especie (científico) — italic via widget
            species = getattr(event, "species", "—")
            sci_w   = QWidget()
            sci_w.setStyleSheet("background: transparent;")
            sci_l   = QHBoxLayout(sci_w)
            sci_l.setContentsMargins(8, 0, 8, 0)
            sci_lbl = QLabel(species)
            sci_lbl.setStyleSheet(
                f"color: {TEXT_PRIMARY}; font-size: 12px; font-style: italic;"
                " background: transparent;"
            )
            sci_l.addWidget(sci_lbl)
            self._table.setCellWidget(local_idx, 3, sci_w)

            # Col 4 — Confianza (badge)
            conf = _get_confidence_level(event)
            badge_w = QWidget()
            badge_w.setStyleSheet("background: transparent;")
            b_lay   = QHBoxLayout(badge_w)
            b_lay.setContentsMargins(8, 4, 8, 4)
            b_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
            b_lbl = QLabel(conf)
            b_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            b_lbl.setFixedHeight(20)
            b_lbl.setStyleSheet(badge_qss_for(conf))
            b_lay.addWidget(b_lbl)
            self._table.setCellWidget(local_idx, 4, badge_w)

            # Col 5 — Decisor
            it5 = QTableWidgetItem(_get_decisor(event))
            it5.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(local_idx, 5, it5)

            # Col 6 — Validación
            val_cell = _ValidationCell(record, self._species_catalog)
            val_cell.validated.connect(
                lambda cat, sp, gi=global_idx: self.validation_happened.emit(gi, cat, sp)
            )
            self._val_cells[global_idx] = val_cell
            self._table.setCellWidget(local_idx, 6, val_cell)

            # Col 7 — Acciones
            act_w = QWidget()
            act_w.setStyleSheet("background: transparent;")
            act_l = QHBoxLayout(act_w)
            act_l.setContentsMargins(4, 2, 4, 2)
            act_l.setSpacing(4)
            act_l.setAlignment(Qt.AlignmentFlag.AlignCenter)

            btn_ver = icon_eye_btn("Ver detalle")
            btn_del = icon_trash_btn("Eliminar")

            btn_ver.clicked.connect(lambda _=False, r=record, gi=global_idx: self.detail_requested.emit(r, gi))
            btn_del.clicked.connect(lambda _=False, gi=global_idx: self._confirm_delete(gi))

            act_l.addWidget(btn_ver)
            act_l.addWidget(btn_del)
            self._table.setCellWidget(local_idx, 7, act_w)

            if global_idx == self._active_global_idx:
                self._table.selectRow(local_idx)

        # Paginación
        total = len(self._filtered)
        tp    = max(1, -(-total // PAGE_SIZE))
        self._total_pages = tp
        self._lbl_page.setText(f"Pág {self._page + 1} / {tp}")
        self._btn_prev.setEnabled(self._page > 0)
        self._btn_next.setEnabled(self._page < tp - 1)

    def update_table_cell_validation(self, global_idx: int, category: str, species: str = "") -> None:
        """Sincroniza visualmente la celda de la tabla cuando el panel valida esa fila."""
        cell = self._val_cells.get(global_idx)
        if cell:
            cell.set_validated(category, species)

    def set_active_row(self, global_idx: int) -> None:
        self._active_global_idx = global_idx
        self._rebuild_table()

    def clear_active_row(self) -> None:
        self._active_global_idx = -1
        self._rebuild_table()

    def _confirm_delete(self, global_idx: int) -> None:
        msg = QMessageBox(self)
        msg.setWindowTitle("Eliminar registro")
        msg.setText("¿Eliminar este evento de la lista? Esta acción no se puede deshacer.")
        msg.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        msg.setDefaultButton(QMessageBox.StandardButton.No)
        msg.setStyleSheet("""
            QMessageBox { background-color: #1a1a1a; color: #edefec; }
            QMessageBox QLabel { color: #edefec; }
            QPushButton {
                background-color: #1f2c1d; color: #99e17a;
                border: 1px solid #99e17a; border-radius: 6px;
                padding: 4px 12px; min-width: 60px;
            }
            QPushButton:hover { background-color: #2d3f2a; }
        """)
        if msg.exec() == QMessageBox.StandardButton.Yes:
            self.delete_requested.emit(global_idx)

    # ── Exportación ───────────────────────────────────────────────────────

    def _collect_rows(self) -> list[dict]:
        rows = []
        for rec in self._filtered:
            ev   = rec["event"]
            st   = getattr(ev, "start_time", None)
            et   = getattr(ev, "end_time", None)
            ivl  = (
                f"{_fmt_time(st)} – {_fmt_time(et)}"
                if st is not None and et is not None else "—"
            )
            val  = rec["validation"]
            rows.append({
                "archivo":          rec["filename"],
                "intervalo":        ivl,
                "especie":          getattr(ev, "species", "—"),
                "nombre_comun":     _get_common_name(ev),
                "confianza":        _get_confidence_level(ev),
                "decisor":          _get_decisor(ev),
                "distancia_coseno": f"{getattr(ev, 'cosine_distance', '—'):.4f}"
                                    if getattr(ev, "cosine_distance", None) is not None
                                    else "—",
                "validacion":       val.get("category") or "",
                "especie_manual":   val.get("custom_species") or "",
            })
        return rows

    def _export_csv(self) -> None:
        rows = self._collect_rows()
        if not rows:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar CSV", "sareko_validacion.csv", "CSV (*.csv)"
        )
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _export_xlsx(self) -> None:
        rows = self._collect_rows()
        if not rows:
            return
        try:
            import openpyxl
        except ImportError:
            self._export_csv()
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar XLSX", "sareko_validacion.xlsx", "Excel (*.xlsx)"
        )
        if not path:
            return
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Validación SAREKO"
        ws.append(list(rows[0].keys()))
        for row in rows:
            ws.append([str(v) for v in row.values()])
        wb.save(path)


# ---------------------------------------------------------------------------
# Pestaña principal
# ---------------------------------------------------------------------------

class ValidacionTab(QWidget):
    """
    Pestaña Validación completa.

    API pública:
        add_events(filename, events, filepath=None)
            — llamar desde AnalisisTab al completarse cada archivo.
        get_records() -> list[dict]
            — devuelve copia de todos los registros para serializar en sesión.
        restore_records(records)
            — restaura registros desde una sesión guardada.
    """

    validation_changed = Signal()  # emitida al guardar cualquier validación

    def __init__(self, species_catalog: list | None = None, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")

        self._species_catalog = species_catalog or []
        self._records:  list[dict] = []     # todos los registros en memoria
        self._filtered: list[dict] = []     # subconjunto tras aplicar filtro
        self._page        = 0
        self._panel_open  = False
        self._panel_row_idx: int = -1

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(16)

        # ── Header — dos cards lado a lado ────────────────────────────────
        header_row = QHBoxLayout()
        header_row.setSpacing(16)

        self._header_card = _HeaderCard()
        self._search_card = _SearchCard()
        header_row.addWidget(self._header_card, 3)
        header_row.addWidget(self._search_card, 2)
        outer.addLayout(header_row)

        # ── Contenido: tabla + panel lateral ─────────────────────────────
        content_row = QHBoxLayout()
        content_row.setSpacing(12)
        content_row.setContentsMargins(0, 0, 0, 0)

        self._registros = _RegistrosSection(self._species_catalog)
        self._panel     = _SidePanel(self._species_catalog)

        content_row.addWidget(self._registros, 1)
        content_row.addWidget(self._panel)
        outer.addLayout(content_row)

        outer.addStretch()

        # ── Señales ───────────────────────────────────────────────────────
        self._search_card.search_requested.connect(self._apply_filter)
        self._search_card.cleared.connect(lambda: self._apply_filter(""))
        self._registros.detail_requested.connect(self._open_panel)
        self._registros.delete_requested.connect(self._delete_record)
        self._registros.validation_happened.connect(self._on_table_validation)
        self._panel.closed.connect(self._on_panel_closed)
        self._panel.validation_saved.connect(self._on_panel_validation_saved)
        self._panel.multi_species_changed.connect(self._on_multi_species_changed)

    # ------------------------------------------------------------------
    # API pública — sesión

    def get_records(self) -> list[dict]:
        """Devuelve copia superficial de todos los registros en memoria."""
        return list(self._records)

    def restore_records(self, records: list[dict]) -> None:
        """Reemplaza los registros en memoria con los cargados desde sesión."""
        self._records = list(records)
        self._apply_filter(self._search_card.query)

    def reset(self) -> None:
        """Deja la pestaña en el mismo estado que al abrir la app por primera vez."""
        if self._panel_open:
            self._panel.close_panel()
        self._search_card._field.clear()
        self.restore_records([])

    # ------------------------------------------------------------------
    # API pública — eventos

    def add_events(
        self,
        filename: str,
        events: list,
        filepath: Optional[Path] = None,
    ) -> None:
        """Agrega los eventos de un archivo procesado a la tabla."""
        for event in events:
            self._records.append({
                "filename":     filename,
                "filepath":     filepath,
                "event":        event,
                "validation": {
                    "state":          "pending",
                    "category":       None,
                    "custom_species": None,
                },
                "extra_species": [],
                "multi_species": False,
            })
        self._apply_filter(self._search_card.query)

    # ------------------------------------------------------------------
    # Filtro y refresco

    def _apply_filter(self, query: str = "") -> None:
        q = query.strip().lower()
        if not q:
            self._filtered = list(self._records)
        else:
            self._filtered = [
                r for r in self._records
                if q in r["filename"].lower()
                or q in _get_common_name(r["event"]).lower()
                or q in getattr(r["event"], "species", "").lower()
            ]
        self._page = 0
        self._registros.set_data(self._filtered, self._page)

    def _delete_record(self, filtered_idx: int) -> None:
        if 0 <= filtered_idx < len(self._filtered):
            rec = self._filtered[filtered_idx]
            # Eliminar del listado maestro
            try:
                self._records.remove(rec)
            except ValueError:
                pass
        self._apply_filter(self._search_card.query)
        if self._panel_open:
            self._panel.close_panel()

    def _open_panel(self, record: dict, global_idx: int = -1) -> None:
        self._panel_row_idx = global_idx
        self._panel.load_record(record, global_idx)
        self._registros.set_active_row(global_idx)
        if not self._panel_open:
            self._panel.open_panel()
            self._panel_open = True
            self._registros._table.setColumnHidden(2, True)
            self._registros._table.setColumnHidden(3, True)

    def _on_panel_closed(self) -> None:
        self._panel_open    = False
        self._panel_row_idx = -1
        self._registros.clear_active_row()
        self._registros._table.setColumnHidden(2, False)
        self._registros._table.setColumnHidden(3, False)

    def _on_panel_validation_saved(self, row_idx: int, category: str, species: str) -> None:
        self._registros.update_table_cell_validation(row_idx, category, species)
        self.validation_changed.emit()

    def _on_table_validation(self, global_idx: int, category: str, species: str) -> None:
        if self._panel_open and global_idx == self._panel_row_idx:
            self._panel.sync_validation(category, species)
        self.validation_changed.emit()

    def _on_multi_species_changed(self, global_idx: int, species_list: list) -> None:
        if 0 <= global_idx < len(self._filtered):
            rec = self._filtered[global_idx]
            rec["extra_species"] = species_list
            rec["multi_species"] = bool(species_list)
        self.validation_changed.emit()
