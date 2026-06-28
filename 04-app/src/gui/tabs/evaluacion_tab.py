"""
Pestaña Evaluación de SAREKO.

Estructura:
    EvaluacionTab
      ├── header_row              — dos cards lado a lado
      │     ├── _EvalHeaderCard   — breadcrumb, título
      │     └── _GlobalSummaryCard — métricas acumuladas
      └── body_row (QHBoxLayout)
            ├── body_widget (QVBoxLayout)
            │     ├── _LatencyCard       — tabla de latencia histórica por archivo
            │     ├── _ErrorEvalCard     — 4 cards clicables + tabla expandible (con "Ver detalle")
            │     ├── _ChartsCard        — grid 2×2 (3 matplotlib + 1 texto)
            │     └── _MultispeciesCard  — clips con multi_species=True (con "Ver detalle")
            └── _SidePanel              — panel deslizable (importado de validacion_tab)
"""

import csv
import re
from collections import Counter
from pathlib import Path
from typing import Optional

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator as _MaxNLocator
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE   = False
    _MaxNLocator     = None  # type: ignore[assignment]

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .validacion_tab import _SidePanel

from ..styles import (
    ACCENT,
    ERROR,
    NEUTRAL,
    SUCCESS,
    TEXT_PRIMARY,
    WARNING,
    badge_qss,
    body_qss,
    card_qss,
    section_label_qss,
    title_qss,
)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

_ORANGE = "#e07840"

_CATEGORY_KEYS = ["Correcta", "Top 5", "Conocida", "Desconocida"]

_CATEGORY_LABELS = {
    "Correcta":    "Correctas",
    "Top 5":       "Top 5",
    "Conocida":    "Conocida fuera del top 5",
    "Desconocida": "Desconocida",
}
_CATEGORY_COLORS = {
    "Correcta":    SUCCESS,
    "Top 5":       WARNING,
    "Conocida":    _ORANGE,
    "Desconocida": ERROR,
}
_CONF_COLORS = {
    "alta":    SUCCESS,
    "baja":    ERROR,
    "ambiguo": WARNING,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sep() -> QFrame:
    s = QFrame()
    s.setFrameShape(QFrame.Shape.HLine)
    s.setStyleSheet("border: none; border-top: 1px solid rgba(153,225,122,50);")
    s.setFixedHeight(1)
    return s


def _fmt_time(sec: float) -> str:
    s = max(0, int(sec))
    return f"{s // 60}:{s % 60:02d}"


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
            width: 6px; margin: 0; border-radius: 3px;
        }}
        QScrollBar::handle:vertical {{
            background: rgba(153,225,122,100);
            border-radius: 3px; min-height: 20px;
        }}
        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {{ height: 0; }}
    """


def _category_key_for(validation: dict) -> "str | None":
    cat = validation.get("category")
    if not cat:
        return None
    if cat == "Correcta":
        return "Correcta"
    if cat == "Top 5":
        return "Top 5"
    if cat.startswith("Conocida"):
        return "Conocida"
    if cat == "Desconocida":
        return "Desconocida"
    return None


def _expected_species(record: dict) -> str:
    val    = record.get("validation", {})
    cat    = val.get("category", "")
    custom = val.get("custom_species")
    if cat == "Correcta":
        return getattr(record["event"], "species", "—")
    if cat == "Top 5":
        return "— (en top 5)"
    if custom:
        return custom
    return cat or "—"


def _get_decisor(event) -> str:
    if getattr(event, "ambiguous", False):
        return "Consenso"
    if getattr(event, "confidence_level", "") == "alta":
        return "BioCLIP"
    return "KNN"


def _color_rgb(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _get_common_name(event) -> str:
    """Retorna nombre común (es_ar > en) o nombre científico si no hay común."""
    for attr in ("nombre_comun_es_ar", "nombre_comun_en"):
        val = getattr(event, attr, "")
        if val and str(val).lower() not in ("", "nan", "none"):
            return str(val)
    return getattr(event, "species", "—")


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


# ---------------------------------------------------------------------------
# Card izquierda del header
# ---------------------------------------------------------------------------

class _EvalHeaderCard(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("evalheadercard")
        self.setStyleSheet(card_qss("evalheadercard"))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 22, 28, 22)
        layout.setSpacing(8)

        bc = QLabel("SAREKO / EVALUACIÓN")
        bc.setStyleSheet(section_label_qss())

        title = QLabel("Indicadores operativos, gráficos y evaluación")
        title.setStyleSheet(title_qss(26))
        title.setWordWrap(True)

        layout.addWidget(bc)
        layout.addWidget(title)


# ---------------------------------------------------------------------------
# Card derecha del header — métricas globales
# ---------------------------------------------------------------------------

class _GlobalSummaryCard(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("evalglobalcard")
        self.setStyleSheet(card_qss("evalglobalcard"))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 22, 28, 22)
        layout.setSpacing(10)

        sec = QLabel("RESUMEN GLOBAL")
        sec.setStyleSheet(section_label_qss())
        layout.addWidget(sec)
        layout.addWidget(_sep())

        grid = QGridLayout()
        grid.setSpacing(12)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setColumnStretch(1, 1)

        self._val_corridas     = self._mk_val("—")
        self._val_registros    = self._mk_val("—")
        self._val_validaciones = self._mk_val("—")

        for row, (lbl_text, val_widget) in enumerate([
            ("Corridas ejecutadas",      self._val_corridas),
            ("Registros de detección",   self._val_registros),
            ("Validaciones realizadas",  self._val_validaciones),
        ]):
            lbl = QLabel(lbl_text)
            lbl.setStyleSheet(section_label_qss())
            grid.addWidget(lbl,        row, 0)
            grid.addWidget(val_widget, row, 1)

        layout.addLayout(grid)

    def _mk_val(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 22px; font-weight: 700;"
            " background: transparent;"
        )
        lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return lbl

    def update_metrics(self, n_corridas: int, n_registros: int, n_validaciones: int) -> None:
        self._val_corridas.setText(str(n_corridas))
        self._val_registros.setText(str(n_registros))
        self._val_validaciones.setText(str(n_validaciones))

    def reset(self) -> None:
        for v in (self._val_corridas, self._val_registros, self._val_validaciones):
            v.setText("—")


# ---------------------------------------------------------------------------
# Card clicable de categoría de error
# ---------------------------------------------------------------------------

class _ErrorCategoryCard(QFrame):
    clicked_key = Signal(str)

    def __init__(self, key: str, parent=None):
        super().__init__(parent)
        self._key   = key
        self._color = _CATEGORY_COLORS[key]
        self._r, self._g, self._b = _color_rgb(self._color)

        obj = f"errcatcard{re.sub(r'[^a-z0-9]', '', key.lower())}"
        self.setObjectName(obj)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(96)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(6)

        lbl_name = QLabel(_CATEGORY_LABELS[key])
        lbl_name.setStyleSheet(
            f"color: {self._color}; font-size: 10px; font-weight: 700;"
            " letter-spacing: 1px; background: transparent;"
        )
        lbl_name.setWordWrap(True)

        self._lbl_count = QLabel("—")
        self._lbl_count.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 34px; font-weight: 700;"
            " background: transparent;"
        )

        layout.addWidget(lbl_name)
        layout.addWidget(self._lbl_count)
        layout.addStretch()

        self._apply_style(False)

    def _apply_style(self, active: bool) -> None:
        name = self.objectName()
        r, g, b = self._r, self._g, self._b
        if active:
            self.setStyleSheet(f"""
                QFrame#{name} {{
                    background: rgba({r}, {g}, {b}, 40);
                    border: 2px solid rgba({r}, {g}, {b}, 180);
                    border-radius: 12px;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QFrame#{name} {{
                    background: rgba(0, 0, 0, 178);
                    border: 2px solid rgba({r}, {g}, {b}, 60);
                    border-radius: 12px;
                }}
                QFrame#{name}:hover {{
                    background: rgba({r}, {g}, {b}, 20);
                    border: 2px solid rgba({r}, {g}, {b}, 110);
                }}
            """)

    def set_count(self, n: int) -> None:
        self._lbl_count.setText(str(n))

    def set_active(self, active: bool) -> None:
        self._apply_style(active)

    def mousePressEvent(self, event) -> None:
        self.clicked_key.emit(self._key)
        super().mousePressEvent(event)


# ---------------------------------------------------------------------------
# Tabla expandible de casos de una categoría
# ---------------------------------------------------------------------------

class _ErrorDetailsTable(QFrame):
    detail_requested = Signal(dict)   # record dict al hacer click en "Ver detalle"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("errdetailstable")
        self.setStyleSheet("""
            QFrame#errdetailstable {
                background: rgba(0, 0, 0, 100);
                border: 1px solid rgba(153, 225, 122, 50);
                border-radius: 10px;
            }
        """)
        self.hide()

        self._rows: list[dict] = []
        self._current_key = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 14, 20, 14)
        layout.setSpacing(10)

        hrow = QHBoxLayout()
        self._title_lbl = QLabel()
        self._title_lbl.setStyleSheet(section_label_qss())
        hrow.addWidget(self._title_lbl)
        hrow.addStretch()

        self._btn_csv  = _export_btn("Exportar CSV")
        self._btn_xlsx = _export_btn("Exportar XLSX")
        self._btn_csv.clicked.connect(self._export_csv)
        self._btn_xlsx.clicked.connect(self._export_xlsx)
        hrow.addWidget(self._btn_csv)
        hrow.addSpacing(6)
        hrow.addWidget(self._btn_xlsx)
        layout.addLayout(hrow)
        layout.addWidget(_sep())

        self._table = QTableWidget(0, 9)
        self._table.setHorizontalHeaderLabels([
            "ARCHIVO", "INTERVALO", "ESP. ESPERADA",
            "ESP. PREDICHA", "CONFIANZA", "DECISOR",
            "TOP 5 CANDIDATOS", "ESPECIES ADICIONALES", "ACCIONES",
        ])
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(8, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(0, 130)
        self._table.setColumnWidth(1, 80)
        self._table.setColumnWidth(4, 100)
        self._table.setColumnWidth(5, 80)
        self._table.setColumnWidth(8, 110)
        self._table.verticalHeader().setVisible(False)
        self._table.setShowGrid(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._table.setStyleSheet(_table_qss())
        self._table.setMinimumHeight(80)
        self._table.setMaximumHeight(300)
        layout.addWidget(self._table)

    # ------------------------------------------------------------------

    def show_category(self, key: str, records: list[dict]) -> None:
        self._current_key = key
        self._rows = records
        self._title_lbl.setText(f"CASOS: {_CATEGORY_LABELS.get(key, key).upper()}")
        self._rebuild()
        self.show()

    def _rebuild(self) -> None:
        self._table.setRowCount(0)
        for i, record in enumerate(self._rows):
            event = record["event"]
            self._table.insertRow(i)
            self._table.setRowHeight(i, 40)

            # Col 0 — Archivo
            it0 = QTableWidgetItem(record["filename"])
            it0.setToolTip(record["filename"])
            self._table.setItem(i, 0, it0)

            # Col 1 — Intervalo
            st = getattr(event, "start_time", None)
            et = getattr(event, "end_time", None)
            ivl = (
                f"{_fmt_time(st)} – {_fmt_time(et)}"
                if st is not None and et is not None else "—"
            )
            it1 = QTableWidgetItem(ivl)
            it1.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(i, 1, it1)

            # Col 2 — Especie esperada (italic)
            self._table.setCellWidget(i, 2, self._italic_cell(_expected_species(record)))

            # Col 3 — Especie predicha (italic)
            self._table.setCellWidget(i, 3, self._italic_cell(getattr(event, "species", "—")))

            # Col 4 — Confianza (badge)
            conf = "ambiguo" if getattr(event, "ambiguous", False) else getattr(event, "confidence_level", "—")
            badge_w = QWidget()
            badge_w.setStyleSheet("background: transparent;")
            bl = QHBoxLayout(badge_w)
            bl.setContentsMargins(6, 3, 6, 3)
            bl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            b_lbl = QLabel(conf)
            b_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            b_lbl.setFixedHeight(20)
            b_lbl.setStyleSheet(badge_qss(_CONF_COLORS.get(conf, NEUTRAL)))
            bl.addWidget(b_lbl)
            self._table.setCellWidget(i, 4, badge_w)

            # Col 5 — Decisor
            it5 = QTableWidgetItem(_get_decisor(event))
            it5.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(i, 5, it5)

            # Col 6 — Top 5 candidatos
            top5 = getattr(event, "top5_candidates", [])
            top5_str = ", ".join(c.get("species", "?") for c in top5) if top5 else "—"
            it6 = QTableWidgetItem(top5_str)
            it6.setToolTip(top5_str)
            self._table.setItem(i, 6, it6)

            # Col 7 — Especies adicionales
            extra = record.get("extra_species", [])
            extra_str = ", ".join(extra) if extra else "—"
            it7 = QTableWidgetItem(extra_str)
            it7.setToolTip(extra_str)
            self._table.setItem(i, 7, it7)

            # Col 8 — Acciones: Ver detalle
            act_w = QWidget()
            act_w.setStyleSheet("background: transparent;")
            act_l = QHBoxLayout(act_w)
            act_l.setContentsMargins(4, 2, 4, 2)
            act_l.setSpacing(0)
            btn_ver = _action_btn("Ver detalle")
            btn_ver.clicked.connect(
                lambda _=False, r=record: self.detail_requested.emit(r)
            )
            act_l.addWidget(btn_ver)
            act_l.addStretch()
            self._table.setCellWidget(i, 8, act_w)

    def _italic_cell(self, text: str) -> QWidget:
        w = QWidget()
        w.setStyleSheet("background: transparent;")
        lay = QHBoxLayout(w)
        lay.setContentsMargins(8, 0, 8, 0)
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 12px; font-style: italic; background: transparent;"
        )
        lbl.setToolTip(text)
        lay.addWidget(lbl)
        return w

    # ------------------------------------------------------------------
    # Exportación

    def _collect_rows(self) -> list[dict]:
        out = []
        for rec in self._rows:
            ev  = rec["event"]
            st  = getattr(ev, "start_time", None)
            et  = getattr(ev, "end_time", None)
            ivl = (
                f"{_fmt_time(st)} – {_fmt_time(et)}"
                if st is not None and et is not None else "—"
            )
            top5  = getattr(ev, "top5_candidates", [])
            extra = rec.get("extra_species", [])
            out.append({
                "archivo":              rec["filename"],
                "intervalo":            ivl,
                "especie_esperada":     _expected_species(rec),
                "especie_predicha":     getattr(ev, "species", "—"),
                "confianza":            getattr(ev, "confidence_level", "—"),
                "decisor":              _get_decisor(ev),
                "top5_candidatos":      ", ".join(c.get("species", "?") for c in top5),
                "especies_adicionales": ", ".join(extra) if extra else "",
            })
        return out

    def _export_csv(self) -> None:
        rows = self._collect_rows()
        if not rows:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar CSV",
            f"sareko_eval_{self._current_key.lower()}.csv", "CSV (*.csv)"
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
            self, "Exportar XLSX",
            f"sareko_eval_{self._current_key.lower()}.xlsx", "Excel (*.xlsx)"
        )
        if not path:
            return
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = f"Eval {self._current_key}"
        ws.append(list(rows[0].keys()))
        for row in rows:
            ws.append([str(v) for v in row.values()])
        wb.save(path)


# ---------------------------------------------------------------------------
# Card de evaluación de latencia (ancho completo) — primera card de contenido
# ---------------------------------------------------------------------------

class _LatencyCard(QFrame):
    """
    Tabla histórica de latencia por archivo procesado.
    Acumula todas las corridas (persiste en history.json via latency_records).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("latencycard")
        self.setStyleSheet(card_qss("latencycard"))

        self._records: list[dict] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(16)

        # ── Header ───────────────────────────────────────────────────────
        hrow = QHBoxLayout()
        hrow.setContentsMargins(0, 0, 0, 0)

        sec = QLabel("EVALUACIÓN DE LATENCIA")
        sec.setStyleSheet(section_label_qss())
        hrow.addWidget(sec)
        hrow.addStretch()

        self._btn_csv  = _export_btn("Exportar CSV")
        self._btn_xlsx = _export_btn("Exportar XLSX")
        self._btn_csv.setEnabled(False)
        self._btn_xlsx.setEnabled(False)
        self._btn_csv.clicked.connect(self._export_csv)
        self._btn_xlsx.clicked.connect(self._export_xlsx)
        hrow.addWidget(self._btn_csv)
        hrow.addSpacing(6)
        hrow.addWidget(self._btn_xlsx)
        layout.addLayout(hrow)
        layout.addWidget(_sep())

        # ── Estado vacío ─────────────────────────────────────────────────
        self._empty_lbl = QLabel("Sin corridas registradas")
        self._empty_lbl.setStyleSheet(
            f"color: {NEUTRAL}; font-size: 13px; font-style: italic;"
            " background: transparent;"
        )
        self._empty_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_lbl.setMinimumHeight(56)
        layout.addWidget(self._empty_lbl)

        # ── Tabla ─────────────────────────────────────────────────────────
        self._table = QTableWidget(0, 7)
        self._table.setHorizontalHeaderLabels([
            "ARCHIVO", "TIPO", "DURACIÓN", "MODO",
            "FRAMES ANALIZADOS", "TIEMPO PROCESAMIENTO", "FACTOR",
        ])
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(6, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(1, 72)
        self._table.setColumnWidth(2, 80)
        self._table.setColumnWidth(3, 90)
        self._table.setColumnWidth(4, 140)
        self._table.setColumnWidth(5, 162)
        self._table.setColumnWidth(6, 76)
        self._table.verticalHeader().setVisible(False)
        self._table.setShowGrid(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._table.setStyleSheet(_table_qss())
        self._table.setMinimumHeight(80)
        self._table.setMaximumHeight(320)
        self._table.hide()
        layout.addWidget(self._table)

    # ------------------------------------------------------------------

    def update_data(self, records: list[dict]) -> None:
        self._records = records
        has = bool(records)
        self._empty_lbl.setVisible(not has)
        self._table.setVisible(has)
        self._btn_csv.setEnabled(has)
        self._btn_xlsx.setEnabled(has)
        if has:
            self._rebuild()

    def _rebuild(self) -> None:
        self._table.setRowCount(0)
        for i, rec in enumerate(self._records):
            self._table.insertRow(i)
            self._table.setRowHeight(i, 36)

            filename      = rec.get("filename", "—")
            file_type     = rec.get("type", "—")
            duration_sec  = rec.get("duration_sec")
            mode          = rec.get("mode", "—")
            frames        = rec.get("frames_processed")
            processing_sec = float(rec.get("processing_sec") or 0.0)

            # Col 0 — Archivo
            it0 = QTableWidgetItem(filename)
            it0.setToolTip(filename)
            self._table.setItem(i, 0, it0)

            # Col 1 — Tipo
            it1 = QTableWidgetItem(file_type.capitalize())
            it1.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(i, 1, it1)

            # Col 2 — Duración
            dur_str = _fmt_time(duration_sec) if duration_sec is not None else "—"
            it2 = QTableWidgetItem(dur_str)
            it2.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(i, 2, it2)

            # Col 3 — Modo
            it3 = QTableWidgetItem(mode)
            it3.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(i, 3, it3)

            # Col 4 — Frames analizados
            it4 = QTableWidgetItem(str(frames) if frames is not None else "—")
            it4.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(i, 4, it4)

            # Col 5 — Tiempo de procesamiento
            it5 = QTableWidgetItem(_fmt_time(processing_sec))
            it5.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(i, 5, it5)

            # Col 6 — Factor
            if duration_sec is not None and duration_sec > 0:
                factor_str = f"{processing_sec / duration_sec:.2f}×"
            else:
                factor_str = "—"
            it6 = QTableWidgetItem(factor_str)
            it6.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(i, 6, it6)

    # ------------------------------------------------------------------
    # Exportación

    def _collect_rows(self) -> list[dict]:
        out = []
        for rec in self._records:
            duration_sec   = rec.get("duration_sec")
            processing_sec = float(rec.get("processing_sec") or 0.0)
            frames         = rec.get("frames_processed")
            dur_str        = _fmt_time(duration_sec) if duration_sec is not None else "—"
            factor_str = (
                f"{processing_sec / duration_sec:.2f}×"
                if duration_sec is not None and duration_sec > 0 else "—"
            )
            out.append({
                "archivo":              rec.get("filename", "—"),
                "tipo":                 rec.get("type", "—"),
                "duracion":             dur_str,
                "modo":                 rec.get("mode", "—"),
                "frames_analizados":    str(frames) if frames is not None else "—",
                "tiempo_procesamiento": _fmt_time(processing_sec),
                "factor":               factor_str,
            })
        return out

    def _export_csv(self) -> None:
        rows = self._collect_rows()
        if not rows:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar CSV", "sareko_latencia.csv", "CSV (*.csv)"
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
            self, "Exportar XLSX", "sareko_latencia.xlsx", "Excel (*.xlsx)"
        )
        if not path:
            return
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Latencia SAREKO"
        ws.append(list(rows[0].keys()))
        for row in rows:
            ws.append([str(v) for v in row.values()])
        wb.save(path)


# ---------------------------------------------------------------------------
# Card de evaluación de errores (ancho completo)
# ---------------------------------------------------------------------------

class _ErrorEvalCard(QFrame):
    detail_requested = Signal(dict)   # propagado desde _ErrorDetailsTable

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("errorevalcard")
        self.setStyleSheet(card_qss("errorevalcard"))

        self._records_by_key: dict[str, list[dict]] = {k: [] for k in _CATEGORY_KEYS}
        self._active_key: "str | None" = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(16)

        sec = QLabel("EVALUACIÓN DE ERRORES")
        sec.setStyleSheet(section_label_qss())
        layout.addWidget(sec)
        layout.addWidget(_sep())

        # 4 cards clicables en fila
        cards_row = QHBoxLayout()
        cards_row.setSpacing(12)
        self._cat_cards: dict[str, _ErrorCategoryCard] = {}
        for key in _CATEGORY_KEYS:
            card = _ErrorCategoryCard(key)
            card.clicked_key.connect(self._on_category_clicked)
            self._cat_cards[key] = card
            cards_row.addWidget(card)
        layout.addLayout(cards_row)

        # Tabla expandible
        self._details = _ErrorDetailsTable()
        self._details.detail_requested.connect(self.detail_requested)
        layout.addWidget(self._details)

    def _on_category_clicked(self, key: str) -> None:
        if self._active_key == key and self._details.isVisible():
            self._details.hide()
            self._cat_cards[key].set_active(False)
            self._active_key = None
            return

        if self._active_key:
            self._cat_cards[self._active_key].set_active(False)

        self._active_key = key
        self._cat_cards[key].set_active(True)
        self._details.show_category(key, self._records_by_key[key])

    def update_data(self, records: list[dict]) -> None:
        by_key: dict[str, list[dict]] = {k: [] for k in _CATEGORY_KEYS}
        for rec in records:
            val = rec.get("validation", {})
            if val.get("state") != "validated":
                continue
            key = _category_key_for(val)
            if key:
                by_key[key].append(rec)

        self._records_by_key = by_key
        for key, card in self._cat_cards.items():
            card.set_count(len(by_key[key]))

        if self._active_key and self._details.isVisible():
            self._details.show_category(
                self._active_key, self._records_by_key[self._active_key]
            )


# ---------------------------------------------------------------------------
# Card de ancho completo — clips con múltiples especies
# ---------------------------------------------------------------------------

class _MultispeciesCard(QFrame):
    """
    Muestra todos los registros donde multi_species=True.
    Emite detail_requested(record, global_idx) al pulsar 'Ver detalle'.
    """

    detail_requested = Signal(dict, int)   # record, índice en la lista global de records

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("multispeciescard")
        self.setStyleSheet(card_qss("multispeciescard"))

        self._records:     list[dict] = []   # sólo los multi_species=True
        self._all_records: list[dict] = []   # referencia completa para calcular índice global

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(16)

        # ── Header ───────────────────────────────────────────────────────
        hrow = QHBoxLayout()
        hrow.setContentsMargins(0, 0, 0, 0)

        sec = QLabel("CLIPS CON MÚLTIPLES ESPECIES")
        sec.setStyleSheet(section_label_qss())
        hrow.addWidget(sec)
        hrow.addStretch()

        self._btn_csv = _export_btn("Exportar CSV")
        self._btn_csv.setEnabled(False)
        self._btn_csv.clicked.connect(self._export_csv)
        hrow.addWidget(self._btn_csv)

        layout.addLayout(hrow)
        layout.addWidget(_sep())

        # ── Estado vacío ─────────────────────────────────────────────────
        self._empty_lbl = QLabel("Sin clips con múltiples especies registrados")
        self._empty_lbl.setStyleSheet(
            f"color: {NEUTRAL}; font-size: 13px; font-style: italic;"
            " background: transparent;"
        )
        self._empty_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_lbl.setMinimumHeight(56)
        layout.addWidget(self._empty_lbl)

        # ── Tabla ─────────────────────────────────────────────────────────
        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels([
            "ARCHIVO", "INTERVALO", "ESPECIE PRINCIPAL",
            "ESPECIES ADICIONALES", "DECISOR", "ACCIONES",
        ])
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(0, 130)
        self._table.setColumnWidth(1, 90)
        self._table.setColumnWidth(4, 86)
        self._table.setColumnWidth(5, 110)
        self._table.verticalHeader().setVisible(False)
        self._table.setShowGrid(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._table.setStyleSheet(_table_qss())
        self._table.setMinimumHeight(80)
        self._table.setMaximumHeight(320)
        self._table.hide()
        layout.addWidget(self._table)

    # ------------------------------------------------------------------

    def update_data(self, records: list[dict]) -> None:
        self._all_records = records
        self._records = [r for r in records if r.get("multi_species")]
        has = bool(self._records)
        self._empty_lbl.setVisible(not has)
        self._table.setVisible(has)
        self._btn_csv.setEnabled(has)
        if has:
            self._rebuild()

    def _rebuild(self) -> None:
        self._table.setRowCount(0)
        for i, record in enumerate(self._records):
            event = record["event"]
            self._table.insertRow(i)
            self._table.setRowHeight(i, 46)

            # Col 0 — Archivo
            it0 = QTableWidgetItem(record["filename"])
            it0.setToolTip(record["filename"])
            self._table.setItem(i, 0, it0)

            # Col 1 — Intervalo
            st = getattr(event, "start_time", None)
            et = getattr(event, "end_time", None)
            ivl = (
                f"{_fmt_time(st)} – {_fmt_time(et)}"
                if st is not None and et is not None else "—"
            )
            it1 = QTableWidgetItem(ivl)
            it1.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(i, 1, it1)

            # Col 2 — Especie principal: nombre común + nombre científico en itálica
            common   = _get_common_name(event)
            sci_name = getattr(event, "species", "—")
            cell_w   = QWidget()
            cell_w.setStyleSheet("background: transparent;")
            cell_l   = QVBoxLayout(cell_w)
            cell_l.setContentsMargins(8, 3, 8, 3)
            cell_l.setSpacing(1)
            lbl_common = QLabel(common)
            lbl_common.setStyleSheet(
                f"color: {TEXT_PRIMARY}; font-size: 12px; background: transparent;"
            )
            lbl_sci = QLabel(sci_name)
            lbl_sci.setStyleSheet(
                f"color: {ACCENT}; font-size: 10px; font-style: italic;"
                " background: transparent;"
            )
            cell_l.addWidget(lbl_common)
            cell_l.addWidget(lbl_sci)
            self._table.setCellWidget(i, 2, cell_w)

            # Col 3 — Especies adicionales
            extra     = record.get("extra_species", [])
            extra_str = ", ".join(extra) if extra else "—"
            it3 = QTableWidgetItem(extra_str)
            it3.setToolTip(extra_str)
            self._table.setItem(i, 3, it3)

            # Col 4 — Decisor
            it4 = QTableWidgetItem(_get_decisor(event))
            it4.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(i, 4, it4)

            # Col 5 — Acciones: Ver detalle
            act_w = QWidget()
            act_w.setStyleSheet("background: transparent;")
            act_l = QHBoxLayout(act_w)
            act_l.setContentsMargins(4, 2, 4, 2)
            act_l.setSpacing(0)
            btn_ver = _action_btn("Ver detalle")
            try:
                global_idx = self._all_records.index(record)
            except ValueError:
                global_idx = -1
            btn_ver.clicked.connect(
                lambda _=False, r=record, gi=global_idx: self.detail_requested.emit(r, gi)
            )
            act_l.addWidget(btn_ver)
            act_l.addStretch()
            self._table.setCellWidget(i, 5, act_w)

    # ------------------------------------------------------------------
    # Exportación

    def _export_csv(self) -> None:
        if not self._records:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar CSV", "sareko_multi_especies.csv", "CSV (*.csv)"
        )
        if not path:
            return
        rows = []
        for rec in self._records:
            ev  = rec["event"]
            st  = getattr(ev, "start_time", None)
            et  = getattr(ev, "end_time", None)
            ivl = (
                f"{_fmt_time(st)} – {_fmt_time(et)}"
                if st is not None and et is not None else "—"
            )
            rows.append({
                "archivo":              rec["filename"],
                "intervalo":            ivl,
                "especie_principal":    getattr(ev, "species", "—"),
                "nombre_comun":         _get_common_name(ev),
                "especies_adicionales": ", ".join(rec.get("extra_species", [])),
                "decisor":              _get_decisor(ev),
            })
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


# ---------------------------------------------------------------------------
# Contenedor individual de gráfico matplotlib
# ---------------------------------------------------------------------------

class _ChartFrame(QFrame):
    def __init__(self, obj_name: str, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName(obj_name)
        self.setStyleSheet(f"""
            QFrame#{obj_name} {{
                background: rgba(0, 0, 0, 120);
                border: 1px solid rgba(153, 225, 122, 40);
                border-radius: 10px;
            }}
        """)
        self.setMinimumHeight(230)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 8)
        layout.setSpacing(4)

        lbl = QLabel(title.upper())
        lbl.setStyleSheet(section_label_qss())
        layout.addWidget(lbl)

        if _MPL_AVAILABLE:
            self._fig    = Figure(figsize=(4, 2.6), facecolor="#0d0d0d")
            self._canvas = FigureCanvasQTAgg(self._fig)
            self._canvas.setStyleSheet("background: transparent;")
            self._ax     = self._fig.add_subplot(111)
            self._style_ax()
            layout.addWidget(self._canvas, 1)
            self._no_data()
        else:
            ph = QLabel("matplotlib no disponible")
            ph.setStyleSheet(body_qss(0.4))
            ph.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(ph, 1)

    def _style_ax(self) -> None:
        self._ax.set_facecolor("#0a0a0a")
        self._ax.tick_params(colors="#4a5248", labelsize=7)
        for spine in self._ax.spines.values():
            spine.set_edgecolor("#1a241a")

    def _no_data(self, msg: str = "Sin datos") -> None:
        if not _MPL_AVAILABLE:
            return
        self._ax.cla()
        self._style_ax()
        self._ax.text(
            0.5, 0.5, msg,
            ha="center", va="center",
            color="#4a5248", transform=self._ax.transAxes, fontsize=9,
        )
        self._canvas.draw()

    def clear(self) -> None:
        if _MPL_AVAILABLE:
            self._ax.cla()
            self._style_ax()

    def draw(self) -> None:
        if _MPL_AVAILABLE:
            try:
                self._fig.tight_layout(pad=0.8)
            except Exception:
                pass
            self._canvas.draw()

    @property
    def ax(self):
        return self._ax if _MPL_AVAILABLE else None

    @property
    def fig(self):
        return self._fig if _MPL_AVAILABLE else None


# ---------------------------------------------------------------------------
# Cuarto cuadrante: resumen operativo en texto
# ---------------------------------------------------------------------------

class _SummaryTextCard(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("evalsumtextcard")
        self.setStyleSheet("""
            QFrame#evalsumtextcard {
                background: rgba(0, 0, 0, 120);
                border: 1px solid rgba(153, 225, 122, 40);
                border-radius: 10px;
            }
        """)
        self.setMinimumHeight(230)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 14, 20, 14)
        layout.setSpacing(10)

        sec = QLabel("RESUMEN OPERATIVO")
        sec.setStyleSheet(section_label_qss())
        layout.addWidget(sec)
        layout.addWidget(_sep())

        grid = QGridLayout()
        grid.setSpacing(10)
        grid.setContentsMargins(0, 4, 0, 0)
        grid.setColumnStretch(1, 1)

        self._vals: list[QLabel] = []
        for i, txt in enumerate([
            "Corridas completadas",
            "Corridas con error",
            "Frames procesados",
            "Registros de detección",
        ]):
            lbl = QLabel(txt)
            lbl.setStyleSheet(section_label_qss())
            val = QLabel("—")
            val.setStyleSheet(
                f"color: {TEXT_PRIMARY}; font-size: 18px; font-weight: 700;"
                " background: transparent;"
            )
            val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            grid.addWidget(lbl, i, 0)
            grid.addWidget(val, i, 1)
            self._vals.append(val)

        layout.addLayout(grid)
        layout.addStretch()

    def update_data(self, batch_summary: dict, n_records: int) -> None:
        files       = batch_summary.get("files", [])
        n_completed = sum(1 for f in files if f.get("state") == "completado")
        n_error     = sum(1 for f in files if f.get("state") == "error")
        n_frames    = batch_summary.get("total_frames", 0)

        self._vals[0].setText(str(n_completed))
        self._vals[1].setText(str(n_error))
        self._vals[2].setText(str(n_frames))
        self._vals[3].setText(str(n_records))

    def reset(self) -> None:
        for v in self._vals:
            v.setText("—")


# ---------------------------------------------------------------------------
# Card con grid 2×2 de gráficos
# ---------------------------------------------------------------------------

class _ChartsCard(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("chartscard")
        self.setStyleSheet(card_qss("chartscard"))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(16)

        sec = QLabel("INDICADORES VISUALES")
        sec.setStyleSheet(section_label_qss())
        layout.addWidget(sec)
        layout.addWidget(_sep())

        grid = QGridLayout()
        grid.setSpacing(12)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        self._ch_species = _ChartFrame("chartspecies",  "Distribución de especies detectadas")
        self._ch_conf    = _ChartFrame("chartconf",     "Distribución de nivel de confianza")
        self._ch_status  = _ChartFrame("chartstatus",   "Corridas por fecha")
        self._ch_summary = _SummaryTextCard()

        grid.addWidget(self._ch_species,  0, 0)
        grid.addWidget(self._ch_conf,     0, 1)
        grid.addWidget(self._ch_status,   1, 0)
        grid.addWidget(self._ch_summary,  1, 1)

        layout.addLayout(grid)

    def update_data(self, records: list[dict], batch_summary: dict, history: "dict | None" = None) -> None:
        self._draw_species(records, history)
        self._draw_confidence(records, history)
        self._draw_runs_by_date(history)
        self._ch_summary.update_data(batch_summary, len(records))

    def reset(self) -> None:
        for ch in (self._ch_species, self._ch_conf, self._ch_status):
            ch._no_data()
        self._ch_summary.reset()

    # ------------------------------------------------------------------
    # Dibujo de cada gráfico

    def _draw_species(self, records: list[dict], history: "dict | None" = None) -> None:
        ch = self._ch_species
        if not _MPL_AVAILABLE:
            return
        ch.clear()
        ax = ch.ax

        if history and history.get("species_counts"):
            counts = Counter(history["species_counts"])
        else:
            counts = Counter(
                getattr(r["event"], "species", "")
                for r in records
                if getattr(r["event"], "species", "")
            )
        if not counts:
            ch._no_data("Sin detecciones")
            return

        top10   = counts.most_common(10)
        species = [s.split()[-1] for s, _ in top10]
        values  = [c for _, c in top10]

        ax.barh(range(len(species)), values, color=ACCENT, alpha=0.85)
        ax.set_yticks(range(len(species)))
        ax.set_yticklabels(
            species, fontsize=7, color=TEXT_PRIMARY, fontstyle="italic"
        )
        ax.set_xlabel("Detecciones", fontsize=6, color="#4a5248")
        ax.tick_params(axis="x", labelsize=6, colors="#4a5248")
        ax.invert_yaxis()
        ax.set_title("Top 10 especies", fontsize=7, color=ACCENT, pad=4)
        ch.draw()

    def _draw_confidence(self, records: list[dict], history: "dict | None" = None) -> None:
        ch = self._ch_conf
        if not _MPL_AVAILABLE:
            return
        ch.clear()
        ax = ch.ax

        if history and history.get("confidence_counts"):
            conf    = history["confidence_counts"]
            n_alta  = conf.get("alta",    0)
            n_ambig = conf.get("ambiguo", 0)
            n_baja  = conf.get("baja",    0)
        else:
            n_alta  = sum(
                1 for r in records
                if not getattr(r["event"], "ambiguous", False)
                and getattr(r["event"], "confidence_level", "") == "alta"
            )
            n_ambig = sum(1 for r in records if getattr(r["event"], "ambiguous", False))
            n_baja  = sum(
                1 for r in records
                if not getattr(r["event"], "ambiguous", False)
                and getattr(r["event"], "confidence_level", "") == "baja"
            )

        total = n_alta + n_baja + n_ambig
        if total == 0:
            ch._no_data("Sin registros")
            return

        data = [
            ("Alta",    n_alta,  SUCCESS),
            ("Baja",    n_baja,  ERROR),
            ("Ambiguo", n_ambig, WARNING),
        ]
        labels = [f"{lbl}\n{cnt}" for lbl, cnt, _ in data if cnt > 0]
        sizes  = [cnt for _, cnt, _ in data if cnt > 0]
        colors = [col for _, cnt, col in data if cnt > 0]

        ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            startangle=90,
            wedgeprops={"width": 0.45, "edgecolor": "#0d0d0d", "linewidth": 1.5},
            textprops={"fontsize": 7, "color": TEXT_PRIMARY},
        )
        ax.set_title("Confianza", fontsize=7, color=ACCENT, pad=4)
        ch.draw()

    def _draw_runs_by_date(self, history: "dict | None" = None) -> None:
        ch = self._ch_status
        if not _MPL_AVAILABLE:
            return
        ch.clear()
        ax = ch.ax

        if not history:
            ch._no_data("Sin historial de corridas")
            return

        runs = history.get("runs", [])
        if not runs:
            ch._no_data("Sin corridas registradas")
            return

        date_counts: dict[str, int] = {}
        for run in runs:
            ts = run.get("timestamp", "")
            if ts:
                date_key = ts[:10]  # YYYY-MM-DD
                date_counts[date_key] = date_counts.get(date_key, 0) + 1

        if not date_counts:
            ch._no_data("Sin corridas registradas")
            return

        dates  = sorted(date_counts.keys())
        counts = [date_counts[d] for d in dates]
        labels = [f"{d[8:10]}/{d[5:7]}" for d in dates]  # DD/MM

        bars = ax.bar(labels, counts, color=ACCENT, alpha=0.85, width=0.5)
        ax.set_ylabel("Corridas", fontsize=6, color="#4a5248")
        ax.tick_params(axis="x", labelsize=6, colors=TEXT_PRIMARY, rotation=30)
        ax.tick_params(axis="y", labelsize=6, colors="#4a5248")
        ax.yaxis.set_major_locator(_MaxNLocator(integer=True))
        ax.set_title("Corridas por fecha", fontsize=7, color=ACCENT, pad=4)

        for bar, val in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                str(val),
                ha="center", va="bottom",
                fontsize=7, color=TEXT_PRIMARY,
            )
        ch.draw()


# ---------------------------------------------------------------------------
# Estado vacío
# ---------------------------------------------------------------------------

class _EmptyState(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(160)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(14)

        icon = QLabel("◈")
        icon.setStyleSheet(
            f"color: {NEUTRAL}; font-size: 44px; background: transparent;"
        )
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)

        msg = QLabel("Sin datos — procesá archivos en la pestaña Análisis")
        msg.setStyleSheet(
            f"color: {NEUTRAL}; font-size: 14px; font-style: italic;"
            " background: transparent;"
        )
        msg.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(icon)
        layout.addWidget(msg)


# ---------------------------------------------------------------------------
# Pestaña principal
# ---------------------------------------------------------------------------

class EvaluacionTab(QWidget):
    """
    Pestaña Evaluación completa de SAREKO.

    API pública:
        update_from_session(records, batch_summary, history=None)
            — recibe los datos de la sesión y actualiza todos los indicadores.

    Señales:
        validation_changed — emitida al guardar una validación desde el panel lateral.
    """

    validation_changed = Signal()

    def __init__(self, species_catalog: "list | None" = None, parent=None):
        super().__init__(parent)
        self._species_catalog  = species_catalog or []
        self._current_records: list[dict] = []
        self._panel_open    = False
        self._panel_row_idx = -1

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(32, 24, 32, 32)
        outer.setSpacing(16)

        # ── Header — dos cards lado a lado ────────────────────────────────
        header_row = QHBoxLayout()
        header_row.setSpacing(16)
        self._header_card  = _EvalHeaderCard()
        self._summary_card = _GlobalSummaryCard()
        header_row.addWidget(self._header_card,  3)
        header_row.addWidget(self._summary_card, 2)
        outer.addLayout(header_row)

        # ── Fila cuerpo: columna de contenido + panel lateral ─────────────
        body_row = QHBoxLayout()
        body_row.setSpacing(12)
        body_row.setContentsMargins(0, 0, 0, 0)

        # Wrapper de la columna de contenido (necesario para QHBoxLayout)
        body_widget = QWidget()
        body_widget.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        body_widget.setStyleSheet("background: transparent;")
        body_col = QVBoxLayout(body_widget)
        body_col.setContentsMargins(0, 0, 0, 0)
        body_col.setSpacing(16)

        # Cards de contenido
        self._latency_card = _LatencyCard()
        self._error_card   = _ErrorEvalCard()
        self._charts_card  = _ChartsCard()
        self._multi_card   = _MultispeciesCard()
        self._empty_state  = _EmptyState()

        body_col.addWidget(self._latency_card)
        body_col.addWidget(self._error_card)
        body_col.addWidget(self._multi_card)
        body_col.addWidget(self._charts_card)
        body_col.addWidget(self._empty_state)
        body_col.addStretch()

        # Panel lateral deslizable (reutilizado de ValidacionTab)
        self._panel = _SidePanel(self._species_catalog)

        body_row.addWidget(body_widget, 1)
        body_row.addWidget(self._panel)
        outer.addLayout(body_row)

        # ── Estado inicial (sin datos) ────────────────────────────────────
        self._latency_card.hide()
        self._error_card.hide()
        self._charts_card.hide()
        self._multi_card.hide()

        # ── Conexiones de señales ─────────────────────────────────────────
        self._error_card.detail_requested.connect(self._on_error_detail_requested)
        self._multi_card.detail_requested.connect(self._open_panel)
        self._panel.closed.connect(self._on_panel_closed)
        self._panel.validation_saved.connect(self._on_panel_validation_saved)
        self._panel.multi_species_changed.connect(self._on_multi_species_changed)

    # ------------------------------------------------------------------
    # Gestión del panel lateral

    def _open_panel(self, record: dict, global_idx: int = -1) -> None:
        self._panel_row_idx = global_idx
        self._panel.load_record(record, global_idx)
        if not self._panel_open:
            self._panel.open_panel()
            self._panel_open = True

    def _on_panel_closed(self) -> None:
        self._panel_open    = False
        self._panel_row_idx = -1

    def _on_panel_validation_saved(self, row_idx: int, category: str, species: str) -> None:
        """El panel guardó una validación — propagar hacia MainWindow."""
        self.validation_changed.emit()

    def _on_multi_species_changed(self, global_idx: int, species_list: list) -> None:
        """El panel modificó las especies adicionales — refrescar card multiespecie."""
        self._multi_card.update_data(self._current_records)
        self.validation_changed.emit()

    def _on_error_detail_requested(self, record: dict) -> None:
        """'Ver detalle' en la tabla de errores — buscar índice global y abrir panel."""
        try:
            global_idx = self._current_records.index(record)
        except ValueError:
            global_idx = -1
        self._open_panel(record, global_idx)

    # ------------------------------------------------------------------
    # API pública

    def update_from_session(
        self,
        records: list[dict],
        batch_summary: dict,
        history: "dict | None" = None,
    ) -> None:
        """Actualiza todos los indicadores, cards y gráficos."""
        self._current_records = records

        if history:
            n_corridas     = history.get("total_runs",        0)
            n_registros    = history.get("total_records",     0)
            n_validaciones = history.get("total_validations", 0)
        else:
            n_corridas     = len(batch_summary.get("files", []))
            n_registros    = len(records)
            n_validaciones = sum(
                1 for r in records
                if r.get("validation", {}).get("state") == "validated"
            )

        session_records = len(records)
        session_files   = len(batch_summary.get("files", []))
        has_data = (
            n_registros > 0 or n_corridas > 0
            or session_records > 0 or session_files > 0
        )

        self._empty_state.setVisible(not has_data)
        self._latency_card.setVisible(has_data)
        self._error_card.setVisible(has_data)
        self._charts_card.setVisible(has_data)
        self._multi_card.setVisible(has_data)

        if not has_data:
            self._summary_card.reset()
            return

        self._summary_card.update_metrics(n_corridas, n_registros, n_validaciones)
        latency_recs = history.get("latency_records", []) if history else []
        self._latency_card.update_data(latency_recs)
        self._error_card.update_data(records)
        self._charts_card.update_data(records, batch_summary, history)
        self._multi_card.update_data(records)

    def reset(self) -> None:
        """Deja la pestaña en el mismo estado que al abrir la app por primera vez."""
        self.update_from_session([], {}, None)
