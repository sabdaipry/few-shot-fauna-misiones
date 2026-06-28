"""
Pestaña Análisis de SAREKO.

Estructura:
    AnalisisTab
      ├── _HeroCard              — breadcrumb, título, 5 métricas rápidas
      ├── _CargaCard             — selector de archivos, modo, botón Procesar
      ├── _SeguimientoCard       — métricas RT, barra progreso, log de etapas
      └── _BatchCard             — tabla de lote con panel de detalle expandible
"""

import csv
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

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
from src.workers.processing_worker import ProcessingWorker

VIDEO_EXTS = {"*.mp4", "*.avi", "*.mov"}
IMAGE_EXTS = {"*.jpg", "*.jpeg", "*.png"}
ALL_FILTER  = "Archivos soportados (*.mp4 *.avi *.mov *.jpg *.jpeg *.png)"

_MODES = [
    ("Básico",    60, "Rápido, menor cobertura temporal."),
    ("Estándar",  30, "Balance velocidad / detalle."),
    ("Profundo",  10, "Lento, máxima cobertura temporal."),
]

_STAGES = [
    "Extracción de frames",
    "Generación de embeddings (BioCLIP)",
    "Clasificación por similitud coseno",
    "Árbitro KNN (si aplica)",
    "Consenso temporal",
    "Escritura de resultados",
]

_BADGE_COLORS = {
    "en cola":    NEUTRAL,
    "procesando": WARNING,
    "completado": SUCCESS,
    "error":      ERROR,
}


def _sep() -> QFrame:
    sep = QFrame()
    sep.setFrameShape(QFrame.Shape.HLine)
    sep.setStyleSheet("border: none; border-top: 1px solid rgba(153,225,122,50);")
    sep.setFixedHeight(1)
    return sep


def _progress_bar(height: int = 8) -> QProgressBar:
    pb = QProgressBar()
    pb.setRange(0, 100)
    pb.setValue(0)
    pb.setTextVisible(False)
    pb.setFixedHeight(height)
    pb.setStyleSheet(f"""
        QProgressBar {{
            background: rgba(255,255,255,25);
            border-radius: {height // 2}px;
            border: none;
        }}
        QProgressBar::chunk {{
            background: {ACCENT};
            border-radius: {height // 2}px;
        }}
    """)
    return pb


def _fmt_time(sec: float) -> str:
    """Formatea segundos a 'M:SS'."""
    s = max(0, int(sec))
    return f"{s // 60}:{s % 60:02d}"


# ---------------------------------------------------------------------------
# Card Hero
# ---------------------------------------------------------------------------

class _HeroCard(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("herocard")
        self.setStyleSheet(card_qss("herocard"))

        outer = QHBoxLayout(self)
        outer.setContentsMargins(36, 28, 36, 28)
        outer.setSpacing(48)

        # Columna izquierda
        left = QVBoxLayout()
        left.setSpacing(8)
        left.setContentsMargins(0, 0, 0, 0)

        bc = QLabel("SAREKO / ANÁLISIS")
        bc.setStyleSheet(section_label_qss())

        title = QLabel("Análisis automatizado de cámaras trampa")
        title.setStyleSheet(title_qss(32))

        desc = QLabel(
            "Cargá videos o imágenes de cámaras trampa para identificar "
            "automáticamente las especies presentes. El procesamiento corre "
            "en segundo plano usando BioCLIP v2 como extractor de características."
        )
        desc.setStyleSheet(body_qss(0.7))
        desc.setWordWrap(True)

        left.addWidget(bc)
        left.addWidget(title)
        left.addSpacing(4)
        left.addWidget(desc)
        left.addStretch()

        # Columna derecha — 5 métricas rápidas
        right = QGridLayout()
        right.setSpacing(16)
        right.setContentsMargins(0, 0, 0, 0)

        self._badge_estado = QLabel("sin corrida activa")
        self._badge_estado.setStyleSheet(badge_qss(NEUTRAL))
        self._badge_estado.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._badge_estado.setFixedHeight(22)

        self._lbl_especies   = self._metric_value("—")
        self._lbl_priorit    = self._metric_value("—")
        self._lbl_alertas    = self._metric_value("—")
        self._lbl_frames     = self._metric_value("—")

        right.addWidget(self._metric_label("Análisis activo"),        0, 0)
        right.addWidget(self._badge_estado,                            0, 1)
        right.addWidget(self._metric_label("Especies detectadas"),    1, 0)
        right.addWidget(self._lbl_especies,                            1, 1)
        right.addWidget(self._metric_label("Especies prioritarias"),  2, 0)
        right.addWidget(self._lbl_priorit,                             2, 1)
        right.addWidget(self._metric_label("Alertas activas"),        3, 0)
        right.addWidget(self._lbl_alertas,                             3, 1)
        right.addWidget(self._metric_label("Frames analizados"),      4, 0)
        right.addWidget(self._lbl_frames,                              4, 1)
        right.setColumnStretch(1, 1)

        outer.addLayout(left, 3)
        outer.addLayout(right, 2)

    def _metric_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(section_label_qss())
        return lbl

    def _metric_value(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 18px; font-weight: 700;"
            " background: transparent;"
        )
        lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return lbl

    def set_active(self, estado: str) -> None:
        color = _BADGE_COLORS.get(estado, NEUTRAL)
        self._badge_estado.setText(estado)
        self._badge_estado.setStyleSheet(badge_qss(color))

    def set_especies(self, n: int) -> None:
        self._lbl_especies.setText(str(n))

    def set_frames(self, n: int) -> None:
        self._lbl_frames.setText(str(n))

    def reset(self) -> None:
        self.set_active("sin corrida activa")
        self._badge_estado.setStyleSheet(badge_qss(NEUTRAL))
        self._lbl_especies.setText("—")
        self._lbl_priorit.setText("—")
        self._lbl_alertas.setText("—")
        self._lbl_frames.setText("—")


# ---------------------------------------------------------------------------
# Card Carga
# ---------------------------------------------------------------------------

class _CargaCard(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("cargacard")
        self.setStyleSheet(card_qss("cargacard"))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self._selected_files: list[Path] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(14)

        sec = QLabel("CARGA Y EJECUCIÓN")
        sec.setStyleSheet(section_label_qss())
        layout.addWidget(sec)
        layout.addWidget(_sep())

        # Botones de selección
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self._btn_files  = self._action_btn("Seleccionar archivos")
        self._btn_folder = self._action_btn("Seleccionar carpeta")
        btn_row.addWidget(self._btn_files)
        btn_row.addWidget(self._btn_folder)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._lbl_files = QLabel("Sin archivos seleccionados")
        self._lbl_files.setStyleSheet(body_qss(0.55))
        layout.addWidget(self._lbl_files)

        layout.addWidget(_sep())

        # Modo de análisis
        lbl_modo = QLabel("MODO DE ANÁLISIS")
        lbl_modo.setStyleSheet(section_label_qss())
        layout.addWidget(lbl_modo)

        self._combo = QComboBox()
        self._combo.setStyleSheet(f"""
            QComboBox {{
                background: rgba(0,0,0,120);
                color: {TEXT_PRIMARY};
                border: 1px solid rgba(153,225,122,80);
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 13px;
            }}
            QComboBox::drop-down {{ border: none; }}
            QComboBox QAbstractItemView {{
                background: #1a1a1a;
                color: {TEXT_PRIMARY};
                selection-background-color: #1f2c1d;
                border: 1px solid rgba(153,225,122,80);
            }}
        """)
        for name, _, _ in _MODES:
            self._combo.addItem(name)
        self._combo.setCurrentIndex(1)  # Estándar por defecto
        layout.addWidget(self._combo)

        self._lbl_desc = QLabel(_MODES[1][2])
        self._lbl_desc.setStyleSheet(body_qss(0.55))
        self._lbl_desc.setWordWrap(True)
        layout.addWidget(self._lbl_desc)

        layout.addWidget(_sep())

        # Botón Procesar
        self._btn_procesar = QPushButton("Procesar")
        self._btn_procesar.setEnabled(False)
        self._btn_procesar.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_procesar.setFixedHeight(40)
        self._btn_procesar.setStyleSheet(self._procesar_qss(False))
        layout.addWidget(self._btn_procesar)

        note = QLabel("El procesamiento corre en segundo plano.")
        note.setStyleSheet(
            f"color: {ACCENT}; font-size: 11px; font-style: italic; background: transparent;"
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        # Conexiones internas
        self._btn_files.clicked.connect(self._pick_files)
        self._btn_folder.clicked.connect(self._pick_folder)
        self._combo.currentIndexChanged.connect(self._on_mode_changed)

    # ------------------------------------------------------------------

    def _action_btn(self, label: str) -> QPushButton:
        btn = QPushButton(label)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setFixedHeight(34)
        btn.setStyleSheet(f"""
            QPushButton {{
                background: rgba(153,225,122,30);
                color: {TEXT_PRIMARY};
                border: 1px solid rgba(153,225,122,100);
                border-radius: 6px;
                font-size: 12px;
                font-weight: 600;
                padding: 0 16px;
            }}
            QPushButton:hover {{ background: rgba(153,225,122,50); }}
        """)
        return btn

    def _procesar_qss(self, enabled: bool) -> str:
        if enabled:
            return f"""
                QPushButton {{
                    background: {ACCENT};
                    color: #0a1a08;
                    border: none;
                    border-radius: 8px;
                    font-size: 14px;
                    font-weight: 700;
                }}
                QPushButton:hover {{ background: #b0f08a; }}
            """
        return f"""
            QPushButton {{
                background: rgba(153,225,122,20);
                color: rgba(237,239,236,80);
                border: 1px solid rgba(153,225,122,40);
                border-radius: 8px;
                font-size: 14px;
                font-weight: 700;
            }}
        """

    def _pick_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Seleccionar archivos", "",
            "Archivos soportados (*.mp4 *.avi *.mov *.jpg *.jpeg *.png)"
        )
        if paths:
            self._selected_files = [Path(p) for p in paths]
            self._update_file_label()

    def _pick_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta")
        if folder:
            p = Path(folder)
            exts = {".mp4", ".avi", ".mov", ".jpg", ".jpeg", ".png"}
            self._selected_files = [f for f in p.iterdir() if f.suffix.lower() in exts]
            self._update_file_label()

    def _update_file_label(self) -> None:
        n = len(self._selected_files)
        if n == 0:
            self._lbl_files.setText("Sin archivos seleccionados")
        elif n == 1:
            self._lbl_files.setText(f"1 archivo seleccionado: {self._selected_files[0].name}")
        else:
            self._lbl_files.setText(f"{n} archivos seleccionados")
        self._btn_procesar.setEnabled(n > 0)
        self._btn_procesar.setStyleSheet(self._procesar_qss(n > 0))

    def _on_mode_changed(self, idx: int) -> None:
        self._lbl_desc.setText(_MODES[idx][2])

    # ------------------------------------------------------------------
    # API pública

    @property
    def selected_files(self) -> list[Path]:
        return self._selected_files

    @property
    def mode_N(self) -> int:
        return _MODES[self._combo.currentIndex()][1]

    @property
    def btn_procesar(self) -> QPushButton:
        return self._btn_procesar

    def set_processing(self, active: bool) -> None:
        self._btn_files.setEnabled(not active)
        self._btn_folder.setEnabled(not active)
        self._combo.setEnabled(not active)
        if active:
            self._btn_procesar.setText("Procesando…")
            self._btn_procesar.setEnabled(False)
            self._btn_procesar.setStyleSheet(self._procesar_qss(False))
        else:
            self._btn_procesar.setText("Procesar")
            n = len(self._selected_files)
            self._btn_procesar.setEnabled(n > 0)
            self._btn_procesar.setStyleSheet(self._procesar_qss(n > 0))


# ---------------------------------------------------------------------------
# Card Seguimiento
# ---------------------------------------------------------------------------

class _SeguimientoCard(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("segucard")
        self.setStyleSheet(card_qss("segucard"))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(12)

        sec = QLabel("ESTADO DEL ANÁLISIS")
        sec.setStyleSheet(section_label_qss())
        layout.addWidget(sec)
        layout.addWidget(_sep())

        # Grid de métricas en tiempo real
        grid = QGridLayout()
        grid.setSpacing(8)
        grid.setContentsMargins(0, 0, 0, 0)

        labels = [
            "Tipo", "Progreso total",
            "Archivo actual", "Estado",
            "Tiempo BioCLIP (s)", "Tiempo total (s)",
            "Frames (archivo)", "Frames (acumulado)",
        ]
        self._metric_vals: list[QLabel] = []
        for i, lbl_text in enumerate(labels):
            row, col = divmod(i, 2)
            lbl = QLabel(lbl_text)
            lbl.setStyleSheet(section_label_qss())
            grid.addWidget(lbl, row, col * 2)

            val = QLabel("—")
            val.setStyleSheet(
                f"color: {TEXT_PRIMARY}; font-size: 13px; font-weight: 600;"
                " background: transparent;"
            )
            grid.addWidget(val, row, col * 2 + 1)
            self._metric_vals.append(val)

        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        layout.addLayout(grid)

        # Badge de estado
        self._badge = QLabel("en espera")
        self._badge.setStyleSheet(badge_qss(NEUTRAL))
        self._badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._badge.setFixedHeight(22)
        self._badge.setMaximumWidth(130)
        layout.addWidget(self._badge)

        layout.addWidget(_sep())

        # Barra de progreso general
        lbl_prog = QLabel("PROGRESO GENERAL")
        lbl_prog.setStyleSheet(section_label_qss())
        layout.addWidget(lbl_prog)

        self._progress_bar = _progress_bar(12)
        layout.addWidget(self._progress_bar)

        self._lbl_pct = QLabel("0 %")
        self._lbl_pct.setStyleSheet(body_qss(0.6))
        self._lbl_pct.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self._lbl_pct)

        layout.addWidget(_sep())

        # Log de etapas
        lbl_log = QLabel("ETAPAS DEL PIPELINE")
        lbl_log.setStyleSheet(section_label_qss())
        layout.addWidget(lbl_log)

        self._stage_bars:   dict[str, QProgressBar] = {}
        self._stage_labels: dict[str, QLabel]       = {}

        for stage in _STAGES:
            row_w = QWidget()
            row_w.setStyleSheet("background: transparent;")
            row_l = QHBoxLayout(row_w)
            row_l.setContentsMargins(0, 0, 0, 0)
            row_l.setSpacing(8)

            name = QLabel(stage)
            name.setStyleSheet(body_qss(0.65))
            name.setFixedWidth(240)

            pb = _progress_bar(6)
            pct_lbl = QLabel("0 %")
            pct_lbl.setStyleSheet(body_qss(0.5))
            pct_lbl.setFixedWidth(36)
            pct_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

            row_l.addWidget(name)
            row_l.addWidget(pb, 1)
            row_l.addWidget(pct_lbl)
            layout.addWidget(row_w)

            self._stage_bars[stage]   = pb
            self._stage_labels[stage] = pct_lbl

    # ------------------------------------------------------------------
    # API pública

    def update_progress(self, pct: int) -> None:
        self._progress_bar.setValue(pct)
        self._lbl_pct.setText(f"{pct} %")

    def update_stage(self, stage: str, pct: int) -> None:
        # Búsqueda parcial para tolerar variaciones de nombre
        for key in self._stage_bars:
            if key.lower() in stage.lower() or stage.lower() in key.lower():
                self._stage_bars[key].setValue(pct)
                self._stage_labels[key].setText(f"{pct} %")
                break

    def update_metrics(self, data: dict) -> None:
        vals = [
            data.get("tipo", "—"),
            f"{data.get('progreso', 0)} %",
            data.get("archivo", "—"),
            "",  # estado va en badge
            str(data.get("tiempo_bioclip", "—")),
            str(data.get("tiempo_total", "—")),
            str(data.get("frames_archivo", "—")),
            str(data.get("frames_acumulados", "—")),
        ]
        for i, v in enumerate(vals):
            if i != 3:
                self._metric_vals[i].setText(v)

        estado = data.get("estado", "en espera")
        color = _BADGE_COLORS.get(estado, NEUTRAL)
        self._badge.setText(estado)
        self._badge.setStyleSheet(badge_qss(color))

    def reset(self) -> None:
        self._progress_bar.setValue(0)
        self._lbl_pct.setText("0 %")
        self._badge.setText("en espera")
        self._badge.setStyleSheet(badge_qss(NEUTRAL))
        for v in self._metric_vals:
            v.setText("—")
        for pb in self._stage_bars.values():
            pb.setValue(0)
        for lbl in self._stage_labels.values():
            lbl.setText("0 %")


# ---------------------------------------------------------------------------
# Panel de detalle expandible (no modal)
# ---------------------------------------------------------------------------

class _DetailPanel(QFrame):
    """Muestra info de un archivo al hacer click en 'Ver detalles'."""

    closed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("detailpanel")
        self.setStyleSheet("""
            QFrame#detailpanel {
                background: rgba(0,0,0,140);
                border: 1px solid rgba(153,225,122,60);
                border-radius: 10px;
            }
        """)
        self.hide()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 14, 20, 14)
        layout.setSpacing(10)

        # Fila de título + botón cerrar
        hrow = QHBoxLayout()
        self._title = QLabel()
        self._title.setStyleSheet(section_label_qss())
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
        btn_close.clicked.connect(self._on_close)
        hrow.addWidget(self._title)
        hrow.addStretch()
        hrow.addWidget(btn_close)
        layout.addLayout(hrow)

        # Grid de contenido dinámico
        self._grid = QGridLayout()
        self._grid.setContentsMargins(0, 4, 0, 0)
        self._grid.setSpacing(8)
        self._grid.setColumnStretch(1, 1)
        layout.addLayout(self._grid)

    def _on_close(self) -> None:
        self.hide()
        self.closed.emit()

    def _clear(self) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def _add_row(self, r: int, label: str, value: str, italic: bool = False) -> None:
        lbl = QLabel(label)
        lbl.setStyleSheet(section_label_qss())
        val = QLabel(value)
        val.setWordWrap(True)
        val.setStyleSheet(
            f"color: rgba(237,239,236,216); font-size: 13px; background: transparent;"
            + (" font-style: italic;" if italic else "")
        )
        self._grid.addWidget(lbl, r, 0)
        self._grid.addWidget(val, r, 1)

    def show_error(self, name: str, stage: str, msg: str, pct: int) -> None:
        self._clear()
        self._title.setText("DETALLE DEL ERROR")
        self._add_row(0, "ARCHIVO",             name)
        self._add_row(1, "ETAPA",               stage)
        self._add_row(2, "ERROR",               msg)
        self._add_row(3, "PROGRESO AL FALLAR",  f"{pct} %")
        self.show()

    def show_completed(self, name: str, events: list, meta: dict | None = None) -> None:
        self._clear()
        self._title.setText("DETALLE DE LA CORRIDA")

        species: set[str] = set()
        n_alta = n_baja = 0
        dists: list[float] = []
        for ev in events:
            sp = getattr(ev, "species", "")
            if sp:
                species.add(sp)
            if getattr(ev, "confidence_level", "") == "alta":
                n_alta += 1
            else:
                n_baja += 1
            d = getattr(ev, "cosine_distance", None)
            if d is not None:
                dists.append(float(d))

        sp_str   = ", ".join(sorted(species)) if species else "—"
        conf_str = f"{n_alta} alta / {n_baja} baja"
        avg_str  = f"{sum(dists) / len(dists):.4f}" if dists else "—"

        self._add_row(0, "ARCHIVO",                 name)
        self._add_row(1, "ESPECIES ENCONTRADAS",    sp_str, italic=True)
        self._add_row(2, "EVENTOS DETECTADOS",      str(len(events)))
        self._add_row(3, "CONFIANZA (alta / baja)", conf_str)
        self._add_row(4, "DISTANCIA COSENO PROM.",  avg_str)

        r = 5
        if meta:
            dur  = meta.get("duration_sec")
            proc = meta.get("processing_sec")
            if dur is not None:
                self._add_row(r, "DURACIÓN ARCHIVO", _fmt_time(dur))
                r += 1
            if proc is not None:
                self._add_row(r, "TIEMPO PROCESAMIENTO", _fmt_time(proc))
                r += 1
            if dur is not None and proc is not None and dur > 0:
                self._add_row(r, "FACTOR DE PROCESAMIENTO", f"{proc / dur:.1f}×")

        self.show()


# ---------------------------------------------------------------------------
# Card Batch — tabla de resumen del lote
# ---------------------------------------------------------------------------

class _BatchCard(QFrame):
    """Card de ancho completo con tabla de lote y panel de detalle expandible."""

    resume_requested = Signal(str)  # nombre del archivo a reanudar

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("batchcard")
        self.setStyleSheet(card_qss("batchcard"))

        self._rows: dict[str, dict] = {}   # name → dict con widgets y estado
        self._current_file:  str = ""      # archivo en procesamiento activo
        self._current_stage: str = ""      # última etapa recibida por señal
        self._detail_target: str = ""      # archivo cuyo detalle está abierto

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(14)

        # ── Header ──────────────────────────────────────────────────────
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)

        sec = QLabel("BATCH OPERATIVO")
        sec.setStyleSheet(section_label_qss())
        header.addWidget(sec)
        header.addStretch()

        self._btn_csv  = self._mk_export_btn("Exportar CSV")
        self._btn_xlsx = self._mk_export_btn("Exportar XLSX")
        self._btn_csv.setEnabled(False)
        self._btn_xlsx.setEnabled(False)
        self._btn_csv.clicked.connect(self._export_csv)
        self._btn_xlsx.clicked.connect(self._export_xlsx)
        header.addWidget(self._btn_csv)
        header.addSpacing(6)
        header.addWidget(self._btn_xlsx)
        layout.addLayout(header)
        layout.addWidget(_sep())

        # ── Tabla ────────────────────────────────────────────────────────
        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels(
            ["ARCHIVO", "ESTADO", "PROGRESO", "ESPECIES", "DETALLE", "ACCIONES"]
        )
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(1, 110)
        self._table.setColumnWidth(2, 150)
        self._table.setColumnWidth(3, 80)
        self._table.setColumnWidth(5, 200)
        self._table.verticalHeader().setVisible(False)
        self._table.setShowGrid(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._table.setCornerButtonEnabled(False)
        self._table.setStyleSheet(self._table_qss())
        self._table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._table.setMinimumHeight(100)
        layout.addWidget(self._table, 1)

        # ── Panel de detalle ─────────────────────────────────────────────
        self._detail = _DetailPanel()
        self._detail.closed.connect(self._on_detail_closed)
        layout.addWidget(self._detail)

    # ── Constructores de widgets de celda ────────────────────────────────

    def _mk_export_btn(self, label: str) -> QPushButton:
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
            QPushButton:hover  {{ background: rgba(153,225,122,40); }}
            QPushButton:disabled {{
                background: rgba(74,82,72,30);
                color: rgba(237,239,236,60);
                border-color: rgba(74,82,72,60);
            }}
        """)
        return btn

    def _table_qss(self) -> str:
        return f"""
            QTableWidget {{
                background: transparent;
                border: none;
                outline: none;
                color: {TEXT_PRIMARY};
                font-size: 13px;
            }}
            QTableWidget::item {{
                padding: 6px 10px;
                border-bottom: 1px solid rgba(255,255,255,15);
                background: transparent;
            }}
            QTableWidget::item:hover    {{ background: rgba(153,225,122,12); }}
            QTableWidget::item:selected {{ background: rgba(31,44,29,160); }}
            QHeaderView::section {{
                background: transparent;
                color: {ACCENT};
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 1px;
                border: none;
                border-bottom: 1px solid rgba(153,225,122,89);
                padding: 4px 10px;
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
        """

    def _mk_badge_cell(self, state: str) -> tuple[QWidget, QLabel]:
        container = QWidget()
        container.setStyleSheet("background: transparent;")
        hl = QHBoxLayout(container)
        hl.setContentsMargins(8, 4, 8, 4)
        hl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl = QLabel(state)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setFixedHeight(22)
        lbl.setStyleSheet(badge_qss(_BADGE_COLORS.get(state, NEUTRAL)))
        hl.addWidget(lbl)
        return container, lbl

    def _mk_progress_cell(self) -> tuple[QWidget, QProgressBar, QLabel]:
        w = QWidget()
        w.setStyleSheet("background: transparent;")
        hl = QHBoxLayout(w)
        hl.setContentsMargins(8, 0, 8, 0)
        hl.setSpacing(6)
        pb = _progress_bar(6)
        pct_lbl = QLabel("0 %")
        pct_lbl.setFixedWidth(32)
        pct_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        pct_lbl.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 11px; background: transparent;"
        )
        hl.addWidget(pb, 1)
        hl.addWidget(pct_lbl)
        return w, pb, pct_lbl

    def _mk_actions_cell(self, name: str) -> tuple[QWidget, QPushButton]:
        w = QWidget()
        w.setStyleSheet("background: transparent;")
        hl = QHBoxLayout(w)
        hl.setContentsMargins(4, 2, 4, 2)
        hl.setSpacing(6)

        btn_ver = QPushButton("Ver detalles")
        btn_ver.setFixedHeight(26)
        btn_ver.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_ver.setStyleSheet(f"""
            QPushButton {{
                background: rgba(153,225,122,20);
                color: {TEXT_PRIMARY};
                border: 1px solid rgba(153,225,122,80);
                border-radius: 5px;
                font-size: 11px;
                font-weight: 600;
                padding: 0 10px;
            }}
            QPushButton:hover {{ background: rgba(153,225,122,40); }}
            QPushButton:disabled {{
                background: rgba(74,82,72,20);
                color: rgba(237,239,236,40);
                border-color: rgba(74,82,72,50);
            }}
        """)
        btn_ver.clicked.connect(lambda: self._toggle_detail(name))

        btn_resume = QPushButton("Reanudar")
        btn_resume.setFixedHeight(26)
        btn_resume.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_resume.setVisible(False)
        btn_resume.setStyleSheet(f"""
            QPushButton {{
                background: rgba(224,92,92,25);
                color: {TEXT_PRIMARY};
                border: 1px solid rgba(224,92,92,100);
                border-radius: 5px;
                font-size: 11px;
                font-weight: 600;
                padding: 0 10px;
            }}
            QPushButton:hover {{ background: rgba(224,92,92,50); }}
        """)
        btn_resume.clicked.connect(lambda: self.resume_requested.emit(name))

        hl.addWidget(btn_ver)
        hl.addWidget(btn_resume)
        hl.addStretch()
        return w, btn_resume, btn_ver

    # ── API pública ──────────────────────────────────────────────────────

    def populate(self, files: list) -> None:
        """Inicializa la tabla con todos los archivos en estado 'en cola'."""
        self._rows.clear()
        self._current_file  = ""
        self._current_stage = ""
        self._detail_target = ""
        self._detail.hide()
        self._table.setRowCount(0)
        self._btn_csv.setEnabled(False)
        self._btn_xlsx.setEnabled(False)

        for file_path in files:
            name = file_path.name if hasattr(file_path, "name") else str(file_path)
            r = self._table.rowCount()
            self._table.insertRow(r)
            self._table.setRowHeight(r, 44)

            # Col 0 — Archivo
            self._table.setItem(r, 0, QTableWidgetItem(name))

            # Col 1 — Estado
            badge_w, badge_lbl = self._mk_badge_cell("en cola")
            self._table.setCellWidget(r, 1, badge_w)

            # Col 2 — Progreso
            prog_w, prog_bar, prog_lbl = self._mk_progress_cell()
            self._table.setCellWidget(r, 2, prog_w)

            # Col 3 — Especies
            it3 = QTableWidgetItem("—")
            it3.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(r, 3, it3)

            # Col 4 — Detalle
            self._table.setItem(r, 4, QTableWidgetItem("—"))

            # Col 5 — Acciones
            act_w, btn_resume, btn_ver = self._mk_actions_cell(name)
            btn_ver.setEnabled(False)
            self._table.setCellWidget(r, 5, act_w)

            self._rows[name] = {
                "row_idx":     r,
                "badge_lbl":   badge_lbl,
                "prog_bar":    prog_bar,
                "prog_lbl":    prog_lbl,
                "btn_resume":  btn_resume,
                "btn_ver":     btn_ver,
                "state":       "en cola",
                "pct":         0,
                "events":      [],
                "meta":        None,
                "error_msg":   "",
                "error_stage": "",
                "path":        file_path,
            }

    def on_file_started(self, name: str) -> None:
        self._current_file = name
        if name not in self._rows:
            return
        row = self._rows[name]
        row["state"] = "procesando"
        row["pct"]   = 0
        self._update_badge(name, "procesando")
        self._update_progress(name, 0)
        row["btn_ver"].setEnabled(False)

    def on_stage_updated(self, stage: str, pct: int) -> None:
        self._current_stage = stage
        if not self._current_file or self._current_file not in self._rows:
            return
        # La etapa de embeddings es la más larga; usarla como proxy del progreso por archivo
        if "embeddings" in stage.lower():
            self._update_progress(self._current_file, pct)
            self._rows[self._current_file]["pct"] = pct

    def on_file_completed(self, name: str, events: list, meta: dict | None = None) -> None:
        if name not in self._rows:
            return
        row = self._rows[name]
        row["state"]  = "completado"
        row["events"] = events
        row["meta"]   = meta
        row["pct"]    = 100
        self._update_badge(name, "completado")
        self._update_progress(name, 100)
        row["btn_ver"].setEnabled(True)

        # Columna Especies: cuenta de especies únicas
        species = sorted({
            getattr(ev, "species", "")
            for ev in events if getattr(ev, "species", "")
        })
        self._table.item(row["row_idx"], 3).setText(
            str(len(species)) if species else "—"
        )

        # Columna Detalle: rangos de tiempo (BiologicalEvent) o descripción breve
        intervals = [
            f"{_fmt_time(ev.start_time)}–{_fmt_time(ev.end_time)}"
            for ev in events if hasattr(ev, "start_time")
        ]
        det = (
            ", ".join(intervals) if intervals
            else ("1 imagen procesada" if events else "sin detecciones")
        )
        self._table.item(row["row_idx"], 4).setText(det)

        self._check_export()
        if self._detail_target == name:
            self._detail.show_completed(name, events, meta)

    def on_file_error(self, name: str, msg: str) -> None:
        if name not in self._rows:
            return
        row = self._rows[name]
        row["state"]       = "error"
        row["error_msg"]   = msg
        row["error_stage"] = self._current_stage or "desconocida"
        self._update_badge(name, "error")
        row["btn_ver"].setEnabled(True)
        row["btn_resume"].setVisible(True)
        self._table.item(row["row_idx"], 4).setText(f"Error: {msg[:60]}")
        if self._detail_target == name:
            self._detail.show_error(name, row["error_stage"], msg, row["pct"])

    def restore_error(self, name: str, msg: str, stage: str, pct: int) -> None:
        """Restaura el estado de error de un archivo desde una sesión guardada."""
        if name not in self._rows:
            return
        row = self._rows[name]
        row["state"]       = "error"
        row["pct"]         = pct
        row["error_msg"]   = msg
        row["error_stage"] = stage or "desconocida"
        self._update_badge(name, "error")
        self._update_progress(name, pct)
        row["btn_ver"].setEnabled(True)
        row["btn_resume"].setVisible(True)
        item = self._table.item(row["row_idx"], 4)
        if item:
            item.setText(f"Error: {msg[:60]}")

    # ── Slots internos ───────────────────────────────────────────────────

    def _on_detail_closed(self) -> None:
        if self._detail_target:
            self._highlight_row(self._detail_target, False)
        self._detail_target = ""

    def _toggle_detail(self, name: str) -> None:
        old_target = self._detail_target
        if old_target == name and self._detail.isVisible():
            self._detail.hide()
            self._detail_target = ""
            self._highlight_row(old_target, False)
            return
        if old_target:
            self._highlight_row(old_target, False)
        self._detail_target = name
        row = self._rows.get(name)
        if not row:
            self._detail_target = ""
            return
        shown = False
        if row["state"] == "completado":
            self._detail.show_completed(name, row["events"], row.get("meta"))
            shown = True
        elif row["state"] == "error":
            self._detail.show_error(
                name, row["error_stage"], row["error_msg"], row["pct"]
            )
            shown = True
        if shown:
            self._highlight_row(name, True)
        else:
            self._detail_target = ""

    def _highlight_row(self, name: str, active: bool) -> None:
        row = self._rows.get(name)
        if not row:
            return
        if active:
            self._table.selectRow(row["row_idx"])
        else:
            self._table.clearSelection()

    def _update_badge(self, name: str, state: str) -> None:
        lbl = self._rows[name]["badge_lbl"]
        lbl.setText(state)
        lbl.setStyleSheet(badge_qss(_BADGE_COLORS.get(state, NEUTRAL)))

    def _update_progress(self, name: str, pct: int) -> None:
        row = self._rows[name]
        row["prog_bar"].setValue(pct)
        row["prog_lbl"].setText(f"{pct} %")

    def _check_export(self) -> None:
        has = any(r["state"] == "completado" and r["events"] for r in self._rows.values())
        self._btn_csv.setEnabled(has)
        self._btn_xlsx.setEnabled(has)

    # ── Exportación ──────────────────────────────────────────────────────

    def _collect_export_rows(self) -> list[dict]:
        out: list[dict] = []
        for name, row in self._rows.items():
            if row["state"] != "completado":
                continue
            for ev in row["events"]:
                sp   = getattr(ev, "species",            "—")
                cn   = getattr(ev, "nombre_comun_es_ar", "—")
                conf = getattr(ev, "confidence_level",   "—")
                dist = getattr(ev, "cosine_distance",    "—")
                ivl  = (
                    f"{_fmt_time(ev.start_time)} – {_fmt_time(ev.end_time)}"
                    if hasattr(ev, "start_time") else "—"
                )
                out.append({
                    "archivo":          name,
                    "intervalo":        ivl,
                    "especie":          sp,
                    "nombre_comun":     cn,
                    "confianza":        conf,
                    "distancia_coseno": dist,
                })
        return out

    def _export_csv(self) -> None:
        rows = self._collect_export_rows()
        if not rows:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar CSV", "sareko_resultados.csv", "CSV (*.csv)"
        )
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _export_xlsx(self) -> None:
        rows = self._collect_export_rows()
        if not rows:
            return
        try:
            import openpyxl
        except ImportError:
            # openpyxl no disponible — fallback a CSV
            self._export_csv()
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar XLSX", "sareko_resultados.xlsx", "Excel (*.xlsx)"
        )
        if not path:
            return
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Resultados SAREKO"
        ws.append(list(rows[0].keys()))
        for row in rows:
            ws.append([str(v) for v in row.values()])
        wb.save(path)


# ---------------------------------------------------------------------------
# Pestaña principal
# ---------------------------------------------------------------------------

class AnalisisTab(QWidget):
    """Pestaña Análisis completa."""

    events_ready  = Signal(str, list, str)  # (filename, events, filepath_str)
    batch_changed = Signal()               # emitida al completarse cada archivo
    run_finished  = Signal()               # emitida cuando el worker termina toda la corrida

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")

        self._worker: ProcessingWorker | None = None
        self._species_seen: set[str] = set()
        self._total_frames = 0
        self._current_mode: str = "Estándar"

        outer = QVBoxLayout(self)
        outer.setContentsMargins(32, 24, 32, 32)
        outer.setSpacing(16)

        # Hero
        self._hero = _HeroCard()
        outer.addWidget(self._hero)

        # Dos cards lado a lado
        mid_row = QHBoxLayout()
        mid_row.setSpacing(16)

        self._carga = _CargaCard()
        self._seg   = _SeguimientoCard()
        mid_row.addWidget(self._carga, 1)
        mid_row.addWidget(self._seg,   1)
        outer.addLayout(mid_row)

        # Tabla de lote — ancho completo
        self._batch = _BatchCard()
        outer.addWidget(self._batch)

        outer.addStretch()

        # Señales
        self._carga.btn_procesar.clicked.connect(self._start_processing)
        self._batch.resume_requested.connect(self._on_resume_requested)

    # ------------------------------------------------------------------
    # Procesamiento

    def _start_processing(self) -> None:
        files = self._carga.selected_files
        if not files:
            return

        self._species_seen.clear()
        self._total_frames = 0
        self._current_mode = _MODES[self._carga._combo.currentIndex()][0]
        self._seg.reset()
        self._hero.set_active("procesando")
        self._carga.set_processing(True)

        # Poblar tabla con todos los archivos en "en cola"
        self._batch.populate(files)

        N = self._carga.mode_N
        self._worker = ProcessingWorker(files, N=N, K=10, M=6, batch_size=8)

        self._worker.progress_updated.connect(self._seg.update_progress)
        self._worker.stage_updated.connect(self._seg.update_stage)
        self._worker.stage_updated.connect(self._batch.on_stage_updated)
        self._worker.metrics_updated.connect(self._seg.update_metrics)
        self._worker.file_started.connect(self._on_file_started)
        self._worker.file_completed.connect(self._on_file_completed)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.finished.connect(self._on_finished)

        self._worker.start()

    def _on_file_started(self, name: str) -> None:
        self._batch.on_file_started(name)

    def _on_file_completed(self, name: str, events: list, meta: dict) -> None:
        self._batch.on_file_completed(name, events, meta)
        for ev in events:
            sp = getattr(ev, "species", None)
            if sp:
                self._species_seen.add(sp)
        self._total_frames += meta.get("frames_processed", 1)
        self._hero.set_especies(len(self._species_seen))
        self._hero.set_frames(self._total_frames)
        matching = [f for f in self._carga.selected_files if f.name == name]
        fp_str = str(matching[0]) if matching else ""
        self.events_ready.emit(name, events, fp_str)
        self.batch_changed.emit()

    def _on_error(self, filename: str, msg: str) -> None:
        self._batch.on_file_error(filename, msg)
        self._hero.set_active("error")

    def _on_finished(self) -> None:
        self._hero.set_active("completado")
        self._carga.set_processing(False)
        self._seg.update_progress(100)
        self._worker = None
        self.run_finished.emit()

    # ------------------------------------------------------------------
    # Persistencia de sesión

    def get_batch_summary(self) -> dict:
        """Devuelve el estado actual del lote para serializar en sesión."""
        files = []
        for name, row in self._batch._rows.items():
            path = row.get("path")
            meta = row.get("meta") or {}
            files.append({
                "name":             name,
                "path":             str(path) if path else None,
                "state":            row["state"],
                "pct":              row["pct"],
                "error_msg":        row.get("error_msg", ""),
                "error_stage":      row.get("error_stage", ""),
                "duration_sec":     meta.get("duration_sec"),
                "processing_sec":   meta.get("processing_sec"),
                "frames_processed": meta.get("frames_processed"),
            })
        return {
            "files":        files,
            "species_seen": list(self._species_seen),
            "total_frames": self._total_frames,
            "mode":         self._current_mode,
        }

    def restore_batch(self, summary: dict, records: list[dict]) -> None:
        """Reconstruye la tabla de lote desde una sesión guardada."""
        files_data = summary.get("files", [])
        if not files_data:
            return

        events_by_file: dict[str, list] = {}
        for r in records:
            events_by_file.setdefault(r["filename"], []).append(r["event"])

        self._species_seen = set(summary.get("species_seen", []))
        self._total_frames = summary.get("total_frames", 0)

        file_paths = [
            Path(fd["path"]) if fd.get("path") else Path(fd["name"])
            for fd in files_data
        ]
        self._batch.populate(file_paths)

        has_error = False
        n_completed = 0
        for fd in files_data:
            name  = fd["name"]
            state = fd["state"]
            if state == "completado":
                meta = {k: fd.get(k) for k in ("duration_sec", "processing_sec", "frames_processed")}
                self._batch.on_file_completed(name, events_by_file.get(name, []), meta)
                n_completed += 1
            elif state in ("error", "procesando"):
                has_error = True
                stage = fd.get("error_stage", "desconocida") if state == "error" else "desconocida"
                msg   = fd.get("error_msg", "Procesamiento interrumpido")
                self._batch.restore_error(name, msg, stage, fd.get("pct", 0))

        self._hero.set_especies(len(self._species_seen))
        self._hero.set_frames(self._total_frames)
        if has_error:
            self._hero.set_active("error")
        elif n_completed > 0:
            self._hero.set_active("completado")

    # ------------------------------------------------------------------

    def current_mode(self) -> str:
        """Devuelve el nombre del modo de análisis actualmente seleccionado."""
        return self._current_mode

    def _on_resume_requested(self, name: str) -> None:
        """Reintenta procesar un archivo que falló."""
        if self._worker is not None:
            return  # hay un worker activo — no interrumpir
        matching = [f for f in self._carga.selected_files if f.name == name]
        if not matching:
            return

        self._current_mode = _MODES[self._carga._combo.currentIndex()][0]
        self._seg.reset()
        self._hero.set_active("procesando")
        self._carga.set_processing(True)

        N = self._carga.mode_N
        self._worker = ProcessingWorker(matching, N=N, K=10, M=6, batch_size=8)

        self._worker.progress_updated.connect(self._seg.update_progress)
        self._worker.stage_updated.connect(self._seg.update_stage)
        self._worker.stage_updated.connect(self._batch.on_stage_updated)
        self._worker.metrics_updated.connect(self._seg.update_metrics)
        self._worker.file_started.connect(self._on_file_started)
        self._worker.file_completed.connect(self._on_file_completed)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.finished.connect(self._on_finished)

        self._worker.start()
