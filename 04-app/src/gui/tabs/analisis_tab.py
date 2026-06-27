"""
Pestaña Análisis de SAREKO.

Estructura:
    AnalisisTab
      ├── _HeroCard              — breadcrumb, título, 5 métricas rápidas
      ├── _CargaCard             — selector de archivos, modo, botón Procesar
      └── _SeguimientoCard       — métricas RT, barra progreso, log de etapas
"""

from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
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

        title = QLabel("Herramienta de análisis automatizado de cámaras trampa")
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
# Pestaña principal
# ---------------------------------------------------------------------------

class AnalisisTab(QWidget):
    """Pestaña Análisis completa."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")

        self._worker: ProcessingWorker | None = None
        self._species_seen: set[str] = set()
        self._total_frames = 0

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

        outer.addStretch()

        # Conectar botón Procesar
        self._carga.btn_procesar.clicked.connect(self._start_processing)

    # ------------------------------------------------------------------
    # Procesamiento

    def _start_processing(self) -> None:
        files = self._carga.selected_files
        if not files:
            return

        self._species_seen.clear()
        self._total_frames = 0
        self._seg.reset()
        self._hero.set_active("procesando")
        self._carga.set_processing(True)

        N = self._carga.mode_N
        self._worker = ProcessingWorker(files, N=N, K=10, M=6, batch_size=8)

        self._worker.progress_updated.connect(self._seg.update_progress)
        self._worker.stage_updated.connect(self._seg.update_stage)
        self._worker.metrics_updated.connect(self._seg.update_metrics)
        self._worker.file_started.connect(self._on_file_started)
        self._worker.file_completed.connect(self._on_file_completed)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.finished.connect(self._on_finished)

        self._worker.start()

    def _on_file_started(self, name: str) -> None:
        pass  # métricas se actualizan vía metrics_updated

    def _on_file_completed(self, name: str, events: list) -> None:
        for ev in events:
            sp = getattr(ev, "species", None)
            if sp:
                self._species_seen.add(sp)
        self._total_frames += 1
        self._hero.set_especies(len(self._species_seen))
        self._hero.set_frames(self._total_frames)

    def _on_error(self, filename: str, msg: str) -> None:
        self._hero.set_active("error")

    def _on_finished(self) -> None:
        self._hero.set_active("completado")
        self._carga.set_processing(False)
        self._seg.update_progress(100)
        self._worker = None
