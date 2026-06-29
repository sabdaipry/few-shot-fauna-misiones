"""
Ventana principal de SAREKO.

Estructura:
    MainWindow
      └── _BackgroundWidget  (dibuja fondo.png escalado a cover)
           └── QVBoxLayout
                ├── _NavBar       (fijo, 64 px)
                └── QScrollArea → QStackedWidget
                     ├── AnalisisTab
                     ├── ValidacionTab
                     └── EvaluacionTab
"""

import csv
from pathlib import Path

from PySide6.QtCore import Qt, Signal, QSize, QEvent, QRectF
from PySide6.QtGui import QIcon, QPainter, QPixmap, QColor, QPainterPath, QPen, QFont
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .styles import (
    ACCENT,
    SCROLL_TRANSPARENT_QSS,
    TEXT_PRIMARY,
    body_qss,
    card_qss,
    section_label_qss,
    title_qss,
)
from .tabs.analisis_tab import AnalisisTab
from .tabs.evaluacion_tab import EvaluacionTab
from .tabs.validacion_tab import ValidacionTab
from ..data.session import SessionManager

_DARK_MSG_QSS = """
    QMessageBox {
        background: #0d1a0b;
        color: #edefec;
    }
    QMessageBox QLabel {
        color: #edefec;
        font-size: 13px;
        background: transparent;
    }
    QPushButton {
        background: rgba(153,225,122,30);
        color: #edefec;
        border: 1px solid rgba(153,225,122,80);
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        padding: 6px 16px;
        min-width: 130px;
    }
    QPushButton:hover { background: rgba(153,225,122,50); }
    QPushButton:default {
        background: rgba(153,225,122,60);
        border-color: rgba(153,225,122,160);
    }
"""

_SVG_TRASH_ICON = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"'
    ' xmlns="http://www.w3.org/2000/svg">'
    '<polyline points="3 6 5 6 21 6"/>'
    '<path d="M19 6l-1 14H6L5 6"/>'
    '<path d="M10 11v6M14 11v6"/>'
    '<path d="M9 6V4h6v2"/>'
    '</svg>'
)


def _render_svg_icon(svg_tpl: str, color: str, size: int) -> "QIcon | None":
    """Renderiza un SVG template (con {color}) a QIcon."""
    try:
        from PySide6.QtSvg import QSvgRenderer
        data = svg_tpl.format(color=color).encode("utf-8")
        renderer = QSvgRenderer(data)
        if not renderer.isValid():
            return None
        px = QPixmap(size, size)
        px.fill(Qt.GlobalColor.transparent)
        p = QPainter(px)
        renderer.render(p)
        p.end()
        return QIcon(px)
    except Exception:
        return None


_ASSETS        = Path(__file__).resolve().parent.parent.parent / "assets"
_CATALOG_CSV   = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "02-benchmarking" / "data" / "dataset_index.csv"
)


def _load_species_catalog() -> list[dict]:
    """Lee dataset_index.csv y devuelve lista de dicts únicos por especie."""
    try:
        catalog: list[dict] = []
        seen: set[str] = set()
        with open(_CATALOG_CSV, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                sp = row.get("species", "").strip()
                if sp and sp not in seen:
                    seen.add(sp)
                    catalog.append({
                        "species":           sp,
                        "nombre_comun_es_ar": row.get("nombre_comun_es_ar", ""),
                        "nombre_comun_en":    row.get("nombre_comun_en", ""),
                    })
        return catalog
    except Exception:
        return []

_TAB_LABELS = ["Análisis", "Validación", "Evaluación"]
_TAB_BREADCRUMBS = [
    "SAREKO / ANÁLISIS",
    "SAREKO / DETALLE OPERATIVO",
    "SAREKO / ANALYTICS",
]
_TAB_SUBTITLES = [
    "Carga de archivos, seguimiento de análisis y resumen del lote.",
    "Registros de detección, validación humana y visor de frames.",
    "Indicadores operativos, gráficos y evaluación de errores.",
]


# ---------------------------------------------------------------------------
# Botón de pestaña con forma de píldora (paintEvent manual)
# ---------------------------------------------------------------------------

class _TabButton(QPushButton):
    """
    QPushButton cuya forma de píldora se dibuja con QPainter.
    Necesario porque Fusion ignora border-radius del QSS en QPushButton.
    """

    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setFlat(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover)
        self.setMinimumWidth(100)
        self._active  = False
        self._hovered = False
        font = self.font()
        font.setPixelSize(13)
        font.setWeight(QFont.Weight.DemiBold)
        self.setFont(font)

    def set_active(self, active: bool) -> None:
        if self._active != active:
            self._active = active
            self.update()

    def event(self, e):
        t = e.type()
        if t == QEvent.Type.HoverEnter:
            self._hovered = True
            self.update()
        elif t == QEvent.Type.HoverLeave:
            self._hovered = False
            self.update()
        return super().event(e)

    def sizeHint(self):
        w = max(100, self.fontMetrics().horizontalAdvance(self.text()) + 44)
        return QSize(w, 36)

    def paintEvent(self, _event):
        if self._active:
            bg  = QColor("#3a7028" if self._hovered else "#2d5a1f")
            brd = QColor(ACCENT)
        else:
            bg  = QColor("#1a1a1a" if self._hovered else "#090909")
            brd = QColor(153, 225, 122, 80)

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect   = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        radius = rect.height() / 2.0

        path = QPainterPath()
        path.addRoundedRect(rect, radius, radius)

        p.fillPath(path, bg)
        p.setPen(QPen(brd, 1.0))
        p.drawPath(path)

        p.setPen(QColor(TEXT_PRIMARY))
        p.setFont(self.font())
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.text())


# ---------------------------------------------------------------------------
# Widget raíz: dibuja el fondo
# ---------------------------------------------------------------------------

class _BackgroundWidget(QWidget):
    """Dibuja fondo.png escalado a cover como fondo de toda la ventana."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = QPixmap(str(_ASSETS / "fondo.png"))

    def paintEvent(self, event):
        if self._pixmap.isNull():
            super().paintEvent(event)
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        scaled = self._pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        p.drawPixmap(x, y, scaled)


# ---------------------------------------------------------------------------
# Navbar
# ---------------------------------------------------------------------------

class _NavBar(QWidget):
    """Barra de navegación fija con logo y tres botones de pestaña."""

    tab_selected    = Signal(int)
    reset_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("navbar")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("QWidget#navbar { background: transparent; }")
        self.setFixedHeight(100)  # 70 card + 20 margen arriba + 10 margen abajo

        outer = QHBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 10)
        outer.setSpacing(0)

        card = QFrame()
        card.setObjectName("navbarcard")
        card.setStyleSheet(card_qss("navbarcard"))

        layout = QHBoxLayout(card)
        layout.setContentsMargins(12, 0, 12, 0)
        layout.setSpacing(0)

        # --- Branding izquierdo ---
        logo = QSvgWidget(str(_ASSETS / "SAREKO.svg"))
        logo.setFixedSize(43, 60)
        logo.setStyleSheet("background: transparent;")

        name_label = QLabel("SAREKO")
        name_label.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 40px; font-weight: 600;"
            " letter-spacing: 12px; background: transparent;"
        )

        sub_label = QLabel("Análisis automatizado de cámaras trampa")
        sub_label.setStyleSheet(
            f"color: {ACCENT}; font-size: 11px; font-weight: 500;"
            " background: transparent;"
        )

        brand_col = QVBoxLayout()
        brand_col.setSpacing(5)
        brand_col.setContentsMargins(0, 0, 0, 0)
        brand_col.addWidget(name_label)
        brand_col.addWidget(sub_label)

        left = QHBoxLayout()
        left.setSpacing(12)
        left.setContentsMargins(0, 0, 0, 0)
        left.addWidget(logo)
        left.addLayout(brand_col)

        layout.addLayout(left)
        layout.addStretch()

        # --- Botones de pestaña ---
        self._buttons: list[QPushButton] = []
        btn_container = QHBoxLayout()
        btn_container.setSpacing(8)
        btn_container.setContentsMargins(0, 0, 0, 0)

        for i, label in enumerate(_TAB_LABELS):
            btn = _TabButton(label)
            btn.clicked.connect(lambda _checked, idx=i: self._on_tab(idx))
            self._buttons.append(btn)
            btn_container.addWidget(btn)

        layout.addLayout(btn_container)
        layout.addSpacing(12)

        # --- Botón Resetear (ícono papelera, extremo derecho) ---
        self._btn_reset = QPushButton()
        self._btn_reset.setToolTip("Resetear historial completo")
        self._btn_reset.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_reset.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._btn_reset.setFixedSize(32, 32)
        _reset_icon = _render_svg_icon(_SVG_TRASH_ICON, "#e05c5c", 18)
        if _reset_icon:
            self._btn_reset.setIcon(_reset_icon)
            self._btn_reset.setIconSize(QSize(18, 18))
        self._btn_reset.setStyleSheet("""
            QPushButton {
                background: rgba(224,92,92,20);
                border: 1px solid rgba(224,92,92,60);
                border-radius: 6px;
            }
            QPushButton:hover {
                background: rgba(224,92,92,50);
                border-color: rgba(224,92,92,120);
            }
        """)
        self._btn_reset.clicked.connect(self.reset_requested)
        layout.addWidget(self._btn_reset)
        outer.addWidget(card)

        self._current = 0
        self._refresh_buttons()

    # ------------------------------------------------------------------

    def _on_tab(self, index: int):
        if index == self._current:
            return
        self._current = index
        self._refresh_buttons()
        self.tab_selected.emit(index)

    def _refresh_buttons(self):
        for i, btn in enumerate(self._buttons):
            btn.set_active(i == self._current)


# ---------------------------------------------------------------------------
# Pestaña placeholder
# ---------------------------------------------------------------------------

class _PlaceholderTab(QWidget):
    """
    Contenido temporal para cada pestaña.
    Será reemplazado por los widgets reales en sesiones siguientes.
    """

    def __init__(self, title: str, breadcrumb: str, subtitle: str, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(32, 32, 32, 32)
        outer.setSpacing(20)
        outer.addStretch(1)

        # Card central — objectName limita el selector al QFrame específico
        card = QFrame()
        card.setObjectName("card")
        card.setStyleSheet(card_qss("card"))
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(36, 32, 36, 32)
        card_layout.setSpacing(12)

        bc = QLabel(breadcrumb)
        bc.setStyleSheet(section_label_qss())

        lbl = QLabel(title)
        lbl.setStyleSheet(title_qss(36))

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"border: none; border-top: 1px solid rgba(153,225,122,60);")
        sep.setFixedHeight(1)

        sub = QLabel(subtitle)
        sub.setStyleSheet(body_qss(0.6))
        sub.setWordWrap(True)

        pending = QLabel("Implementación pendiente — próxima sesión.")
        pending.setStyleSheet(
            f"color: {ACCENT}; font-size: 12px; font-style: italic; background: transparent;"
        )

        card_layout.addWidget(bc)
        card_layout.addWidget(lbl)
        card_layout.addWidget(sep)
        card_layout.addWidget(sub)
        card_layout.addWidget(pending)

        outer.addWidget(card)
        outer.addStretch(3)


# ---------------------------------------------------------------------------
# Ventana principal
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """Ventana principal de SAREKO."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAREKO — Análisis automatizado de cámaras trampa")
        self.setMinimumSize(1100, 660)
        self.resize(1400, 860)

        # Widget raíz con fondo
        root = _BackgroundWidget()
        self.setCentralWidget(root)

        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Navbar fijo
        self._navbar = _NavBar()
        layout.addWidget(self._navbar)

        # Área de contenido scrolleable
        self._stack = QStackedWidget()
        self._stack.setAutoFillBackground(False)
        self._stack.setStyleSheet("background: transparent;")

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet(SCROLL_TRANSPARENT_QSS)
        scroll.setAutoFillBackground(False)
        scroll.viewport().setAutoFillBackground(False)
        scroll.viewport().setStyleSheet("background: transparent;")
        scroll.setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(self._stack)

        layout.addWidget(scroll)

        # Catálogo de especies (cargado una sola vez)
        _species_catalog = _load_species_catalog()

        # Páginas de cada pestaña
        self._analisis_tab   = AnalisisTab()
        self._validacion_tab = ValidacionTab(species_catalog=_species_catalog)
        self._evaluacion_tab = EvaluacionTab(species_catalog=_species_catalog)

        self._stack.addWidget(self._analisis_tab)
        self._stack.addWidget(self._validacion_tab)
        self._stack.addWidget(self._evaluacion_tab)

        # Conectar navbar → stack
        self._navbar.tab_selected.connect(self._stack.setCurrentIndex)
        self._navbar.reset_requested.connect(self._on_reset)

        # Propagar eventos completados de Análisis → Validación
        self._analisis_tab.events_ready.connect(self._on_events_ready)

        # Persistencia de sesión: guardar al completar cada archivo y al validar
        self._analisis_tab.batch_changed.connect(self._save_session)
        self._validacion_tab.validation_changed.connect(self._save_session)

        # Actualizar pestaña Evaluación al completar archivos o al validar
        self._analisis_tab.batch_changed.connect(self._update_evaluacion)
        self._validacion_tab.validation_changed.connect(self._update_evaluacion)

        # Sincronizar validaciones hechas desde el panel lateral de Evaluación
        self._evaluacion_tab.validation_changed.connect(self._on_evaluacion_validation)

        # Al terminar toda la corrida: acumular en historial
        self._analisis_tab.run_finished.connect(self._on_run_finished)

        # Restaurar sesión previa si existe
        self._restore_session()

    # ------------------------------------------------------------------

    def _on_events_ready(self, filename: str, events: list, filepath_str: str) -> None:
        fp = Path(filepath_str) if filepath_str else None
        self._validacion_tab.add_events(filename, events, fp)

    def _save_session(self) -> None:
        records = self._validacion_tab.get_records()
        summary = self._analisis_tab.get_batch_summary()
        SessionManager.save(records, summary)

    def _on_run_finished(self) -> None:
        records = self._validacion_tab.get_records()
        summary = self._analisis_tab.get_batch_summary()
        history = SessionManager.append_history(summary, records)
        self._evaluacion_tab.update_from_session(records, summary, history)

    def _restore_session(self) -> None:
        result = SessionManager.load()
        if result is None:
            return
        records, summary = result
        self._validacion_tab.restore_records(records)
        self._analisis_tab.restore_batch(summary, records)
        self._evaluacion_tab.update_from_session(records, summary, SessionManager.load_history())

    def _update_evaluacion(self) -> None:
        records = self._validacion_tab.get_records()
        summary = self._analisis_tab.get_batch_summary()
        self._evaluacion_tab.update_from_session(records, summary, SessionManager.load_history())

    def _on_evaluacion_validation(self) -> None:
        """
        Una validación fue guardada desde el panel lateral de EvaluacionTab.

        Los dicts de registros son compartidos por referencia entre ambas pestañas,
        por lo que el cambio ya está en memoria. Sólo hay que:
        1. Guardar la sesión en disco.
        2. Refrescar la tabla de ValidacionTab para que refleje el estado actualizado.
        """
        self._save_session()
        # restore_records reconstruye la tabla leyendo el estado actual de los dicts
        self._validacion_tab.restore_records(self._validacion_tab.get_records())

    def _on_reset(self) -> None:
        msg = QMessageBox(self)
        msg.setWindowTitle("Confirmar reset")
        msg.setText(
            "Esto eliminará todos los registros de validación, el historial de corridas "
            "y los datos de evaluación.\n\n"
            "Esta acción no se puede deshacer. ¿Confirmar?"
        )
        msg.setStyleSheet(_DARK_MSG_QSS)
        btn_ok     = msg.addButton("Confirmar", QMessageBox.ButtonRole.AcceptRole)
        btn_cancel = msg.addButton("Cancelar",  QMessageBox.ButtonRole.RejectRole)
        msg.setDefaultButton(btn_cancel)
        msg.exec()
        if msg.clickedButton() != btn_ok:
            return
        SessionManager.clear()
        SessionManager.clear_history()
        self._analisis_tab.reset()
        self._validacion_tab.reset()
        self._evaluacion_tab.reset()

    def closeEvent(self, event) -> None:
        self._save_session()
        worker = SessionManager._worker
        if worker and worker.isRunning():
            worker.wait(2000)
        super().closeEvent(event)
