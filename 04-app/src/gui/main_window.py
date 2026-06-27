"""
Ventana principal de SAREKO.

Estructura:
    MainWindow
      └── _BackgroundWidget  (dibuja fondo.png escalado a cover)
           └── QVBoxLayout
                ├── _NavBar     (fijo, 64 px)
                └── QScrollArea → QStackedWidget
                     ├── _PlaceholderTab  "Análisis"
                     ├── _PlaceholderTab  "Registros"
                     └── _PlaceholderTab  "Evaluación"
"""

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
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
    tab_button_qss,
    title_qss,
)
from .tabs.analisis_tab import AnalisisTab

_ASSETS = Path(__file__).resolve().parent.parent.parent / "assets"

_TAB_LABELS = ["Análisis", "Registros", "Evaluación"]
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

    tab_selected = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("navbar")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")
        self.setFixedHeight(84)  # 64 card + 10 margen arriba + 10 margen abajo

        outer = QHBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(0)

        card = QFrame()
        card.setObjectName("navbarcard")
        card.setStyleSheet(card_qss("navbarcard"))

        layout = QHBoxLayout(card)
        layout.setContentsMargins(24, 0, 24, 0)
        layout.setSpacing(0)

        # --- Branding izquierdo ---
        logo = QSvgWidget(str(_ASSETS / "SAREKO.svg"))
        logo.setFixedSize(38, 38)
        logo.setStyleSheet("background: transparent;")

        name_label = QLabel("SAREKO")
        name_label.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 20px; font-weight: 700;"
            " letter-spacing: 2px; background: transparent;"
        )

        sub_label = QLabel("Análisis automatizado de cámaras trampa")
        sub_label.setStyleSheet(
            f"color: {ACCENT}; font-size: 10px; font-weight: 500;"
            " background: transparent;"
        )

        brand_col = QVBoxLayout()
        brand_col.setSpacing(1)
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
            btn = QPushButton(label)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.clicked.connect(lambda _checked, idx=i: self._on_tab(idx))
            self._buttons.append(btn)
            btn_container.addWidget(btn)

        layout.addLayout(btn_container)
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
            btn.setStyleSheet(tab_button_qss(i == self._current))


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
        scroll.setWidget(self._stack)

        layout.addWidget(scroll)

        # Páginas de cada pestaña
        self._stack.addWidget(AnalisisTab())
        for title, breadcrumb, subtitle in zip(
            _TAB_LABELS[1:], _TAB_BREADCRUMBS[1:], _TAB_SUBTITLES[1:]
        ):
            self._stack.addWidget(
                _PlaceholderTab(title, breadcrumb, subtitle)
            )

        # Conectar navbar → stack
        self._navbar.tab_selected.connect(self._stack.setCurrentIndex)
