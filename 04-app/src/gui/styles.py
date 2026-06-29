"""
Constantes de color y utilidades de estilo QSS para SAREKO.

Paleta: modo oscuro único. Sin modo claro.
"""

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QPushButton

# ---------------------------------------------------------------------------
# Paleta base
# ---------------------------------------------------------------------------
ACCENT          = "#99e17a"   # verde lima — logo, bordes, acentos
TEXT_PRIMARY    = "#edefec"   # texto principal
TAB_ACTIVE_BG   = "#2d5a1f"   # pestaña activa
TAB_INACTIVE_BG = "#090909"   # pestañas inactivas

# Semánticos
SUCCESS = "#99e17a"
WARNING = "#e6b84a"
ERROR   = "#e05c5c"
NEUTRAL = "#4a5248"

# Fondos con alpha (valores 0-255 para Qt QSS)
_NAVBAR_ALPHA  = 217
_CARD_ALPHA    = 178
_BORDER_ALPHA  = 89


# ---------------------------------------------------------------------------
# Colores de badges centralizados (fondo oscuro, texto claro para legibilidad)
# ---------------------------------------------------------------------------

BADGE_COLORS: dict[str, tuple[str, str]] = {
    # Estado de procesamiento
    "en cola":        ("#4a5248", "#edefec"),
    "procesando":     ("#7a5c1a", "#edefec"),
    "completado":     ("#2d5a1f", "#edefec"),
    "error":          ("#6b1f1f", "#edefec"),
    # Confianza (claves cortas usadas internamente)
    "alta":           ("#2d5a1f", "#edefec"),
    "baja":           ("#6b1f1f", "#edefec"),
    "ambiguo":        ("#7a5c1a", "#edefec"),
    # Confianza (claves descriptivas)
    "alta confianza": ("#2d5a1f", "#edefec"),
    "baja confianza": ("#6b1f1f", "#edefec"),
    # Estados auxiliares
    "en espera":           ("#4a5248", "#edefec"),
    "sin corrida activa":  ("#4a5248", "#edefec"),
}

VALIDATION_COLORS: dict[str, tuple[str, str]] = {
    "Correcta":                 ("#2d5a1f", "#edefec"),
    "Top 5":                    ("#7a5c1a", "#edefec"),
    "Conocida fuera del top 5": ("#7a3a1a", "#edefec"),
    "Conocida":                 ("#7a3a1a", "#edefec"),   # alias para "Conocida — X"
    "Desconocida":              ("#6b1f1f", "#edefec"),
    "Vacío / Ruido":            ("#4a5248", "#edefec"),
}


# ---------------------------------------------------------------------------
# SVG inline — plantillas para íconos de acción en tablas
# ---------------------------------------------------------------------------

_SVG_MAGNIFIER = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"'
    ' xmlns="http://www.w3.org/2000/svg">'
    '<circle cx="11" cy="11" r="8"/>'
    '<line x1="21" y1="21" x2="16.65" y2="16.65"/>'
    '</svg>'
)
_SVG_EYE = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"'
    ' xmlns="http://www.w3.org/2000/svg">'
    '<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>'
    '<circle cx="12" cy="12" r="3"/>'
    '</svg>'
)
_SVG_TRASH = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"'
    ' xmlns="http://www.w3.org/2000/svg">'
    '<polyline points="3 6 5 6 21 6"/>'
    '<path d="M19 6l-1 14H6L5 6"/>'
    '<path d="M10 11v6M14 11v6"/>'
    '<path d="M9 6V4h6v2"/>'
    '</svg>'
)


def _svg_to_pixmap(svg_tpl: str, color: str, size: int = 16):
    """Renderiza un SVG template a QPixmap. Devuelve None si falla."""
    try:
        from PySide6.QtSvg import QSvgRenderer
        from PySide6.QtGui import QPixmap, QPainter
        data = svg_tpl.format(color=color).encode("utf-8")
        renderer = QSvgRenderer(data)
        if not renderer.isValid():
            return None
        px = QPixmap(size, size)
        px.fill(Qt.GlobalColor.transparent)
        p = QPainter(px)
        renderer.render(p)
        p.end()
        return px
    except Exception:
        return None


class _HoverIconBtn(QPushButton):
    """QPushButton que intercambia ícono entre estado normal y hover."""

    def __init__(self, svg_tpl: str, tooltip: str,
                 btn_size: int = 28, icon_size: int = 16, parent=None):
        super().__init__(parent)
        self._norm_px = _svg_to_pixmap(svg_tpl, TEXT_PRIMARY, icon_size)
        self._over_px = _svg_to_pixmap(svg_tpl, ACCENT,       icon_size)
        self.setToolTip(tooltip)
        self.setFixedSize(btn_size, btn_size)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFlat(True)
        self.setStyleSheet("QPushButton { background: transparent; border: none; }")
        if self._norm_px:
            self.setIcon(QIcon(self._norm_px))
            self.setIconSize(QSize(icon_size, icon_size))

    def enterEvent(self, event) -> None:
        if self._over_px:
            self.setIcon(QIcon(self._over_px))
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        if self._norm_px:
            self.setIcon(QIcon(self._norm_px))
        super().leaveEvent(event)


def icon_eye_btn(tooltip: str = "Ver detalle",
                 btn_size: int = 28, icon_size: int = 16) -> _HoverIconBtn:
    """Botón con ícono de ojo para 'Ver detalle'."""
    return _HoverIconBtn(_SVG_EYE, tooltip, btn_size, icon_size)


def icon_trash_btn(tooltip: str = "Eliminar",
                   btn_size: int = 28, icon_size: int = 16) -> _HoverIconBtn:
    """Botón con ícono de basurero para 'Eliminar'."""
    return _HoverIconBtn(_SVG_TRASH, tooltip, btn_size, icon_size)


def icon_magnifier_btn(tooltip: str = "Abrir imagen",
                       btn_size: int = 28, icon_size: int = 16) -> _HoverIconBtn:
    """Botón con ícono de lupa minimalista para 'Abrir imagen'."""
    return _HoverIconBtn(_SVG_MAGNIFIER, tooltip, btn_size, icon_size)


def magnifier_icon(color: str = TEXT_PRIMARY, size: int = 16):
    """Renderiza el ícono de lupa como QIcon. Devuelve None si falla."""
    px = _svg_to_pixmap(_SVG_MAGNIFIER, color, size)
    if px:
        from PySide6.QtGui import QIcon
        return QIcon(px)
    return None


# ---------------------------------------------------------------------------
# QSS: navbar
# ---------------------------------------------------------------------------
NAVBAR_QSS = f"""
    QWidget#navbar {{
        background-color: rgba(0, 0, 0, {_NAVBAR_ALPHA});
    }}
"""


# ---------------------------------------------------------------------------
# QSS: scroll area + viewport transparentes
# ---------------------------------------------------------------------------
SCROLL_TRANSPARENT_QSS = """
    QScrollArea { background: transparent; border: none; }
    QScrollBar:vertical {
        background: rgba(255,255,255,15);
        width: 6px;
        margin: 0;
        border-radius: 3px;
    }
    QScrollBar::handle:vertical {
        background: rgba(153, 225, 122, 120);
        border-radius: 3px;
        min-height: 24px;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
"""


# ---------------------------------------------------------------------------
# QSS: card base
# ---------------------------------------------------------------------------
# IMPORTANTE: usar siempre con un objectName para que el selector no cascade
# a los QLabel hijos (que en Qt heredan de QFrame y quedarían con borde).
def card_qss(object_name: str) -> str:
    """QSS para una card con el objectName dado. No afecta a hijos."""
    return f"""
        QFrame#{object_name} {{
            background-color: rgba(0, 0, 0, {_CARD_ALPHA});
            border: 1px solid rgba(153, 225, 122, {_BORDER_ALPHA});
            border-radius: 14px;
        }}
    """

CARD_QSS = card_qss("card")


# ---------------------------------------------------------------------------
# Funciones de estilo dinámico
# ---------------------------------------------------------------------------

def tab_button_qss(active: bool) -> str:
    """QSS para un botón de pestaña del navbar, activo o inactivo."""
    bg         = TAB_ACTIVE_BG if active else TAB_INACTIVE_BG
    hover_bg   = "#3a7028"     if active else "#1a1a1a"
    border_col = ACCENT        if active else "rgba(153, 225, 122, 80)"
    return f"""
        QPushButton {{
            background-color: {bg};
            color:            {TEXT_PRIMARY};
            border:           1px solid {border_col};
            border-radius:    20px;
            font-size:        13px;
            font-weight:      600;
            padding:          8px 22px;
            min-width:        100px;
        }}
        QPushButton:hover {{
            background-color: {hover_bg};
        }}
    """


def badge_qss(bg: str, text: str = TEXT_PRIMARY) -> str:
    """QSS para un badge de estado (pill). bg = fondo, text = color del texto."""
    return f"""
        QLabel {{
            background-color: {bg};
            color:            {text};
            border-radius:    8px;
            padding:          2px 10px;
            font-size:        11px;
            font-weight:      600;
            min-width:        120px;
        }}
    """


def badge_qss_for(state: str) -> str:
    """QSS de badge usando BADGE_COLORS centralizado."""
    bg, text = BADGE_COLORS.get(state, ("#4a5248", TEXT_PRIMARY))
    return badge_qss(bg, text)


def validation_badge_qss(category: str) -> str:
    """QSS para badge de validación según categoría de error."""
    for key, (bg, text) in VALIDATION_COLORS.items():
        if category.startswith(key):
            return badge_qss(bg, text)
    return badge_qss("#2d5a1f", TEXT_PRIMARY)


def section_label_qss() -> str:
    """QSS para etiquetas de sección (uppercase, pequeño, acento)."""
    return f"""
        QLabel {{
            color:           {ACCENT};
            font-size:       10px;
            font-weight:     700;
            letter-spacing:  1px;
            background:      transparent;
        }}
    """


def title_qss(size: int = 28) -> str:
    """QSS para títulos grandes bold."""
    return f"""
        QLabel {{
            color:       {TEXT_PRIMARY};
            font-size:   {size}px;
            font-weight: 700;
            background:  transparent;
        }}
    """


def body_qss(alpha: float = 1.0) -> str:
    """QSS para texto de cuerpo."""
    a = int(alpha * 255)
    return f"""
        QLabel {{
            color:      rgba(237, 239, 236, {a});
            font-size:  14px;
            background: transparent;
        }}
    """
