"""
Constantes de color y utilidades de estilo QSS para SAREKO.

Paleta: modo oscuro único. Sin modo claro.
"""

# ---------------------------------------------------------------------------
# Paleta base
# ---------------------------------------------------------------------------
ACCENT          = "#99e17a"   # verde lima — logo, bordes, acentos
TEXT_PRIMARY    = "#edefec"   # texto principal
TAB_ACTIVE_BG   = "#1f2c1d"   # pestaña activa
TAB_INACTIVE_BG = "#090909"   # pestañas inactivas

# Semánticos
SUCCESS = "#99e17a"
WARNING = "#e6b84a"
ERROR   = "#e05c5c"
NEUTRAL = "#4a5248"

# Fondos con alpha (valores 0-255 para Qt QSS)
# rgba(0,0,0, 217) ≈ 0.85 de opacidad
# rgba(0,0,0, 178) ≈ 0.70 de opacidad
# rgba(153,225,122, 89) ≈ 0.35 de opacidad (borde verde suave)
_NAVBAR_ALPHA  = 217
_CARD_ALPHA    = 178
_BORDER_ALPHA  = 89


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
# Uso: card.setObjectName("card"); card.setStyleSheet(card_qss("card"))
def card_qss(object_name: str) -> str:
    """QSS para una card con el objectName dado. No afecta a hijos."""
    return f"""
        QFrame#{object_name} {{
            background-color: rgba(0, 0, 0, {_CARD_ALPHA});
            border: 1px solid rgba(153, 225, 122, {_BORDER_ALPHA});
            border-radius: 14px;
        }}
    """

# Alias para retrocompatibilidad (si algún módulo lo importa como constante)
CARD_QSS = card_qss("card")


# ---------------------------------------------------------------------------
# Funciones de estilo dinámico
# ---------------------------------------------------------------------------

def tab_button_qss(active: bool) -> str:
    """QSS para un botón de pestaña del navbar, activo o inactivo."""
    bg       = TAB_ACTIVE_BG if active else TAB_INACTIVE_BG
    hover_bg = "#2a3d27"     if active else "#1a1a1a"
    return f"""
        QPushButton {{
            background-color: {bg};
            color:            {TEXT_PRIMARY};
            border:           1px solid {ACCENT};
            border-radius:    6px;
            font-size:        13px;
            font-weight:      600;
            padding:          8px 22px;
            min-width:        100px;
        }}
        QPushButton:hover {{
            background-color: {hover_bg};
        }}
    """


def badge_qss(color: str) -> str:
    """QSS para un badge de estado (pill)."""
    return f"""
        QLabel {{
            background-color: {color};
            color:            {TEXT_PRIMARY};
            border-radius:    8px;
            padding:          2px 10px;
            font-size:        11px;
            font-weight:      600;
        }}
    """


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
