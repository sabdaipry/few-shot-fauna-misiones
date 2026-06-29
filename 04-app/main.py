"""
Punto de entrada de SAREKO.

Uso:
    cd 04-app
    python main.py
"""

import json
import logging
import sys
from pathlib import Path

# 04-app/ debe estar en sys.path para que 'src' sea importable
_app_dir = Path(__file__).resolve().parent
if str(_app_dir) not in sys.path:
    sys.path.insert(0, str(_app_dir))

from PySide6.QtGui import QColor, QIcon, QPalette
from PySide6.QtWidgets import QApplication

from src.gui.main_window import MainWindow

_app_data_dir = Path(__file__).resolve().parent / "data"
_config_path  = _app_data_dir / "config.json"

def _setup_logging() -> None:
    try:
        config = json.loads(_config_path.read_text(encoding="utf-8"))
    except Exception:
        config = {}

    if config.get("debug_mode", False):
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(name)s] %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(_app_data_dir / "sareko_debug.log", encoding="utf-8"),
            ],
        )
    else:
        logging.basicConfig(level=logging.WARNING)


def main() -> None:
    _setup_logging()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window,          QColor(13,  13,  13))
    dark_palette.setColor(QPalette.WindowText,      QColor(237, 239, 236))
    dark_palette.setColor(QPalette.Base,            QColor(13,  13,  13))
    dark_palette.setColor(QPalette.AlternateBase,   QColor(26,  26,  26))
    dark_palette.setColor(QPalette.ToolTipBase,     QColor(13,  13,  13))
    dark_palette.setColor(QPalette.ToolTipText,     QColor(237, 239, 236))
    dark_palette.setColor(QPalette.Text,            QColor(237, 239, 236))
    dark_palette.setColor(QPalette.Button,          QColor(26,  26,  26))
    dark_palette.setColor(QPalette.ButtonText,      QColor(237, 239, 236))
    dark_palette.setColor(QPalette.Highlight,       QColor(31,  44,  29))
    dark_palette.setColor(QPalette.HighlightedText, QColor(153, 225, 122))
    app.setPalette(dark_palette)

    app.setApplicationName("SAREKO")
    app.setOrganizationName("FI-UBA")

    icon_path = _app_dir / "assets" / "SAREKO.svg"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
