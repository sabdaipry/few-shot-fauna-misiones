"""
Punto de entrada de SAREKO.

Uso:
    cd 04-app
    python main.py
"""

import sys
from pathlib import Path

# 04-app/ debe estar en sys.path para que 'src' sea importable
_app_dir = Path(__file__).resolve().parent
if str(_app_dir) not in sys.path:
    sys.path.insert(0, str(_app_dir))

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from src.gui.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
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
