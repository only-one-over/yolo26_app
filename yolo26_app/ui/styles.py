DARK_STYLE = """
QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
    font-size: 13px;
}

QPushButton {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 6px 14px;
    min-height: 20px;
}
QPushButton:hover {
    background-color: #45475a;
    border-color: #89b4fa;
}
QPushButton:pressed {
    background-color: #585b70;
    border-color: #89b4fa;
}
QPushButton:disabled {
    background-color: #313244;
    color: #585b70;
    border-color: #313244;
}
QPushButton:checked {
    background-color: #89b4fa;
    color: #1e1e2e;
    border-color: #89b4fa;
}

QComboBox {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 5px 10px;
    min-height: 22px;
}
QComboBox:hover {
    border-color: #89b4fa;
}
QComboBox::drop-down {
    border: none;
    width: 24px;
}
QComboBox::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #cdd6f4;
    margin-right: 6px;
}
QComboBox QAbstractItemView {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    selection-background-color: #89b4fa;
    selection-color: #1e1e2e;
    outline: none;
}

QLineEdit {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 5px 10px;
    min-height: 22px;
}
QLineEdit:hover {
    border-color: #89b4fa;
}
QLineEdit:focus {
    border-color: #89b4fa;
}

QTextEdit {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 5px;
}
QTextEdit:focus {
    border-color: #89b4fa;
}

QTabWidget::pane {
    border: 1px solid #45475a;
    border-radius: 4px;
    background-color: #1e1e2e;
}
QTabBar::tab {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 7px 18px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background-color: #1e1e2e;
    border-bottom: 2px solid #89b4fa;
    color: #89b4fa;
}
QTabBar::tab:hover:!selected {
    background-color: #45475a;
}

QListWidget {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    outline: none;
}
QListWidget::item {
    padding: 5px 8px;
    border-radius: 4px;
}
QListWidget::item:selected {
    background-color: #89b4fa;
    color: #1e1e2e;
}
QListWidget::item:hover:!selected {
    background-color: #45475a;
}

QTableWidget {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    gridline-color: #45475a;
    outline: none;
}
QTableWidget::item {
    padding: 4px 8px;
}
QTableWidget::item:selected {
    background-color: #89b4fa;
    color: #1e1e2e;
}
QHeaderView::section {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    padding: 5px 8px;
    font-weight: bold;
}

QProgressBar {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 6px;
    text-align: center;
    color: #cdd6f4;
    min-height: 18px;
}
QProgressBar::chunk {
    background-color: #89b4fa;
    border-radius: 5px;
}

QScrollBar:vertical {
    background-color: #1e1e2e;
    width: 10px;
    border-radius: 5px;
}
QScrollBar::handle:vertical {
    background-color: #585b70;
    border-radius: 5px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background-color: #89b4fa;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: none;
}
QScrollBar:horizontal {
    background-color: #1e1e2e;
    height: 10px;
    border-radius: 5px;
}
QScrollBar::handle:horizontal {
    background-color: #585b70;
    border-radius: 5px;
    min-width: 30px;
}
QScrollBar::handle:horizontal:hover {
    background-color: #89b4fa;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
    background: none;
}

QSplitter::handle {
    background-color: #45475a;
}
QSplitter::handle:horizontal {
    width: 2px;
}
QSplitter::handle:vertical {
    height: 2px;
}
QSplitter::handle:hover {
    background-color: #89b4fa;
}

QToolBar {
    background-color: #181825;
    border: none;
    padding: 4px;
    spacing: 4px;
}
QToolBar::separator {
    background-color: #45475a;
    width: 1px;
    margin: 4px 6px;
}

QGroupBox {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 8px;
    margin-top: 12px;
    padding: 14px 10px 10px 10px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    color: #89b4fa;
}

QLabel {
    color: #cdd6f4;
    background-color: transparent;
}

QSpinBox, QDoubleSpinBox {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 4px 8px;
    min-height: 22px;
}
QSpinBox:hover, QDoubleSpinBox:hover {
    border-color: #89b4fa;
}
QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 20px;
    border-left: 1px solid #45475a;
    border-bottom: 1px solid #45475a;
    border-top-right-radius: 6px;
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 20px;
    border-left: 1px solid #45475a;
    border-bottom-right-radius: 6px;
}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-bottom: 5px solid #cdd6f4;
}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #cdd6f4;
}

QCheckBox {
    color: #cdd6f4;
    spacing: 8px;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border: 2px solid #45475a;
    border-radius: 4px;
    background-color: #313244;
}
QCheckBox::indicator:checked {
    background-color: #89b4fa;
    border-color: #89b4fa;
}
QCheckBox::indicator:hover {
    border-color: #89b4fa;
}

QRadioButton {
    color: #cdd6f4;
    spacing: 8px;
}
QRadioButton::indicator {
    width: 18px;
    height: 18px;
    border: 2px solid #45475a;
    border-radius: 9px;
    background-color: #313244;
}
QRadioButton::indicator:checked {
    background-color: #89b4fa;
    border-color: #89b4fa;
}
QRadioButton::indicator:hover {
    border-color: #89b4fa;
}

QSlider::groove:horizontal {
    background-color: #45475a;
    height: 6px;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background-color: #89b4fa;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}
QSlider::handle:horizontal:hover {
    background-color: #b4d0fb;
}
QSlider::groove:vertical {
    background-color: #45475a;
    width: 6px;
    border-radius: 3px;
}
QSlider::handle:vertical {
    background-color: #89b4fa;
    width: 16px;
    height: 16px;
    margin: 0 -5px;
    border-radius: 8px;
}
QSlider::handle:vertical:hover {
    background-color: #b4d0fb;
}

QDockWidget {
    color: #cdd6f4;
    titlebar-close-icon: none;
}
QDockWidget::title {
    background-color: #313244;
    padding: 6px 10px;
    border: 1px solid #45475a;
}
QDockWidget::close-button, QDockWidget::float-button {
    background-color: #313244;
    border: none;
    padding: 2px;
}

QMenu {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 4px;
}
QMenu::item {
    padding: 6px 28px 6px 20px;
    border-radius: 4px;
}
QMenu::item:selected {
    background-color: #89b4fa;
    color: #1e1e2e;
}
QMenu::separator {
    height: 1px;
    background-color: #45475a;
    margin: 4px 8px;
}

QMenuBar {
    background-color: #181825;
    color: #cdd6f4;
    border-bottom: 1px solid #313244;
}
QMenuBar::item {
    padding: 6px 12px;
    border-radius: 4px;
}
QMenuBar::item:selected {
    background-color: #313244;
}

QStatusBar {
    background-color: #181825;
    color: #a6adc8;
    border-top: 1px solid #313244;
    padding: 2px;
}

QToolTip {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 4px 8px;
}

QDialog {
    background-color: #1e1e2e;
}

QStackedWidget {
    background-color: #1e1e2e;
}

QFrame#sidebar {
    background-color: #181825;
    border-right: 1px solid #313244;
}

QPushButton#navButton {
    background-color: transparent;
    border: none;
    border-radius: 8px;
    padding: 10px 4px;
    color: #a6adc8;
    font-size: 12px;
    min-height: 60px;
    min-width: 64px;
}
QPushButton#navButton:hover {
    background-color: #313244;
    color: #cdd6f4;
}
QPushButton#navButton:checked {
    background-color: #313244;
    color: #89b4fa;
    border-left: 3px solid #89b4fa;
}
"""

LIGHT_STYLE = """
QWidget {
    background-color: #eff1f5;
    color: #4c4f69;
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
    font-size: 13px;
}

QPushButton {
    background-color: #ccd0da;
    color: #4c4f69;
    border: 1px solid #bcc0cc;
    border-radius: 6px;
    padding: 6px 14px;
    min-height: 20px;
}
QPushButton:hover {
    background-color: #bcc0cc;
    border-color: #1e66f5;
}
QPushButton:pressed {
    background-color: #acb0ba;
    border-color: #1e66f5;
}
QPushButton:disabled {
    background-color: #ccd0da;
    color: #9ca0b0;
    border-color: #ccd0da;
}
QPushButton:checked {
    background-color: #1e66f5;
    color: #eff1f5;
    border-color: #1e66f5;
}

QComboBox {
    background-color: #ccd0da;
    color: #4c4f69;
    border: 1px solid #bcc0cc;
    border-radius: 6px;
    padding: 5px 10px;
    min-height: 22px;
}
QComboBox:hover {
    border-color: #1e66f5;
}
QComboBox::drop-down {
    border: none;
    width: 24px;
}
QComboBox::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #4c4f69;
    margin-right: 6px;
}
QComboBox QAbstractItemView {
    background-color: #ccd0da;
    color: #4c4f69;
    border: 1px solid #bcc0cc;
    selection-background-color: #1e66f5;
    selection-color: #eff1f5;
    outline: none;
}

QLineEdit {
    background-color: #ccd0da;
    color: #4c4f69;
    border: 1px solid #bcc0cc;
    border-radius: 6px;
    padding: 5px 10px;
    min-height: 22px;
}
QLineEdit:hover {
    border-color: #1e66f5;
}
QLineEdit:focus {
    border-color: #1e66f5;
}

QTextEdit {
    background-color: #ccd0da;
    color: #4c4f69;
    border: 1px solid #bcc0cc;
    border-radius: 6px;
    padding: 5px;
}
QTextEdit:focus {
    border-color: #1e66f5;
}

QTabWidget::pane {
    border: 1px solid #bcc0cc;
    border-radius: 4px;
    background-color: #eff1f5;
}
QTabBar::tab {
    background-color: #ccd0da;
    color: #4c4f69;
    border: 1px solid #bcc0cc;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 7px 18px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background-color: #eff1f5;
    border-bottom: 2px solid #1e66f5;
    color: #1e66f5;
}
QTabBar::tab:hover:!selected {
    background-color: #bcc0cc;
}

QListWidget {
    background-color: #ccd0da;
    color: #4c4f69;
    border: 1px solid #bcc0cc;
    border-radius: 6px;
    outline: none;
}
QListWidget::item {
    padding: 5px 8px;
    border-radius: 4px;
}
QListWidget::item:selected {
    background-color: #1e66f5;
    color: #eff1f5;
}
QListWidget::item:hover:!selected {
    background-color: #bcc0cc;
}

QTableWidget {
    background-color: #ccd0da;
    color: #4c4f69;
    border: 1px solid #bcc0cc;
    border-radius: 6px;
    gridline-color: #bcc0cc;
    outline: none;
}
QTableWidget::item {
    padding: 4px 8px;
}
QTableWidget::item:selected {
    background-color: #1e66f5;
    color: #eff1f5;
}
QHeaderView::section {
    background-color: #ccd0da;
    color: #4c4f69;
    border: 1px solid #bcc0cc;
    padding: 5px 8px;
    font-weight: bold;
}

QProgressBar {
    background-color: #ccd0da;
    border: 1px solid #bcc0cc;
    border-radius: 6px;
    text-align: center;
    color: #4c4f69;
    min-height: 18px;
}
QProgressBar::chunk {
    background-color: #1e66f5;
    border-radius: 5px;
}

QScrollBar:vertical {
    background-color: #eff1f5;
    width: 10px;
    border-radius: 5px;
}
QScrollBar::handle:vertical {
    background-color: #9ca0b0;
    border-radius: 5px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background-color: #1e66f5;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: none;
}
QScrollBar:horizontal {
    background-color: #eff1f5;
    height: 10px;
    border-radius: 5px;
}
QScrollBar::handle:horizontal {
    background-color: #9ca0b0;
    border-radius: 5px;
    min-width: 30px;
}
QScrollBar::handle:horizontal:hover {
    background-color: #1e66f5;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
    background: none;
}

QSplitter::handle {
    background-color: #bcc0cc;
}
QSplitter::handle:horizontal {
    width: 2px;
}
QSplitter::handle:vertical {
    height: 2px;
}
QSplitter::handle:hover {
    background-color: #1e66f5;
}

QToolBar {
    background-color: #e6e9ef;
    border: none;
    padding: 4px;
    spacing: 4px;
}
QToolBar::separator {
    background-color: #bcc0cc;
    width: 1px;
    margin: 4px 6px;
}

QGroupBox {
    background-color: #ccd0da;
    border: 1px solid #bcc0cc;
    border-radius: 8px;
    margin-top: 12px;
    padding: 14px 10px 10px 10px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    color: #1e66f5;
}

QLabel {
    color: #4c4f69;
    background-color: transparent;
}

QSpinBox, QDoubleSpinBox {
    background-color: #ccd0da;
    color: #4c4f69;
    border: 1px solid #bcc0cc;
    border-radius: 6px;
    padding: 4px 8px;
    min-height: 22px;
}
QSpinBox:hover, QDoubleSpinBox:hover {
    border-color: #1e66f5;
}
QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 20px;
    border-left: 1px solid #bcc0cc;
    border-bottom: 1px solid #bcc0cc;
    border-top-right-radius: 6px;
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 20px;
    border-left: 1px solid #bcc0cc;
    border-bottom-right-radius: 6px;
}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-bottom: 5px solid #4c4f69;
}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #4c4f69;
}

QCheckBox {
    color: #4c4f69;
    spacing: 8px;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border: 2px solid #bcc0cc;
    border-radius: 4px;
    background-color: #ccd0da;
}
QCheckBox::indicator:checked {
    background-color: #1e66f5;
    border-color: #1e66f5;
}
QCheckBox::indicator:hover {
    border-color: #1e66f5;
}

QRadioButton {
    color: #4c4f69;
    spacing: 8px;
}
QRadioButton::indicator {
    width: 18px;
    height: 18px;
    border: 2px solid #bcc0cc;
    border-radius: 9px;
    background-color: #ccd0da;
}
QRadioButton::indicator:checked {
    background-color: #1e66f5;
    border-color: #1e66f5;
}
QRadioButton::indicator:hover {
    border-color: #1e66f5;
}

QSlider::groove:horizontal {
    background-color: #bcc0cc;
    height: 6px;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background-color: #1e66f5;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}
QSlider::handle:horizontal:hover {
    background-color: #4c80f7;
}
QSlider::groove:vertical {
    background-color: #bcc0cc;
    width: 6px;
    border-radius: 3px;
}
QSlider::handle:vertical {
    background-color: #1e66f5;
    width: 16px;
    height: 16px;
    margin: 0 -5px;
    border-radius: 8px;
}
QSlider::handle:vertical:hover {
    background-color: #4c80f7;
}

QDockWidget {
    color: #4c4f69;
    titlebar-close-icon: none;
}
QDockWidget::title {
    background-color: #ccd0da;
    padding: 6px 10px;
    border: 1px solid #bcc0cc;
}
QDockWidget::close-button, QDockWidget::float-button {
    background-color: #ccd0da;
    border: none;
    padding: 2px;
}

QMenu {
    background-color: #ccd0da;
    color: #4c4f69;
    border: 1px solid #bcc0cc;
    border-radius: 6px;
    padding: 4px;
}
QMenu::item {
    padding: 6px 28px 6px 20px;
    border-radius: 4px;
}
QMenu::item:selected {
    background-color: #1e66f5;
    color: #eff1f5;
}
QMenu::separator {
    height: 1px;
    background-color: #bcc0cc;
    margin: 4px 8px;
}

QMenuBar {
    background-color: #e6e9ef;
    color: #4c4f69;
    border-bottom: 1px solid #ccd0da;
}
QMenuBar::item {
    padding: 6px 12px;
    border-radius: 4px;
}
QMenuBar::item:selected {
    background-color: #ccd0da;
}

QStatusBar {
    background-color: #e6e9ef;
    color: #7c7f93;
    border-top: 1px solid #ccd0da;
    padding: 2px;
}

QToolTip {
    background-color: #ccd0da;
    color: #4c4f69;
    border: 1px solid #bcc0cc;
    border-radius: 4px;
    padding: 4px 8px;
}

QDialog {
    background-color: #eff1f5;
}

QStackedWidget {
    background-color: #eff1f5;
}

QFrame#sidebar {
    background-color: #e6e9ef;
    border-right: 1px solid #ccd0da;
}

QPushButton#navButton {
    background-color: transparent;
    border: none;
    border-radius: 8px;
    padding: 10px 4px;
    color: #7c7f93;
    font-size: 12px;
    min-height: 60px;
    min-width: 64px;
}
QPushButton#navButton:hover {
    background-color: #ccd0da;
    color: #4c4f69;
}
QPushButton#navButton:checked {
    background-color: #ccd0da;
    color: #1e66f5;
    border-left: 3px solid #1e66f5;
}
"""
