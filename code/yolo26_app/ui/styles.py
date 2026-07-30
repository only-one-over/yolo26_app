"""Design token system and QSS style sheet generation for YOLO26 App."""

DARK_TOKENS = {
    "color_base": "#1e1e2e",
    "color_mantle": "#181825",
    "color_crust": "#11111b",
    "color_surface_0": "#313244",
    "color_surface_1": "#45475a",
    "color_surface_2": "#585b70",
    "color_text": "#cdd6f4",
    "color_text_subtle": "#a6adc8",
    "color_text_muted": "#7f849c",
    "color_primary": "#89b4fa",
    "color_primary_hover": "#b4befe",
    "color_primary_pressed": "#74c7ec",
    "color_success": "#a6e3a1",
    "color_success_hover": "#94d891",
    "color_success_pressed": "#7dc97c",
    "color_warning": "#f9e2af",
    "color_warning_bg": "#3a3520",
    "color_danger": "#f38ba8",
    "color_danger_hover": "#eb7891",
    "color_danger_pressed": "#e06c8a",
    "color_accent": "#cba6f7",
    "font_sans": '"Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif',
    "font_mono": '"JetBrains Mono", "Cascadia Code", "Consolas", monospace',
    "font_size_xs": "11px",
    "font_size_sm": "12px",
    "font_size_base": "13px",
    "font_size_md": "14px",
    "font_size_lg": "16px",
    "font_size_xl": "20px",
    "space_1": "4px",
    "space_2": "8px",
    "space_3": "12px",
    "space_4": "16px",
    "space_5": "20px",
    "space_6": "24px",
    "space_8": "32px",
    "radius_sm": "4px",
    "radius_md": "6px",
    "radius_lg": "8px",
    "radius_xl": "12px",
}

LIGHT_TOKENS = {
    "color_base": "#eff1f5",
    "color_mantle": "#e6e9ef",
    "color_crust": "#dce0e8",
    "color_surface_0": "#ccd0da",
    "color_surface_1": "#bcc0cc",
    "color_surface_2": "#acb0ba",
    "color_text": "#4c4f69",
    "color_text_subtle": "#6c6f85",
    "color_text_muted": "#9ca0b0",
    "color_primary": "#1e66f5",
    "color_primary_hover": "#2c6ef5",
    "color_primary_pressed": "#0d52d4",
    "color_success": "#40a02b",
    "color_success_hover": "#348b23",
    "color_success_pressed": "#2a7a1c",
    "color_warning": "#df8e1d",
    "color_warning_bg": "#f5e6c8",
    "color_danger": "#d20f39",
    "color_danger_hover": "#c00d33",
    "color_danger_pressed": "#a80c2d",
    "color_accent": "#8839ef",
    "font_sans": '"Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif',
    "font_mono": '"JetBrains Mono", "Cascadia Code", "Consolas", monospace',
    "font_size_xs": "11px",
    "font_size_sm": "12px",
    "font_size_base": "13px",
    "font_size_md": "14px",
    "font_size_lg": "16px",
    "font_size_xl": "20px",
    "space_1": "4px",
    "space_2": "8px",
    "space_3": "12px",
    "space_4": "16px",
    "space_5": "20px",
    "space_6": "24px",
    "space_8": "32px",
    "radius_sm": "4px",
    "radius_md": "6px",
    "radius_lg": "8px",
    "radius_xl": "12px",
}


def _build_style(t):
    lines = []
    lines.append(f"QWidget {{ background-color: {t['color_base']}; color: {t['color_text']}; font-family: {t['font_sans']}; font-size: {t['font_size_base']}; }}")
    lines.append(f"QPushButton {{ background-color: {t['color_surface_0']}; color: {t['color_text']}; border: 1px solid {t['color_surface_1']}; border-radius: {t['radius_md']}; padding: {t['space_1']} {t['space_3']}; min-height: 20px; }}")
    lines.append(f"QPushButton:hover {{ background-color: {t['color_surface_1']}; border-color: {t['color_primary']}; }}")
    lines.append(f"QPushButton:pressed {{ background-color: {t['color_surface_2']}; border-color: {t['color_primary']}; }}")
    lines.append(f"QPushButton:disabled {{ background-color: {t['color_surface_0']}; color: {t['color_text_muted']}; border-color: {t['color_surface_0']}; }}")
    lines.append(f"QPushButton:checked {{ background-color: {t['color_primary']}; color: {t['color_base']}; border-color: {t['color_primary']}; }}")
    lines.append(f"QComboBox {{ background-color: {t['color_surface_0']}; color: {t['color_text']}; border: 1px solid {t['color_surface_1']}; border-radius: {t['radius_md']}; padding: {t['space_1']} {t['space_2']}; min-height: 22px; }}")
    lines.append(f"QComboBox:hover {{ border-color: {t['color_primary']}; }}")
    lines.append(f"QComboBox::drop-down {{ border: none; width: 24px; }}")
    lines.append(f"QComboBox::down-arrow {{ image: none; border-left: 5px solid transparent; border-right: 5px solid transparent; border-top: 6px solid {t['color_text']}; margin-right: 6px; }}")
    lines.append(f"QComboBox QAbstractItemView {{ background-color: {t['color_surface_0']}; color: {t['color_text']}; border: 1px solid {t['color_surface_1']}; selection-background-color: {t['color_primary']}; selection-color: {t['color_base']}; outline: none; }}")
    lines.append(f"QLineEdit {{ background-color: {t['color_surface_0']}; color: {t['color_text']}; border: 1px solid {t['color_surface_1']}; border-radius: {t['radius_md']}; padding: {t['space_1']} {t['space_2']}; min-height: 22px; }}")
    lines.append(f"QLineEdit:hover {{ border-color: {t['color_primary']}; }}")
    lines.append(f"QLineEdit:focus {{ border-color: {t['color_primary']}; }}")
    lines.append(f"QTextEdit {{ background-color: {t['color_surface_0']}; color: {t['color_text']}; border: 1px solid {t['color_surface_1']}; border-radius: {t['radius_md']}; padding: {t['space_1']}; }}")
    lines.append(f"QTextEdit:focus {{ border-color: {t['color_primary']}; }}")
    lines.append(f"QTextEdit#logView {{ background-color: {t['color_crust']}; color: {t['color_text_subtle']}; font-family: {t['font_mono']}; font-size: {t['font_size_sm']}; border: 1px solid {t['color_surface_1']}; border-radius: {t['radius_md']}; padding: {t['space_2']}; }}")
    lines.append(f"QTabWidget::pane {{ border: 1px solid {t['color_surface_1']}; border-radius: {t['radius_sm']}; background-color: {t['color_base']}; }}")
    lines.append(f"QTabBar::tab {{ background-color: {t['color_surface_0']}; color: {t['color_text']}; border: 1px solid {t['color_surface_1']}; border-bottom: none; border-top-left-radius: {t['radius_md']}; border-top-right-radius: {t['radius_md']}; padding: 7px 18px; margin-right: 2px; }}")
    lines.append(f"QTabBar::tab:selected {{ background-color: {t['color_base']}; border-bottom: 2px solid {t['color_primary']}; color: {t['color_primary']}; }}")
    lines.append(f"QTabBar::tab:hover:!selected {{ background-color: {t['color_surface_1']}; }}")
    lines.append(f"QListWidget {{ background-color: {t['color_surface_0']}; color: {t['color_text']}; border: 1px solid {t['color_surface_1']}; border-radius: {t['radius_md']}; outline: none; }}")
    lines.append(f"QListWidget::item {{ padding: {t['space_1']} {t['space_2']}; border-radius: {t['radius_sm']}; }}")
    lines.append(f"QListWidget::item:selected {{ background-color: {t['color_primary']}; color: {t['color_base']}; }}")
    lines.append(f"QListWidget::item:hover:!selected {{ background-color: {t['color_surface_1']}; }}")
    lines.append(f"QTableWidget {{ background-color: {t['color_surface_0']}; color: {t['color_text']}; border: 1px solid {t['color_surface_1']}; border-radius: {t['radius_md']}; gridline-color: {t['color_surface_1']}; outline: none; }}")
    lines.append(f"QTableWidget::item {{ padding: {t['space_1']} {t['space_2']}; }}")
    lines.append(f"QTableWidget::item:selected {{ background-color: {t['color_primary']}; color: {t['color_base']}; }}")
    lines.append(f"QHeaderView::section {{ background-color: {t['color_surface_0']}; color: {t['color_text']}; border: 1px solid {t['color_surface_1']}; padding: {t['space_1']} {t['space_2']}; font-weight: bold; }}")
    lines.append(f"QProgressBar {{ background-color: {t['color_surface_0']}; border: 1px solid {t['color_surface_1']}; border-radius: {t['radius_md']}; text-align: center; color: {t['color_text']}; min-height: 18px; }}")
    lines.append(f"QProgressBar::chunk {{ background-color: {t['color_primary']}; border-radius: {t['radius_sm']}; }}")
    lines.append(f"QScrollBar:vertical {{ background-color: {t['color_base']}; width: 10px; border-radius: 5px; }}")
    lines.append(f"QScrollBar::handle:vertical {{ background-color: {t['color_surface_2']}; border-radius: 5px; min-height: 30px; }}")
    lines.append(f"QScrollBar::handle:vertical:hover {{ background-color: {t['color_primary']}; }}")
    lines.append(f"QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}")
    lines.append(f"QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}")
    lines.append(f"QScrollBar:horizontal {{ background-color: {t['color_base']}; height: 10px; border-radius: 5px; }}")
    lines.append(f"QScrollBar::handle:horizontal {{ background-color: {t['color_surface_2']}; border-radius: 5px; min-width: 30px; }}")
    lines.append(f"QScrollBar::handle:horizontal:hover {{ background-color: {t['color_primary']}; }}")
    lines.append(f"QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0px; }}")
    lines.append(f"QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{ background: none; }}")
    lines.append(f"QSplitter::handle {{ background-color: {t['color_surface_1']}; }}")
    lines.append(f"QSplitter::handle:horizontal {{ width: 2px; }}")
    lines.append(f"QSplitter::handle:vertical {{ height: 2px; }}")
    lines.append(f"QSplitter::handle:hover {{ background-color: {t['color_primary']}; }}")
    lines.append(f"QToolBar {{ background-color: {t['color_mantle']}; border: none; padding: {t['space_1']}; spacing: {t['space_1']}; }}")
    lines.append(f"QToolBar::separator {{ background-color: {t['color_surface_1']}; width: 1px; margin: {t['space_1']} {t['space_2']}; }}")
    lines.append(f"QGroupBox {{ background-color: {t['color_surface_0']}; border: 1px solid {t['color_surface_1']}; border-radius: {t['radius_lg']}; margin-top: 12px; padding: 14px 10px 10px 10px; }}")
    lines.append(f"QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; left: 12px; padding: 0 6px; color: {t['color_primary']}; }}")
    lines.append(f"QGroupBox#configCard {{ background-color: {t['color_surface_0']}; border: 1px solid {t['color_surface_1']}; border-radius: {t['radius_lg']}; margin-top: 8px; padding: 16px 12px 12px 12px; }}")
    lines.append(f"QGroupBox#configCard::title {{ subcontrol-origin: margin; subcontrol-position: top left; left: 12px; padding: 0 6px; color: {t['color_primary']}; font-weight: bold; }}")
    lines.append(f"QLabel {{ color: {t['color_text']}; background-color: transparent; }}")
    lines.append(f"QLabel#infoLabel {{ color: {t['color_text_subtle']}; font-size: {t['font_size_xs']}; }}")
    lines.append(f"QLabel#warningLabel {{ color: {t['color_warning']}; background-color: {t['color_warning_bg']}; border-radius: {t['radius_sm']}; padding: {t['space_1']} {t['space_2']}; }}")
    lines.append(f"QLabel#projectLabel {{ color: {t['color_text_subtle']}; font-size: {t['font_size_sm']}; }}")
    lines.append(f"QSpinBox, QDoubleSpinBox {{ background-color: {t['color_surface_0']}; color: {t['color_text']}; border: 1px solid {t['color_surface_1']}; border-radius: {t['radius_md']}; padding: {t['space_1']} {t['space_2']}; min-height: 22px; }}")
    lines.append(f"QSpinBox:hover, QDoubleSpinBox:hover {{ border-color: {t['color_primary']}; }}")
    lines.append(f"QSpinBox::up-button, QDoubleSpinBox::up-button {{ subcontrol-origin: border; subcontrol-position: top right; width: 20px; border-left: 1px solid {t['color_surface_1']}; border-bottom: 1px solid {t['color_surface_1']}; border-top-right-radius: {t['radius_md']}; }}")
    lines.append(f"QSpinBox::down-button, QDoubleSpinBox::down-button {{ subcontrol-origin: border; subcontrol-position: bottom right; width: 20px; border-left: 1px solid {t['color_surface_1']}; border-bottom-right-radius: {t['radius_md']}; }}")
    lines.append(f"QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: none; border-left: 4px solid transparent; border-right: 4px solid transparent; border-bottom: 5px solid {t['color_text']}; }}")
    lines.append(f"QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: none; border-left: 4px solid transparent; border-right: 4px solid transparent; border-top: 5px solid {t['color_text']}; }}")
    lines.append(f"QCheckBox {{ color: {t['color_text']}; spacing: {t['space_2']}; }}")
    lines.append(f"QCheckBox::indicator {{ width: 18px; height: 18px; border: 2px solid {t['color_surface_1']}; border-radius: {t['radius_sm']}; background-color: {t['color_surface_0']}; }}")
    lines.append(f"QCheckBox::indicator:checked {{ background-color: {t['color_primary']}; border-color: {t['color_primary']}; }}")
    lines.append(f"QCheckBox::indicator:hover {{ border-color: {t['color_primary']}; }}")
    lines.append(f"QRadioButton {{ color: {t['color_text']}; spacing: {t['space_2']}; }}")
    lines.append(f"QRadioButton::indicator {{ width: 18px; height: 18px; border: 2px solid {t['color_surface_1']}; border-radius: 9px; background-color: {t['color_surface_0']}; }}")
    lines.append(f"QRadioButton::indicator:checked {{ background-color: {t['color_primary']}; border-color: {t['color_primary']}; }}")
    lines.append(f"QRadioButton::indicator:hover {{ border-color: {t['color_primary']}; }}")
    lines.append(f"QSlider::groove:horizontal {{ background-color: {t['color_surface_1']}; height: 6px; border-radius: 3px; }}")
    lines.append(f"QSlider::handle:horizontal {{ background-color: {t['color_primary']}; width: 16px; height: 16px; margin: -5px 0; border-radius: 8px; }}")
    lines.append(f"QSlider::handle:horizontal:hover {{ background-color: {t['color_primary_hover']}; }}")
    lines.append(f"QSlider::groove:vertical {{ background-color: {t['color_surface_1']}; width: 6px; border-radius: 3px; }}")
    lines.append(f"QSlider::handle:vertical {{ background-color: {t['color_primary']}; width: 16px; height: 16px; margin: 0 -5px; border-radius: 8px; }}")
    lines.append(f"QSlider::handle:vertical:hover {{ background-color: {t['color_primary_hover']}; }}")
    lines.append(f"QMenu {{ background-color: {t['color_surface_0']}; color: {t['color_text']}; border: 1px solid {t['color_surface_1']}; border-radius: {t['radius_md']}; padding: {t['space_1']}; }}")
    lines.append(f"QMenu::item {{ padding: {t['space_1']} 28px {t['space_1']} 20px; border-radius: {t['radius_sm']}; }}")
    lines.append(f"QMenu::item:selected {{ background-color: {t['color_primary']}; color: {t['color_base']}; }}")
    lines.append(f"QMenu::separator {{ height: 1px; background-color: {t['color_surface_1']}; margin: {t['space_1']} {t['space_2']}; }}")
    lines.append(f"QMenuBar {{ background-color: {t['color_mantle']}; color: {t['color_text']}; border-bottom: 1px solid {t['color_surface_0']}; }}")
    lines.append(f"QMenuBar::item {{ padding: {t['space_1']} {t['space_3']}; border-radius: {t['radius_sm']}; }}")
    lines.append(f"QMenuBar::item:selected {{ background-color: {t['color_surface_0']}; }}")
    lines.append(f"QStatusBar {{ background-color: {t['color_mantle']}; color: {t['color_text_subtle']}; border-top: 1px solid {t['color_surface_0']}; padding: 2px; }}")
    lines.append(f"QToolTip {{ background-color: {t['color_surface_0']}; color: {t['color_text']}; border: 1px solid {t['color_surface_1']}; border-radius: {t['radius_sm']}; padding: {t['space_1']} {t['space_2']}; }}")
    lines.append(f"QDialog {{ background-color: {t['color_base']}; }}")
    lines.append(f"QStackedWidget {{ background-color: {t['color_base']}; }}")
    lines.append(f"QFrame#sidebar {{ background-color: {t['color_mantle']}; border-right: 1px solid {t['color_surface_0']}; }}")
    lines.append(f"QFrame#topBar {{ background-color: {t['color_mantle']}; border-bottom: 1px solid {t['color_surface_0']}; }}")
    lines.append(f"QFrame#workspaceToolbar {{ background-color: {t['color_mantle']}; border-bottom: 1px solid {t['color_surface_0']}; }}")
    lines.append(f"QFrame#annotateToolbar {{ background-color: {t['color_mantle']}; border-bottom: 1px solid {t['color_surface_0']}; }}")
    lines.append(f"QFrame#navSeparator {{ background-color: {t['color_surface_1']}; max-width: 1px; }}")
    lines.append(f"QPushButton#navButton {{ background-color: transparent; border: none; border-radius: {t['radius_md']}; padding: {t['space_2']} {t['space_1']}; color: {t['color_text_subtle']}; font-size: {t['font_size_sm']}; min-height: 56px; min-width: 48px; text-align: center; }}")
    lines.append(f"QPushButton#navButton:hover {{ background-color: {t['color_surface_0']}; color: {t['color_text']}; }}")
    lines.append(f"QPushButton#navButton:checked {{ background-color: {t['color_surface_0']}; color: {t['color_primary']}; border-left: 3px solid {t['color_primary']}; }}")
    lines.append(f"QPushButton#iconButton {{ background-color: transparent; border: none; border-radius: {t['radius_md']}; padding: {t['space_1']} {t['space_2']}; min-width: 36px; min-height: 32px; }}")
    lines.append(f"QPushButton#iconButton:hover {{ background-color: {t['color_surface_0']}; }}")
    lines.append(f"QPushButton#iconButton:pressed {{ background-color: {t['color_surface_1']}; }}")
    lines.append(f"QScrollArea {{ background-color: {t['color_base']}; border: none; }}")
    return "\n".join(lines) + "\n"


def _build_toolbar_button_style(t):
    lines = []
    lines.append(f"QPushButton {{ padding: {t['space_1']} {t['space_2']}; border: 1px solid {t['color_surface_1']}; border-radius: {t['radius_sm']}; background-color: {t['color_surface_0']}; color: {t['color_text']}; }}")
    lines.append(f"QPushButton:hover {{ background-color: {t['color_surface_1']}; border-color: {t['color_primary']}; }}")
    lines.append(f"QPushButton:checked {{ background-color: {t['color_primary']}; color: {t['color_base']}; border-color: {t['color_primary']}; }}")
    lines.append(f"QPushButton:pressed {{ background-color: {t['color_primary_pressed']}; }}")
    return "\n".join(lines) + "\n"


def _build_start_button_style(t):
    lines = []
    lines.append(f"QPushButton {{ padding: {t['space_1']} {t['space_4']}; border: 1px solid {t['color_success']}; border-radius: {t['radius_md']}; background-color: {t['color_success']}; color: {t['color_base']}; font-weight: bold; }}")
    lines.append(f"QPushButton:hover {{ background-color: {t['color_success_hover']}; border-color: {t['color_success_hover']}; }}")
    lines.append(f"QPushButton:pressed {{ background-color: {t['color_success_pressed']}; border-color: {t['color_success_pressed']}; }}")
    lines.append(f"QPushButton:disabled {{ background-color: {t['color_surface_2']}; color: {t['color_text_muted']}; border-color: {t['color_surface_2']}; }}")
    return "\n".join(lines) + "\n"


def _build_stop_button_style(t):
    lines = []
    lines.append(f"QPushButton {{ padding: {t['space_1']} {t['space_4']}; border: 1px solid {t['color_danger']}; border-radius: {t['radius_md']}; background-color: {t['color_danger']}; color: {t['color_base']}; font-weight: bold; }}")
    lines.append(f"QPushButton:hover {{ background-color: {t['color_danger_hover']}; border-color: {t['color_danger_hover']}; }}")
    lines.append(f"QPushButton:pressed {{ background-color: {t['color_danger_pressed']}; border-color: {t['color_danger_pressed']}; }}")
    lines.append(f"QPushButton:disabled {{ background-color: {t['color_surface_2']}; color: {t['color_text_muted']}; border-color: {t['color_surface_2']}; }}")
    return "\n".join(lines) + "\n"


def _build_result_label_style(t):
    return f"QLabel {{ background-color: {t['color_crust']}; border: 1px solid {t['color_surface_1']}; border-radius: {t['radius_sm']}; }}\n"


def _build_group_box_style(t):
    lines = []
    lines.append(f"QGroupBox {{ font-weight: bold; border: 1px solid {t['color_surface_1']}; border-radius: {t['radius_sm']}; margin-top: {t['space_2']}; padding-top: {t['space_4']}; }}")
    lines.append(f"QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 5px; }}")
    return "\n".join(lines) + "\n"


DARK_STYLE = _build_style(DARK_TOKENS)
LIGHT_STYLE = _build_style(LIGHT_TOKENS)

_STYLE_CACHE = {"dark": DARK_STYLE, "light": LIGHT_STYLE}


def get_style(theme="dark"):
    return _STYLE_CACHE.get(theme, DARK_STYLE)
TOOLBAR_BUTTON_STYLE = _build_toolbar_button_style(DARK_TOKENS)
START_BUTTON_STYLE = _build_start_button_style(DARK_TOKENS)
STOP_BUTTON_STYLE = _build_stop_button_style(DARK_TOKENS)
RESULT_LABEL_STYLE = _build_result_label_style(DARK_TOKENS)
SCENE_BACKGROUND_STYLE = f"background-color: {DARK_TOKENS['color_crust']};"
GROUP_BOX_STYLE = _build_group_box_style(DARK_TOKENS)
INFO_LABEL_STYLE = f"QLabel {{ color: {DARK_TOKENS['color_text_subtle']}; font-size: {DARK_TOKENS['font_size_xs']}; }}"
