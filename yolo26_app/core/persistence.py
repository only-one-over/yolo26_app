from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

from PyQt6.QtCore import QIODevice, QSaveFile


def write_json_atomic(path: Union[str, Path], data: Any) -> None:
    """Write JSON through QSaveFile so readers never see a partial document."""
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")

    save_file = QSaveFile(str(target_path))
    if not save_file.open(QIODevice.OpenModeFlag.WriteOnly):
        raise OSError(save_file.errorString())

    if save_file.write(payload) != len(payload):
        error = save_file.errorString()
        save_file.cancelWriting()
        raise OSError(error)

    if not save_file.commit():
        raise OSError(save_file.errorString())
