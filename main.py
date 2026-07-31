import sys
from pathlib import Path

# 代码核心在 code/ 文件夹下,用户更新项目只需替换 code/
sys.path.insert(0, str(Path(__file__).parent / "code"))

from yolo26_app.app import main


if __name__ == "__main__":
    sys.exit(main())
