# 初期設定
content = '''
SERVER_HOST = "192.168.0.154"
PORT = "8080"
USERNAME = "your_note_user"
PASSWORD = "secret"
'''

# pathのセット
from pathlib import Path

CURRENT_DIR = Path(__file__).parent
PJROOT_DIR = CURRENT_DIR.parent
WORKSPACE_DIR = PJROOT_DIR.parent
SETTINGS_DIR = WORKSPACE_DIR / "settings"

# import objects
import os

# body
# 設定ファイルディレクトリが存在しない場合は作成
settings_dir = str(SETTINGS_DIR)
if not os.path.exists(settings_dir):
    os.makedirs(settings_dir)

# 設定ファイルが存在する場合は例外を発生させて処理を中断
fp = str(SETTINGS_DIR/"__init__.py")
if os.path.exists(fp):
    raise FileExistsError(f'\n設定ファイルはすでに存在します。\n\
設定を変更する場合は、直接設定ファイルを編集してください。\n\n\
設定ファイル: {fp}')

# 初期設定ファイルを作成

with open(fp, "w") as f:
    f.write(content)