# pathの設定
from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).parent
PJROOT_DIR = CURRENT_DIR.parent
WORKSPACE_DIR = PJROOT_DIR.parent
DATA_DIR = PJROOT_DIR / "data"

sys.path.append(str(PJROOT_DIR))

# オブジェクトのインポート
from lib import Client

# download可能なファイル
DOWNLOADABLE_FILES = [
    "raw_pricelist.parquet",
    "reviced_pricelist.parquet"
]

# fileのダウンロード
client = Client()

for f in DOWNLOADABLE_FILES:
    client.download(f)

fp = DATA_DIR/f
client.upload(fp)

