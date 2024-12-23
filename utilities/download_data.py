# pathの設定
from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).parent
PJROOT_DIR = CURRENT_DIR.parent
LIB_DIR = PJROOT_DIR / "lib"
WORKSPACE_DIR = PJROOT_DIR.parent
DATA_DIR = PJROOT_DIR / "data"

sys.path.append(str(LIB_DIR))

# オブジェクトのインポート
from lib_api import Client

# download可能なファイル
DOWNLOADABLE_FILES = [
    "creditbalance.parquet",
    "finance_quote.parquet",
    "kessan.parquet",
    "meigaralist.parquet",
    "nh225.parquet",
    "raw_pricelist.parquet",
    "reviced_pricelist.parquet",
    "shikiho.parquet"
]

# fileのダウンロード
client = Client()

for f in DOWNLOADABLE_FILES:
    client.download(f)

# fp = DATA_DIR/f
# client.upload(fp)

