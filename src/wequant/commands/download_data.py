# pathの設定
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PJROOT_DIR = CURRENT_DIR.parents[2]
WORKSPACE_DIR = PJROOT_DIR.parent
DATA_DIR = PJROOT_DIR / "data"


# オブジェクトのインポート
from wequant.api import Client

# download可能なファイル
DOWNLOADABLE_FILES = [
    "creditbalance.parquet",
    "finance_quote.parquet",
    "kessan.parquet",
    "meigaralist.parquet",
    "nh225.parquet",
    "raw_pricelist.parquet",
    "reviced_pricelist.parquet",
    "shikiho_online.parquet",
    "sp500.parquet",
    "base_portfolio.parquet"
]

def download_data():
    """最新データをdeliverサーバからdownloadする"""
    print("Downloading latest data from deliver server...")
    client = Client()
    for f in DOWNLOADABLE_FILES:
        client.download(f)

# fp = DATA_DIR/f
# client.upload(fp)

if __name__ == "__main__":
    download_data()
