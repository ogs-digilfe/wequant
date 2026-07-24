# wequqnt/cli.py
import typer

app = typer.Typer(help="wequant CLI")

@app.command()
def get_app_name():
    """appの名前を表示する"""
    print("wequant")

@app.command()
def describe():
    """appの説明を表示する"""
    print("analysis tool for stock market")


## 最新データをdeliverサーバからdownloadする
from utilities.download_data import download_data
@app.command()
def dl_pq():
    """最新データをdeliverサーバからdownloadする"""
    print("Downloading latest data from deliver server...")
    # ここに実際のダウンロード処理を実装
    download_data()
    print("Download complete.")