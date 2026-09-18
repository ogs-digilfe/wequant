# wequqnt/cli.py
import typer

from wequant.flows.download_data import download_data_flow

app = typer.Typer(help="wequant CLI")

@app.command()
def get_app_name():
    """appの名前を表示する"""
    print("wequant")

@app.command()
def describe():
    """appの説明を表示する"""
    print("analysis tool for stock market")


@app.command()
def dl_pq():
    """最新データをdeliverサーバからdownloadする"""
    print("Downloading latest data from deliver server...")
    download_data_flow()
    print("Download complete.")
