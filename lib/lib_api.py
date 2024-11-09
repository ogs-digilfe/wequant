# pathの設定
from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).parent
PJROOT_DIR = CURRENT_DIR.parent
WORKSPACE_DIR = PJROOT_DIR.parent
DATA_DIR = PJROOT_DIR / "data"
TMP_DATA_DIR = PJROOT_DIR / "data" / "tmp_data"
LIB_DIR = PJROOT_DIR / "lib"

sys.path.append(str(WORKSPACE_DIR))
sys.path.append(str(LIB_DIR))

# import objects
import requests, os, glob, re
from settings_wequant import SERVER_HOST, PORT, USERNAME, PASSWORD
from lib_dataprocess import PricelistPl
import polars as pl

BASE_URL = f'http://{SERVER_HOST}:{PORT}'

LOGIN_DATA = {
    "username": USERNAME,
    "password": PASSWORD
    # "password": "secret"
}
    
# ログインしてtokenを取得
class Client():
    def __init__(self):
        url = f'{BASE_URL}/token'
        token_response = requests.post(url, data=LOGIN_DATA)
        if token_response.status_code == 200:
            token = token_response.json()["access_token"]

            self.headers = {
                "Authorization": f"Bearer {token}"
            }
        else:
            print(token_response)
            raise ValueError(f'status_code=401: ユーザー"{LOGIN_DATA["username"]}"の認証に失敗しました')
    
    # 差分ダウンロード
    def download(self, filename):
        route = "download"
        data_dir = str(DATA_DIR)

        # ダウンロードディレクトリが存在しない場合は作成
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # GETリクエストでファイルをダウンロード
        fp = str(DATA_DIR/filename)
        url = f'{BASE_URL}/{route}/?filename={filename}'

        # コンストラクタで取得したTokenで作成したself.headersをheadersにセットして
        # get requestを送信
        response = requests.get(url, stream=True, headers=self.headers)

        # レスポンスが成功(status_code=200)したらファイルを所定のフォルダに保存
        if response.status_code == 200:
            with open(fp, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f'サーバから"{filename}"をダウンロードし、"{data_dir}"に保存しました')
        else:
            raise ValueError(f'statis_code={response.status_code}: "{filename}"のダウンロードに失敗しました。ファイル名を確認してください。')    

    # データ管理者用
    def upload(self, fp):
        fp = str(fp)
        
        route = "upload-stockdb-parquet-data"
        url = f'{BASE_URL}/{route}'
        
        # upload fileが存在しなければ、raise
        if not os.path.exists(fp):
            raise ValueError(f'"{fp}"が存在しません。アップロードファイルのパスを確認してください。')
        
        with open(fp, "rb") as f:
            files = {"file": f}
            response = requests.post(url, headers=self.headers, files=files)
            
            # レスポンスを出力
            if response.status_code == 200:
                print(f'status_code: {response.status_code} ファイル"{fp}"は"{SERVER_HOST}"にアップロードされました')
            else:
                raise ValueError(f'status_code: {response.status_code}')
    
    # downloadしたファイルのサーバ上のファイル名をhttp responseから取得する
    def _get_download_filename_on_server(self, response):
        content_disposition = response.headers["content-disposition"]
        
        extracted = re.search(r'filename="(.+?)"', content_disposition)
        
        # extractedの()内のみreturn
        return extracted.group(1)
        
        
        
