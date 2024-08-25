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
        data_dir = str(DATA_DIR)

        # ダウンロードディレクトリが存在しない場合は作成
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # pricelist関連以外のfileはサイズが小さいので全体をダウンロード
        if not filename in [
            "raw_pricelist.parquet",
            "reviced_pricelist.parquet"
        ]:
            self.download_wholefile(filename)
            return

        # filenameが存在しなかったら全体をダウンロード
        fp = str(DATA_DIR/filename)
        if not os.path.exists(fp):
            self.download_wholefile(filename)
            return

        # pricelist関連のファイルはサイズが大きいので、
        # fileが存在する場合は、差分のみダウンロードする
        SORCE_PL = PricelistPl(filename)
        last_date = SORCE_PL.df["p_key"].max()

        # GETリクエストでファイルをダウンロード
        route = "download-diff-pricelist-parquet"
        url = f'{BASE_URL}/{route}/?filename={filename}&last_date={last_date.strftime("%Y-%m-%d")}'

        response = requests.get(url, stream=True, headers=self.headers)
        print(response.headers)

        # temporary data保存dirがなければ作成
        tmp_data_dir = str(TMP_DATA_DIR)
        if not os.path.exists(tmp_data_dir):
            os.makedirs(tmp_data_dir)
        
        # 差分pricelistを一時ファイルとして保存
        fp = str(TMP_DATA_DIR/f'tmp_{filename}')
        if response.status_code == 200:
            with open(fp, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
        else:
            raise ValueError(f'statis_code={response.status_code}: "{filename}"のダウンロードに失敗しました。ファイル名を確認してください。')    

        # 元のファイルとダウンロードしてきた差分ファイルを読み込んで連結
        # ただし、差分が0の場合は、連結しない
        diff_df = PricelistPl(fp).df
        if diff_df.shape[0] == 0:
            print(f'{filename}はすでに最新データに更新済みです。最新データ: {last_date}')
        else:
            sd = diff_df["p_key"].min()
            ed = diff_df["p_key"].max()
            print(f'{filename}を更新しました。 更新差分: {sd}から{ed}のデータを追加')
            source_df = SORCE_PL.df  
            tmp_df = pl.concat([source_df, diff_df])
            concatted_df = tmp_df.sort([
                pl.col("mcode"),
                pl.col("p_key")
            ])
            
            # 連結したparquetファイルを更新データとして保存
            fp = str(DATA_DIR/filename)
            concatted_df.write_parquet(fp)
        
        # tmpfileを削除
        fp = str(TMP_DATA_DIR/"*")
        files = glob.glob(fp)
        for f in files:
            try:
                os.remove(f)
            except Exception as e:
                print(f'ファイル{f}の削除に失敗しました。失敗理由：{e}')
        
        # deliver-serverのtempfileを削除        
        route = "delete-tmp-data"
        filename = self._get_download_filename_on_server(response)
        url = f'{BASE_URL}/{route}/?filename={filename}'
        requests.get(url, headers=self.headers)
                
        
        # for debug
        # self.download_wholefile(filename)
        print(response.status_code)
        print(f'filename={filename}')
        
        
                
    # ファイル全体をダウンロード
    def download_wholefile(self, filename):
        route = "download-raw-parquet"
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
        
        
        
