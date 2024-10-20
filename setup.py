from pathlib import Path
import os

CURRENT_DIR = Path(os.path.abspath(__file__)).parent
PJ_DIR = CURRENT_DIR
WS_DIR = PJ_DIR.parent
CREDENTIALS_DIR = WS_DIR / "credentials"
CREDENTIALS_FNAME = "deliver.py"


import getpass, json

def input_user_info():
    username =           input("    user: ")
    password = getpass.getpass("password: ")
    
    return {
        "username": username,
        "password": password,
    }
    

def setup():
    dp = str(CREDENTIALS_DIR)
    fp = str(CREDENTIALS_DIR/CREDENTIALS_FNAME)

    # 認証情報ファイルが存在する場合は処理を終了
    if os.path.exists(fp):
        print(f'認証ファイルは既に存在します。--> 処理を終了')
        return
    
    user_dct = input_user_info()
    
    # dirがなければ作成
    if not os.path.exists(dp):
        os.makedirs(dp)
    
    # credential fileを作成
    with open(fp, "w", encoding="utf-8") as f:
        credential = json.dumps(user_dct, indent=4)
        content = f'DELIVER_USER = {credential}'
        f.write(content)
        
    
if __name__ == "__main__":
    setup()