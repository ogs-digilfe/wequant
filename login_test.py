from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).parent
PJROOT_DIR = CURRENT_DIR
WORKSPACE_DIR = PJROOT_DIR.parent

CREDDENTIALS_DIR = WORKSPACE_DIR / "credentials"

sys.path.append(str(CREDDENTIALS_DIR))

# import and set global objects
import requests


from deliver import DELIVER_USER

SERVER = "192.168.0.154"
PORT = "8080"
BASE_URL = f'http://{SERVER}:{PORT}'

token_response = requests.post(f'{BASE_URL}/token', data=DELIVER_USER)

if token_response.status_code == 200:
    token = token_response.json()["access_token"]
    print(f"Access token: {token}")

    # ヘッダーにBearerトークンを追加して、/users/me2 にリクエストを送信
    headers = {
        "Authorization": f"Bearer {token}"
    }
    response = requests.get(f"{BASE_URL}/logintest", headers=headers)

    # レスポンスの内容を確認
    if response.status_code == 200:
        print("User Info:", response.json())
    else:
        print(f"Failed to access /users/me2: {response.status_code}, {response.text}")


else:
    print(f"Failed to authenticate: {token_response.status_code}, {token_response.text}")
