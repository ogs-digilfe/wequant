# CLIリファレンス

## 概要

`wequant`はTyperで実装した`wq`コマンドを提供します。リポジトリルートで
`uv run`を介して実行します。

```bash
uv run wq --help
uv run wq <command> --help
```

## `get-app-name`

アプリケーション名を表示します。

```bash
uv run wq get-app-name
```

出力例:

```text
wequant
```

## `describe`

アプリケーションの短い説明を表示します。

```bash
uv run wq describe
```

出力例:

```text
analysis tool for stock market
```

## `dl-pq`

Deliver APIから管理対象のParquetファイルをダウンロードします。

```bash
uv run wq dl-pq
```

このコマンドは外部通信を行います。取得ファイルはリポジトリルートの`data/`へ保存され、
同名ファイルがある場合は上書きされます。実行前に接続先と保存対象を確認してください。

正常終了時には、進行状況、各ファイルの保存結果、完了メッセージが表示されます。

```text
Downloading latest data from deliver server...
...
Download complete.
```

### 必要な設定

| 環境変数 | 用途 |
| --- | --- |
| `WEQUANT_DELIVER_BASE_URL` | Deliver APIのベースURL |
| `WEQUANT_DELIVER_USERNAME` | 認証ユーザー名 |
| `WEQUANT_DELIVER_PASSWORD` | 認証パスワード |

リポジトリルートの`.env`、またはプロセス環境変数に設定します。両方に値がある場合は
プロセス環境変数が優先されます。設定値や認証後のトークンを出力しないでください。

### ダウンロード対象

- `creditbalance.parquet`
- `finance_quote.parquet`
- `kessan.parquet`
- `meigaralist.parquet`
- `nh225.parquet`
- `raw_pricelist.parquet`
- `reviced_pricelist.parquet`
- `shikiho_online.parquet`
- `sp500.parquet`
- `base_portfolio.parquet`

この一覧は`src/wequant/data_files.py`の`DOWNLOADABLE_FILES`で管理されています。

### エラー時の確認

- 必須設定が不足している場合は、エラーに示された環境変数名を確認する。
- 認証に失敗した場合は、ベースURL、ユーザー名、パスワードを安全な方法で確認する。
- ダウンロードに失敗した場合は、接続先とサーバー上の対象ファイルを確認する。
- 保存できない場合は、リポジトリルートと`data/`の書き込み権限を確認する。

エラーや問い合わせ内容へ、パスワード、アクセストークン、Cookieなどを含めないでください。
