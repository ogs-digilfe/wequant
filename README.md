# wequant

株式市場分析ツールです。Pythonパッケージは`src/wequant/`に配置し、
コマンドラインインターフェースとして`wq`を提供します。

## 開発環境のセットアップ

Python環境と依存関係は`uv`で管理します。Python 3.12と`uv`を利用できる
環境で、リポジトリルートから次のコマンドを実行してください。

```bash
uv sync
```

`.python-version`に基づいてPython 3.12が選択され、`uv.lock`に従って
依存関係が`.venv/`へインストールされます。`.venv/`を手動で有効化せず、
コマンドは`uv run`経由で実行できます。

依存関係は`pyproject.toml`へ追加し、`uv.lock`も同じ変更に含めます。
`requirements.txt`は使用しません。

## ローカル設定

Deliver APIを利用するコマンドには、ローカルな環境設定が必要です。

1. `.env.sample`を`.env`へコピーします。
2. `.env`内のプレースホルダーを手元の接続情報へ置き換えます。

```dotenv
WEQUANT_DELIVER_BASE_URL=https://example.invalid
WEQUANT_DELIVER_USERNAME=replace-with-your-username
WEQUANT_DELIVER_PASSWORD=replace-with-your-password
```

`.env`はGit管理対象外です。設定済みのプロセス環境変数がある場合は、
その値が`.env`より優先されます。秘密値をログ、Issue、チャット、
Notebook出力へ記載しないでください。

## CLI

利用できるコマンドは、リポジトリルートから次のコマンドで確認できます。

```bash
uv run wq --help
```

アプリケーション名と説明は、それぞれ次のコマンドで表示できます。

```bash
uv run wq get-app-name
uv run wq describe
```

## Parquetファイルのダウンロード

リポジトリルートから次のコマンドを実行します。

```bash
uv run wq dl-pq
```

Deliver APIから取得したParquetファイルは、Git管理対象外の`data/`へ
保存されます。同名のファイルがすでに存在する場合は上書きされます。

## テスト

外部通信を行わない単体テストは、リポジトリルートから次のコマンドで
実行します。

```bash
uv run python -m unittest discover -s tests -v
```
