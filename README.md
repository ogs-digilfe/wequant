# wequant

`wequant`は、株式投資分析のために収集したデータを、加工・分析・可視化するPythonプロジェクトです。
収集したデータは、データサーバ(deliver)にparquet型式で保存されており、各データは定期的に自動更新されています。

本プロジェクトのPythonパッケージは`src/wequant/`に配置し、CLIとして`wq`を提供します。

## 必要な環境

- Python 3.12以上
- [uv](https://docs.astral.sh/uv/)

標準のPythonバージョンは`.python-version`、依存関係は`pyproject.toml`と
`uv.lock`で管理します。

## セットアップ

リポジトリルート（この`README.md`があるディレクトリ）で実行します。

```bash
uv sync
```

`uv`は`uv.lock`に従い、リポジトリ直下の`.venv/`へ環境を作成します。
仮想環境を手動で有効化せず、以降のコマンドは`uv run`経由で実行できます。

Deliver APIを使用する場合は、設定例をコピーして`.env`を作成します。

```bash
cp .env.sample .env
```

`.env`のプレースホルダーを実際の接続情報へ置き換えてください。

```dotenv
WEQUANT_DELIVER_BASE_URL=https://example.invalid
WEQUANT_DELIVER_USERNAME=replace-with-your-username
WEQUANT_DELIVER_PASSWORD=replace-with-your-password
```

`.env`はGit管理対象外です。同名のプロセス環境変数がある場合は、その値が
`.env`より優先されます。秘密値をログ、Issue、チャット、Notebook出力へ
記載しないでください。

## ビルド

配布用のwheelとsource distributionを作成します。

```bash
uv build
```

成果物はGit管理対象外の`dist/`に出力されます。

## 動作確認

```bash
uv run wq --help
uv run python -m unittest discover -s tests -v
```

単体テストは外部通信を行いません。

## ドキュメント

- [アーキテクチャ](docs/ARCHITECTURE.md): コード配置、各モジュールの責務、実装時の基本設計
- [CLIリファレンス](docs/REFERENCE.md): `wq`コマンドの仕様、設定、利用例
- [リファクタリング開始時点の記録](docs/refactoring-baseline.md): 現在の構成へ移行する前のスナップショット
- [エージェント向け作業規約](AGENTS.md): AIエージェントが変更を行う際の規則と検証方針

外部ホストを含むシステム全体の設計は、今後別途整理します。
