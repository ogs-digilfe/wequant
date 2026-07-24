# Refactoring baseline

## この文書について

この文書は、段階的なリファクタリングを開始する前の状態を記録するためのスナップショットである。

- 記録日: 2026-07-24
- Gitリポジトリ: `wequant`
- ブランチ: `feature/refuctor-2026-0724`
- 記録時のコミット: `2c8bd055d2060d4bf51ef59c7e257e9498222ae8`
- コミットメッセージ: `revice`

リファクタリング後の最新情報は`README.md`と`AGENTS.md`へ記載する。この文書は開始時点の記録として原則変更しない。

## リファクタリングの目的

- Python環境と依存関係をuvで再現できるようにする
- 認証情報と接続設定を安全かつ一貫した方法で読み込めるようにする
- Pythonパッケージを`src`レイアウトへ移行する
- 人とAIエージェントが同じ手順とルールで作業できるようにする
- 変更前後を確認できる小さな単位で進める

## ローカル環境

- Python: 3.12.3
- uv: 未インストール（`uv: command not found`）
- `pyproject.toml`の要求Python: 3.12以上

## 現在の主な構成

```text
wequant/
├── __init__.py
├── cli.py
├── lib/
│   ├── lib_api.py
│   └── lib_dataprocess.py
├── utilities/
├── notebooks/
├── my_notebooks/
├── data/                 # .gitignore対象
├── pyproject.toml
├── requirements.txt
├── setup.py
└── .gitignore
```

Pythonコードはリポジトリ直下、`lib/`、`utilities/`に分散している。`src/`はまだ存在しない。

## パッケージと依存関係

`requirements.txt`には以下がバージョン指定なしで記載されている。

- requests
- pandas
- polars-lts-cpu
- pyarrow
- pydantic
- jupyterlab
- plotly
- typer

`pyproject.toml`の状態は以下のとおり。

- ビルドバックエンド: setuptools
- プロジェクト名: `wequant`
- バージョン: `0.1.0`
- 依存関係: `pandas>=2.2.3`のみ
- CLI: `wq = "wequant.cli:app"`
- setuptoolsの検索対象: `wequant`

`requirements.txt`と`pyproject.toml`の依存関係は一致していない。現在のソース配置とパッケージ検出設定にも不整合の可能性がある。

`setup.py`はビルド設定ではなく、対話形式で認証情報ファイルを作るスクリプトとして使用されている。

## コードから確認できる実行経路

`cli.py`はTyperアプリケーションで、以下のコマンドを持つ。

- `get-app-name`: `wequant`を表示
- `describe`: アプリケーションの説明を表示
- `dl-pq`: deliverサーバーからデータをダウンロード

`dl-pq`は`utilities.download_data`を経由して`lib/lib_api.py`の`Client`を利用する。

`utilities/download_data.py`はimport時に`Client()`を生成するため、importだけでも認証用HTTPリクエストが発生する可能性がある。ダウンロード先の`data/`はGit管理対象外である。

`notebooks/`には分析、スクリーニング、動作確認などのNotebookがある。正式な起動方法とimport方法は未確認。

## 認証情報と接続設定

秘密値そのものは調査していない。

- リポジトリの一段上に`credentials/`が存在する
- リポジトリ直下に`.env`は存在しない
- `.gitignore`に`.env`の明示的な除外設定がない
- `setup.py`は一段上の`credentials/deliver.py`を作成する
- `login_test.py`などは`credentials/deliver.py`を参照する
- `lib/lib_api.py`は`settings_wequant`から接続・認証設定を読む
- `utilities/setup.py`は一段上の`settings_wequant/__init__.py`を作成する
- 一部のスクリプトにサーバーアドレスが直接記述されている

設定の読み込み方法が`credentials/deliver.py`と`settings_wequant`の2系統に分かれている。

## テストと動作確認

独立した`tests/`ディレクトリとpytest設定は確認できない。`*_test.py`は存在するが、以下の副作用があるため一般的な単体テストとして一括実行しない。

- 外部サーバーへHTTPリクエストを送る
- ファイルをアップロードまたはダウンロードする可能性がある
- アクセストークンを標準出力へ表示する箇所がある

記録時点では外部サーバーへの接続確認を実施していない。

## 既知の課題

- uvと`uv.lock`が未導入
- 依存関係の定義が一致していない
- `sys.path.append()`に依存したimportがある
- Pythonコードが複数ディレクトリに分散している
- 認証情報の読み込み方が統一されていない
- `.env`のignore設定がない
- import時に外部通信が発生する可能性がある
- アクセストークンを表示するコードがある
- 外部通信なしで実行できる自動テストがない
- `myapp.egg-info/`と`wequant.egg-info/`がGit管理されている
- `README.md`に利用方法が記載されていない

## 段階的な計画

1. 初期版の`AGENTS.md`を作成する
2. `.env`と環境変数を使用する設定方式へ変更する
3. uvで環境と依存関係を管理できるようにする
4. Pythonパッケージを`src`レイアウトへ移行する
5. `README.md`と`AGENTS.md`を更新する

各段階は原則として別コミットにし、機能変更、依存関係変更、ファイル移動を可能な限り混在させない。

## リファクタリング前の確認項目

- [ ] 正式なCLIの起動方法を確認する
- [ ] `get-app-name`と`describe`が動作する
- [ ] 正式なNotebookの起動方法を確認する
- [ ] 主要なNotebookから必要なモジュールをimportできる
- [ ] 認証が成功する
- [ ] deliverサーバーからデータをダウンロードできる
- [ ] 必要であればdeliverサーバーへデータをアップロードできる

外部通信を伴う確認は、接続先、実行対象、秘密情報の扱いを確認してから実施する。
