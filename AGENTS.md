# AGENTS.md

## 目的

このリポジトリでは、株式市場分析ツール`wequant`を段階的にリファクタリングする。
既存の動作を維持しながら、環境・設定・パッケージ構成・テストを小さな単位で改善する。

リファクタリング開始時点の詳細は`docs/refactoring-baseline.md`を参照すること。

## 作業ルート

- このファイルがある`wequant/`をリポジトリルートおよび作業ディレクトリとして扱う。
- パスは原則として、このリポジトリルートからの相対パスで記述する。
- リポジトリ外のファイルは、明示的な依頼なしに作成・変更・削除しない。

## 現在の構成

```text
wequant/
├── src/
│   └── wequant/
│       ├── __init__.py
│       ├── api.py
│       ├── cli.py
│       ├── config.py
│       ├── data_files.py
│       ├── data_loading.py
│       ├── data_processing.py
│       ├── graph_processing.py
│       └── commands/
│           ├── __init__.py
│           └── download_data.py
├── utilities/
│   └── create_and_save_reviced_pricelist_parquet.py
├── tests/
│   ├── test_api.py
│   ├── test_cli.py
│   ├── test_config.py
│   └── test_download_data.py
├── notebooks/
├── my_notebooks/
├── data/                 # Git管理対象外
├── .venv/                # Git管理対象外
├── docs/
│   ├── ARCHITECTURE.md
│   ├── REFERENCE.md
│   └── refactoring-baseline.md
├── .env.sample
├── .python-version
├── pyproject.toml
├── uv.lock
└── .gitignore
```

Pythonパッケージは`src/wequant/`に配置する。パッケージ外の補助スクリプトは
`utilities/`に配置する。詳しい配置方針は`docs/ARCHITECTURE.md`を参照すること。

## 管理ドキュメント

- `README.md`: セットアップ、ビルド、テストの入口と、他の文書への案内を記載する。
- `docs/ARCHITECTURE.md`: コード配置、モジュールの責務、コード作成の基本設計を記載する。
- `docs/REFERENCE.md`: CLIの仕様、設定、利用例、実行時の注意事項を記載する。
- `AGENTS.md`: エージェント向けの作業規約、安全上の注意、検証方針を記載する。
- `docs/refactoring-baseline.md`: リファクタリング開始時点の記録として保持し、原則として変更しない。

外部ホスト間の連携を含むシステム全体の設計は、専用文書が追加されるまで
`docs/ARCHITECTURE.md`の対象に含めない。

## 編集前の合意

- ファイルやコードを編集する前に、目的、設計、主な変更対象をユーザーへ提案する。
- 提案がユーザーに受け入れられてから編集を開始する。
- 合意した範囲を超える変更が必要になった場合は、理由と影響を提示し、再度合意を得る。
- 読み取り、調査、設計案の作成は編集に含まれないが、秘密情報やローカルデータは読み取らない。

## 開発環境

- Python 3.12以上を前提とする。
- 標準のPythonバージョンは`.python-version`で3.12に指定する。
- 依存関係は`pyproject.toml`の`[project].dependencies`で管理する。
- Python環境とロックファイルは`uv`と`uv.lock`で管理する。
- `uv sync`でリポジトリ直下の`.venv/`へ同期する。
- `requirements.txt`は使用せず、`uv.lock`をGit管理する。
- `uv.lock`は直接編集せず、依存関係変更時にuvで更新する。

## 現在の実行経路

Parquetファイルのダウンロードは、リポジトリルートから次のコマンドで実行する。

```bash
uv run wq dl-pq
```

この処理は外部通信を行い、`data/`内の同名Parquetファイルを上書きする。
ユーザーによる実環境でのダウンロード成功を確認済みである。

CLIは`src/wequant/cli.py`でTyperアプリケーションとして定義し、
`pyproject.toml`の`wq`エントリーポイントから実行する。コマンド一覧は
`uv run wq --help`で確認する。詳しい仕様と利用例は`docs/REFERENCE.md`を参照する。

## ローカル設定

- ローカル設定はリポジトリ直下の`.env`から読み込む。
- `.env`はGit管理対象でなく、Git管理する項目名の例は`.env.sample`に記載する。
- OSの環境変数を`.env`より優先する。
- Deliver API設定は`WEQUANT_DELIVER_BASE_URL`、`WEQUANT_DELIVER_USERNAME`、
  `WEQUANT_DELIVER_PASSWORD`を使用する。
- 設定不足のエラーやログには秘密値を含めない。

## 安全上の注意

- 認証情報、アクセストークン、パスワード、Cookieなどの秘密値を読み取ったり、表示・ログ出力・コミットしたりしない。
- `.env`や認証情報ファイルを追加する場合は、先に`.gitignore`の対象であることを確認する。
- `data/`内のParquetファイルはローカルデータとして扱い、明示的な依頼なしに更新・削除・コミットしない。
- `uv run wq dl-pq`の実行は、外部通信とローカルデータの上書きを伴うため自動実行しない。
- 外部通信を伴う確認は、接続先、対象操作、秘密情報の扱いを確認し、ユーザーの了承を得てから実行する。
- Notebookを一括実行しない。実行が必要な場合は、対象Notebookと副作用を先に確認する。

## 変更方針

- 変更前に`git status --short`を確認し、既存の未コミット変更を保持する。
- 1回の変更では目的を1つに絞り、機能変更、依存関係変更、ファイル移動を可能な限り混在させない。
- 現行動作を確認できない箇所は、推測で「修正」せず、既知の挙動と仮定を記録する。
- 大規模な移動や置換の前に、参照元、import経路、CLI、Notebookへの影響を検索する。
- 互換性を壊す変更やファイル削除は、影響範囲を示してから実施する。
- 生成物、キャッシュ、仮想環境、秘密情報、ローカルデータをGit管理へ追加しない。
- コードを編集した際は、管理ドキュメントに修正が必要な箇所がないか確認し、
  該当箇所を同じ変更内で修正する。
- 構成、標準コマンド、利用方法、CLI仕様を変更した場合は、対応する管理ドキュメントを
  同じ変更内で修正する。
- ファイルやコードを編集した場合は、完了報告で編集・追加・削除したファイルを
  ユーザーへ明示する。

## 検証方針

外部通信なしで実行できる単体テストが`tests/`にある。

```bash
uv run python -m unittest discover -s tests -v
```

- CLI、ダウンロード処理、Deliver API認証のテストでは、外部通信部分をモックする。
- 変更箇所に対して、外部通信を行わない最小限の構文確認・静的確認を優先する。
- 新しいテストは`tests/`に追加し、ネットワークと実データに依存しないようにする。
- 外部サービスはモックまたは依存性注入で置き換える。
- テストで秘密値やアクセストークンを標準出力しない。
- 実行していない検証や、環境不足で実行できなかった検証は、完了報告に明記する。

## 段階的なリファクタリング計画

1. [x] `AGENTS.md`を整備する。
2. [x] `.env`と環境変数を使用する安全な設定方式へ統一する。
3. [x] `uv`でPython環境と依存関係を再現可能にする。
4. [x] Pythonパッケージを`src`レイアウトへ移行する。
5. [x] 外部通信なしで動く自動テストを拡充する。
6. [x] 実際の構成と手順に合わせて`README.md`と本ファイルを更新する。

計画の各段階は、原則として独立して確認・コミットできる状態にする。

## ドキュメントの更新

現時点で`docs/`配下の管理ドキュメントは次の2つとする。

- `docs/ARCHITECTURE.md`
- `docs/REFERENCE.md`

- `docs/refactoring-baseline.md`は開始時点の記録であり、原則として変更しない。
- 最新のセットアップ、ビルド、テストの入口は`README.md`に記載する。
- コード配置と実装設計は`docs/ARCHITECTURE.md`に記載する。
- CLIの仕様と利用例は`docs/REFERENCE.md`に記載する。
- エージェント向けの作業規約と検証方針は本ファイルに記載する。
- 構成や標準コマンドが変わった場合は、実装と同じ変更内で該当ドキュメントを更新する。
