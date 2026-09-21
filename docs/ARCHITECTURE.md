# アーキテクチャ

## この文書の対象

この文書では、`wequant`リポジトリ内のコード配置と、コードを追加・変更する際の
基本設計を説明します。Deliver APIなど、複数ホストが連携するシステム全体の設計は
現時点の対象に含めません。

## ディレクトリ構成

```text
wequant/
├── src/wequant/            # Pythonパッケージ
│   ├── commands/           # 既存の互換用ユースケース
│   ├── flows/              # Taskを組み合わせる処理
│   ├── tasks/              # 読み込み・加工などの処理単位
│   ├── api.py              # Deliver APIとの通信
│   ├── cli.py              # TyperによるCLI定義
│   ├── config.py           # 環境変数と.envの読み込み
│   ├── data_files.py       # 管理対象データファイルの定義
│   ├── data_loading.py     # データパスの検証と読み込み
│   ├── data_processing.py  # 株式データの加工・分析
│   └── graph_processing.py # グラフの生成
├── tests/                  # 外部通信や実データに依存しない単体テスト
├── utilities/              # パッケージ外の補助スクリプト
├── notebooks/              # 分析・検証用Notebook
├── my_notebooks/           # 個人用Notebook
├── data/                   # ローカルデータ（Git管理対象外）
├── docs/                   # プロジェクト文書
├── pyproject.toml          # パッケージ情報と依存関係
└── uv.lock                 # uvが管理する依存関係のロック
```

リポジトリルートは、この文書の一つ上にある`wequant/`です。親ディレクトリは
リポジトリルートとして扱いません。

## モジュールの責務

### CLIとコマンド

`cli.py`はTyperアプリケーションと公開コマンドを定義します。CLI固有の引数解釈や
メッセージ表示にとどめ、まとまった処理は`flows/`以下へ委譲します。

`flows/download_data.py`は管理対象ファイルを列挙し、`tasks/download_file.py`の
Taskを通じてAPIクライアントへ取得を依頼します。

`flows/quarterly_valuation.py`は四半期銘柄評価の読み込みと加工を組み立て、
DataFrameを返します。`tasks/quarterly_valuation.py`の読み込みTaskは
`KessanPl.from_file()`、`FinancequotePl.from_file()`、
`PricelistPl.from_file("reviced_pricelist.parquet")`、`MeigaralistPl.from_file()`、
`IndexPricelistPl.from_file()`（nh225）を使用します。
加工Taskは、`KessanPl.get_quarterly_valuation`と
`FinancequotePl.get_price_adjusted_valuations`の結果を銘柄コードで左結合し、
銘柄名の結合、`codes="all"`または整数リストによる絞り込み、`sls`・`grsl`・選択利益`prft`の上下限による絞り込み、出力13列の選択とソートを行います。
`KessanPl.get_quarterly_performance`には選択済み決算と分割調整済み株価の
DataFrameと指数のDataFrameを渡し、今回・次回の発表翌取引日の始値による騰落率`perf`（%）と、
同じ取引日の指数始値による`bm`（%）を結合します。指数は日付の完全一致で結合します。
指数引数を省略した既存の加工メソッドはcode/perfを返し、加工Taskではbmをnullにします。
次回は次の決算期の最初の四半期実績発表で、評価日より後の履歴も使用します。
入力の状態は変更せず、次回発表や有効な始値がない場合はnullを返します。
範囲フィルタは最新決算の選択後に丸め前の値へ適用します。
`profit`はCLIからFlow・加工Taskを経由して`KessanPl.get_quarterly_valuation`へ渡します。
既定は営業利益（operating）、経常利益（ordinary）も選択できます。`prft`に選択利益、直後の`pr`に当期利益率（利益/売上×100、%）を返します。
`pr`は売上が正で売上・利益・結果が有限の場合に計算し、赤字・0を保持、それ以外はnullです。
加工Taskは最後に、抽出・ソート済みの一覧へ`start_row`・`end_row`の行範囲を適用します。加工メソッドは状態を変更せず、他データを
必要とする場合はDataFrameを引数で受け取ります。表示はCLIの責務です。
`--portfolio`指定時は、Flowから同じTaskモジュールの
`load_portfolio_codes(valuation_date)`を呼び出します。このTaskは
`PortfolioManager.from_file().get_individual_stocks(...)`を使い、評価日以前の
ポートフォリオ全体の最新日から個別株コードを返します。整数に変換できないコードは
件数を標準エラーへ通知して除外します。Flowは取得コードと`codes`の共通部分を
既存の加工Taskへ渡します。フラグ未指定時はポートフォリオを読み込みません。

Task・Flowは通常のPython関数であり、実行基盤の追加依存はありません。

### 設定と外部通信

`config.py`はリポジトリルートの`.env`とプロセス環境変数からDeliver API設定を
読み込みます。プロセス環境変数を優先し、不足している変数名のみをエラーへ含め、
秘密値は表示しません。

`api.py`はDeliver APIとのHTTP通信と、取得したファイルの`data/`への保存を担当します。
外部通信を行うコードはこの境界へ集約し、呼び出し側からモックできる状態を保ちます。

### データ定義、読み込み、加工

`data_files.py`は、wequantがダウンロード・管理するParquetファイル名を一元管理します。
対象を変更するときは、ダウンロード、読み込み、CLIリファレンスへの影響を確認します。

`data_loading.py`はデータ保存先の解決、管理対象ファイル名の検証、Parquetの読み込みを
担当し、分析ロジックからパス処理を分離します。

`data_processing.py`はPolarsを中心とした株式データの加工・分析を担当します。
`graph_processing.py`は加工済みデータからPlotlyのグラフを生成します。可能な範囲で、
データ読み込みや加工を既存モジュールへ委譲します。

### 補助コードとNotebook

再利用するアプリケーションコードは`src/wequant/`へ、パッケージとして公開しない
単発の補助処理は`utilities/`へ配置します。

Notebookは分析や試行の利用者であり、再利用する処理の実装場所にはしません。
共通処理はPythonパッケージへ移してNotebookからimportします。既存Notebookは現在の
APIを参照しているため、公開名や呼び出し方を変える前に参照箇所を検索します。

## 処理の流れ

```text
wq dl-pq
  → cli.py
  → flows/download_data.py
  → tasks/download_file.py
  → config.py（接続設定の読み込み）
  → api.py（認証、HTTP通信、ファイル保存）
  → data/（同名ファイルを上書き）
```

ローカルデータを使う分析処理は、原則として次の依存方向にします。

```text
Notebookまたは呼び出し元
  → graph_processing.py
  → data_processing.py
  → data_loading.py
  → data_files.py / data/
```

## コード作成の基本設計

- Pythonパッケージのコードは`src/wequant/`へ配置し、`wequant.*`の絶対importを使用する。
- モジュールには一つの明確な責務を持たせ、CLI、通信、読み込み、加工、描画を分離する。
- パスは`pathlib.Path`で扱い、リポジトリルートから解決する。
- 依存関係は`pyproject.toml`で管理し、変更時は`uv`で`uv.lock`も更新する。
- 認証情報をソースコードへ埋め込まず、`.env`または環境変数から読み込む。
- 外部通信や実データに依存する処理は境界を明確にし、テストではモックまたは依存性注入で置き換える。
- 機能変更には、ネットワークと実データに依存しないテストを追加または更新する。
- 既存APIやNotebookへ影響する変更は、参照元と既知の挙動を確認してから行う。
- コード配置、設定、CLIなどを変えた場合は、同じ変更内で関連文書を更新する。

## データ処理クラスの設計

設計規則は本節を正本とします。作成・変更の作業手順は、リポジトリ内の
[wequant-data-processing skill](../.agents/skills/wequant-data-processing/SKILL.md)に
まとめます。以下では、現在の実装から確認できる挙動と、新規実装・段階的な改善に
適用する方針を区別します。方針への適合を理由に、既存APIを一括変更しません。

### 現在の実装から確認できる構成と慣習

`src/wequant/data_processing.py`のデータ処理クラスは、PolarsのDataFrameを
`self.df`に保持し、データの種類に固有の加工・分析操作を提供します。

| クラス | 主な対象 | `from_file()`の既定ファイル |
| --- | --- | --- |
| `CreditbalancePl` | 信用残高 | `creditbalance.parquet` |
| `FinancequotePl` | 日々の財務指標 | `finance_quote.parquet` |
| `IndexPricelistPl` | 株価指数 | `nh225.parquet` |
| `PricelistPl` | 未調整・調整済み株価 | `reviced_pricelist.parquet` |
| `KessanPl` | 決算実績・予想 | `kessan.parquet` |
| `MeigaralistPl` | 銘柄一覧 | `meigaralist.parquet` |
| `ShikihoOnlinePl` | 四季報データ | `shikiho_online.parquet` |
| `PortfolioManager` | ポートフォリオ | `base_portfolio.parquet` |

同じモジュールには、汎用の集計・統計を扱う`CommonPl`、`CalcStatistics`もあります。
これらには既定ファイルがなく、今回のファイル読み込み入口の統一対象には含めません。

- 上記8クラスは、DataFrameを渡す`Class(df)`・`Class(df=df)`と、ファイルを読み込む
  `Class.from_file(fp)`・`Class.from_file()`を共通の入口として提供します。
  従来のコンストラクタによるファイル読み込みも、移行期間中の互換APIとして残しています。
- 列名変換を行う6クラスは、クラスごとの対応表と共通の補助関数で、変換前・変換済み・
  両者が混在した入力を扱います。同じ列の旧名と新名が併存する場合はエラーにします。
  型はまだ統一していません。`CreditbalancePl`の`code`を`pl.Int64`へ変換する処理など、
  既存の型変換はそのまま維持しています。
- 公開メソッドの`get_*`は`self.df`を変更せず結果を返し、`filter_*`、
  `with_columns_*`、`convert_*`は`self.df`を書き換えて`None`を返します。共通加工を
  担う非公開の`_with_columns_*`は、加工済みDataFrameを返します。
- 株価の`with_columns_moving_average`と信用残高の`with_columns_margin_ratio`は、
  行と行順を保持します。移動平均は銘柄内の入力順（日付昇順を前提）で計算し、
  期間不足や期間内の欠損は`null`にします。信用倍率（買残高 / 売残高）は銘柄を
  絞らず、売残高が0または欠損、買残高が欠損なら`null`にします。
  `null`の除外などは後続工程で判断します。両メソッドとも既存の結果列は更新します。
- 他の`with_columns_*`では行が減る場合があります。例えば
  `with_columns_margin_volume_ratio`は結合後に`drop_nulls()`を行います。
  列追加だけとは限りません。
- `KessanPl`は初期化時に既知の決算日データの補正・除外と古い期間の行除外を行います。
  他データの読み込みや標準出力を伴うメソッドもあります。

これは現在の挙動の記録であり、例外や不統一を新しいクラスへそのまま引き継ぐ規則では
ありません。正確な引数・戻り値・副作用は、対象メソッドの実装とテストで確認します。

### クラスの単位と責務

- クラスはデータの意味・列構成と、それに固有の操作を単位にします。同じ操作を持つ
  データは保存ファイルが違っても同じクラスで扱えます。新しいファイルの追加だけを
  理由にクラスを増やしません。
- PolarsのDataFrameを保持するクラスは、既存の`<name>Pl`という命名を踏襲できます。
  既存クラスやメソッドの名称・綴りは公開APIとして扱い、命名整理だけで変更しません。
- データ加工は`data_processing.py`、読み込みは`data_loading.py`、グラフ生成は
  `graph_processing.py`を基本の境界とします。新しい加工メソッドは計算結果を返すか
  保持データを更新し、表示や保存は呼び出し側で組み立てます。
- 複数データを横断する分析が大きくなる場合は、必要なDataFrameを明示的に受け取る
  関数などへ切り出します。共通処理のための継承や基底クラスは必須にしません。

### 入力と初期化

加工用の入口はDataFrame、ファイル読み込みの入口は`from_file`に統一します。

```python
# taskなどで読み込んだDataFrameから生成する（ファイル読み込みなし）。
prices = PricelistPl(df)
credit = CreditbalancePl(df=df)

# 管理対象ファイル名、またはそのファイルへのstr・Pathを指定する。
prices = PricelistPl.from_file("raw_pricelist.parquet")
credit = CreditbalancePl.from_file("creditbalance.parquet")

# 引数を省略すると、上表の既定ファイルを読み込む。
prices = PricelistPl.from_file()
credit = CreditbalancePl.from_file()
```

- `from_file`はclassmethodとし、引数名を`fp`、型を`str | Path`に揃えます。
  `load_data_file(fp)`で読み込んだDataFrameを`cls(df=...)`へ渡し、列名・型の変換や
  既存の補正を通常のコンストラクタと同じ経路で行います。空のDataFrameを渡した場合も、
  既定ファイルへの読み替えは行いません。必要な列など、各クラスの入力条件は維持します。
- `from_file`は`data_loading.py`のパス解決・ファイル名検証を利用します。ファイル名だけ
  なら`DATA_DIR`から、ディレクトリ付きなら指定パスから読み込みます。
  管理対象外の名前も許可する任意パス読み込みが必要なら、呼び出し側で明示的に
  `read_data(fp)`を実行してDataFrameを渡します。既存の`tmp_`を含む名前の許可条件は
  `data_loading.py`に従います。
- 新しいクラスはコンストラクタで`df: pl.DataFrame`を必須とし、読み込みを必要とする
  クラスに`from_file`を設けます。既存8クラスについては、以下の移行方針に従います。
- 列名は以下の共通規則で正規化します。必須列・型の共通定義と検証は次の段階で
  決める方針であり、今回の列名変換では追加していません。
- 分析期間の絞り込みや業務上の補正は、明示的な加工処理として分けることを基本と
  します。既存の`KessanPl`の初期化処理は、挙動を確認せずに削除・移動しません。
- 他データを使う加工では、必要なDataFrameを引数で受け取り、内部で暗黙にファイルを
  読み込む依存を増やしません。株価調整の有無や日足・週足など、計算に影響する入力の
  違いは引数・docstringで明示します。

### 列名の正規化（現在の到達点）

列名変換を行うクラスは`_COLUMN_RENAMES`に対応表を持ち、
`data_processing.py`の`_normalize_column_names`を初期化時に使います。
既存の対応関係を保ち、処理方法を共通化した段階です。

| クラス | 旧名 → 正規化後の列名 |
| --- | --- |
| `FinancequotePl` | `mcode` → `code`、`p_key` → `date` |
| `IndexPricelistPl` | `p_key` → `date`、`p_open` → `open`、`p_high` → `high`、`p_low` → `low`、`p_close` → `close` |
| `PricelistPl` | `mcode` → `code`、`p_key` → `date`、`p_open` → `open`、`p_high` → `high`、`p_low` → `low`、`p_close` → `close` |
| `KessanPl` | `mcode` → `code` |
| `MeigaralistPl`、`ShikihoOnlinePl` | `mcode` → `code`、`mname` → `name` |
| `CreditbalancePl`、`PortfolioManager` | 初期化時の列名変換なし。`code`、`ticker_code`など既存の列名を維持 |

共通の補助関数は、対応表の各列について次のように処理します。

- 旧名だけがある場合は、新名へ変換します。
- 新名だけがある場合は、そのまま受け付けます。一部の列だけ変換済みでも、残りの列を
  個別に変換します。例えば`code`と`p_key`の入力は`code`と`date`になります。
- 旧名と新名が両方ある場合は、値が同じでも`ValueError`にします。エラーには衝突した
  列名を含め、データの値は含めません。旧名・新名のどちらかを暗黙に優先しません。
- どちらもない場合、その対応項目は処理しません。列を補完したり、新しい必須列の
  検証を行ったりはしません。後続の初期化や加工で必要な列が不足すれば、従来どおり
  その処理でエラーになる可能性があります。
- 対応表にない列は保持します。列名変換そのものは、列順・行順・行数・値・型を変えず、
  呼び出し元のDataFrameも変更しません。行が0件でも同じ規則を適用します。

列名の正規化を繰り返しても、列名は変わりません。`from_file`も同じコンストラクタを
通るため、変換済みの列名を持つファイルを受け付けます。ただし、`KessanPl`の補正や
行除外など、列名変換以外の初期化処理まで繰り返しの結果が同じになることを保証する
ものではありません。

今回の変更で、`FinancequotePl`は変換済み入力を受け付けるようになり、株価や銘柄一覧
なども特定の旧名の有無に依存せず列ごとに変換します。型の統一、必須列の共通検証、
対応表への別名の追加、既存の業務上の補正・行除外の見直しは別の段階で扱います。

### 生成APIの移行と互換性

新しく書く呼び出しは`Class(df)`または`Class.from_file(...)`に揃えます。
`data_processing.py`内の直接のファイル指定・引数なしの生成呼び出しは`from_file`へ
移行済みです。Notebookなどの既存利用を維持するため、今回は次の互換APIを残します。

- 8クラスの引数なし生成は、従来の既定ファイルを従来の読み込み経路で読み込みます。
  特に`KessanPl()`は引き続き`read_data`を使います。
- `PricelistPl`と`IndexPricelistPl`は、位置引数のファイル名・パス、および旧`fp=`での
  DataFrame・ファイル指定を受け付けます。新しい`df=`と旧`fp=`の両方に非None値を
  渡すと、曖昧な入力として`TypeError`になります。
- `CreditbalancePl(df=None, fp=...)`は従来どおり`read_data`で任意パスを読み込みます。
  両方が指定された場合も、従来の`df`優先を維持します。この互換用の優先順位を
  新しいクラスの設計規則にはしません。

入口の統一では、`KessanPl`の日付補正・行除外などの加工を維持しています。
列名正規化の共通化は上記の段階として実施済みです。既存クラスの`df`必須化や
旧APIの削除は行っていません。
taskやNotebookの利用箇所の移行を確認したうえで、別の変更として扱います。

### メソッドの状態変更と戻り値

新規メソッドでは、現行の主要な慣習に合わせて次を基本とします。

| 命名 | 状態変更と戻り値 |
| --- | --- |
| `get_*` | `self.df`を変更せず、DataFrame・Series・数値などの結果を返す |
| `filter_*` | `self.df`を絞り込み、`None`を返す |
| `with_columns_*` | `self.df`へ加工列を追加・更新し、`None`を返す |
| `convert_*` | 日足から週足への変換など、`self.df`の構造を変え、`None`を返す |

Polars自身の同名メソッドとは状態変更の扱いが異なるため、型注釈とdocstringに
明記します。`inplace`による切り替えは使わず、取得と更新を別のメソッドに分けます。

更新処理はローカル変数で結果を組み立て、成功後に`self.df`へ代入することを基本と
します。新規の列追加は行を保持することを基本とし、行除外や並び替えも必要な場合は
その条件を明示します。呼び出し元から渡されたDataFrameを直接変更しません。

同じ加工を取得用と更新用の双方から使う場合は、DataFrameを受け取って加工済みDataFrameを
返す非公開処理へ分離します。`get_*`はその結果をローカル変数として使い、`self.df`へ代入
しません。`with_columns_*`は同じ非公開処理の結果を`self.df`へ代入し、`None`を返します。
`KessanPl`の年度決算日、四半期累積値、差分成長率、差分成長率による次回予想の加工は
この形で共有しています。これにより、取得途中の並び替えやNULL行の除外がインスタンスへ
残ることを防ぎます。

### 取得と更新の分離

日付で抽出したDataFrameを取得・更新する場合は、次のメソッドを使います。

| クラス | 取得用メソッド | 更新用メソッド |
| --- | --- | --- |
| `FinancequotePl` | `get_finance_quotes(valuation_date=...)` | `filter_finance_quotes_by_date(specific_date=...)` |
| `PortfolioManager` | `get_portfolio_as_of_specific_date(specific_date=...)` | `filter_portfolio_as_of_specific_date(specific_date=...)` |

両取得メソッドは`date <= 指定日`の行から、データ全体の最新日を抽出します。銘柄ごとの
最新日ではありません。列・行順を保持したDataFrameを返し、`self.df`を変更しません。
対象がない場合は同じスキーマの空DataFrameを返します。

```python
quotes_df = quotes.get_finance_quotes(valuation_date=valuation_date)
portfolio_df = portfolio.get_portfolio_as_of_specific_date(specific_date=valuation_date)
```

更新用の`filter_*`は対応する取得メソッドへ抽出処理を委譲し、その結果で`self.df`を
更新して`None`を返します。`inplace`引数はありません。

```python
quotes.filter_finance_quotes_by_date(specific_date=valuation_date)
portfolio.filter_portfolio_as_of_specific_date(specific_date=valuation_date)
```

`get_individual_stocks`と`get_individual_stocks_info`、および
`notebooks/dev/FinancequotePl.ipynb`・`notebooks/dev/PortfolioManager.ipynb`の取得用の
呼び出しは`get_*`を使います。PortfolioManagerのNotebookにある更新例も、更新専用の
`filter_*`を使います。

`get_portfolio_as_of_specific_date`と`filter_portfolio_as_of_specific_date`は日付省略時に
呼び出し時の今日を使います。既存の`get_finance_quotes`と
`filter_finance_quotes_by_date`の日付引数・既定値は維持しています。再現可能な分析では
評価日を明示します。

`tests/test_data_processing_getters.py`で取得結果、日付境界、取得時に状態を変更しないこと、
更新メソッドの戻り値と状態変更、`inplace`を受け付けないこと、移行した呼び出し側を
検証します。

### データと時系列の前提

必要な規則をクラスまたはメソッドのdocstringとテストに記録します。

- 必須列・型、結合キー、追加列名、値の単位、丸め桁数を明示します。`code`、`date`、
  `open`、`close`など既存の意味と一致する列は、既存名を使います。
- 差分・移動平均・`first`・`last`を使う処理は、銘柄ごとの区切りと時系列の並び順を
  明確にします。必要なソートを行うか、前提を検証し、入力ファイルの順序だけに依存
  しない実装にします。結合で行が増える可能性がある場合は、キー重複の扱いを決めます。
- 「最新」は、データ全体の最新日か銘柄ごとの最新日かを明示します。現行の
  `CreditbalancePl.get_latest_df`や`FinancequotePl.get_finance_quotes`は全体の最新日を
  使用します。日付の上限を含むかも処理ごとに確認します。例えば
  `ShikihoOnlinePl.get_latest_df`は`issue < target_date`です。
- 過去時点の分析では、取引日・決算期末日・発表日・発行日のどれを参照するかを決め、
  評価日時点で利用可能な情報だけを使う条件を明示します。新しい日付引数の既定値は、
  日付の明示指定、または`None`を受けて呼び出し時に`date.today()`を評価する形にします。
- 空データ、対象銘柄なし、null、計算期間不足、ゼロ除算について、該当するものの
  戻り値・行除外・例外の扱いを決めます。これらを一律にゼロ埋めしません。

### 互換性と検証

既存クラスを変更する場合は、今後の主な呼び出し側となる`tasks/`に加え、
`graph_processing.py`、補助スクリプト、Notebookの
コードセル、テストの利用箇所を確認します。公開import、引数、状態変更、出力列・型、
行の選択条件を変更する場合は、その影響を設計に含めます。コメント・型注釈と実装に
食い違いがあれば、確認できた挙動と未確認の意図を分けて記録します。

検証には小さな人工DataFrameを使います。`tests/test_data_processing.py`のように、
列名変換、抽出結果、計算結果、状態変更の有無を確認し、変更に関係する日付境界や
銘柄間の混入などのケースを追加します。読み込み経路を検証する場合は読み込みを
モックし、ネットワーク・秘密情報・ローカルのParquetファイルに依存させません。
`tests/test_data_processing_construction.py`では、8クラスの共通の生成API、既存APIの
互換性、決算日の補正・古い行の除外を人工データで検証します。列名正規化についても、
変換前・変換済み・混在・空入力、衝突、追加列・型の保持を同ファイルで検証します。
文書やskillのみの変更では、参照先や内容の整合性を確認します。

## ローカルデータと安全性

`data/`はGit管理対象外です。`wq dl-pq`は外部通信を行い、同名のParquetファイルを
上書きするため、通常のテストや動作確認として自動実行しません。

Notebookには長時間処理やファイル更新などの副作用が含まれる可能性があります。
対象と副作用を確認せずに一括実行しないでください。

### 株価履歴一覧

`cli.py`の`price-history`は引数検証と表の標準出力を担当します。
`flows/price_history.py`は読み込み・抽出Taskを組み合わせ、DataFrameを返します。
`tasks/price_history.py`の`load_price_history`は選択されたファイルを
`PricelistPl.from_file()`で読み込み、`select_price_history`は銘柄・期間の
抽出、7列の選択、日付昇順の並べ替えを行います。
銘柄コードは文字列として照合し、整数型と英字入りの文字列型に対応します。
入力のDataFrameは変更せず、表示・保存はTask・Flowでは行いません。
