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


## `quarterly-valuation`

目的  
- 指定日(--valuation-date)における最新四半期単体決算で、業績改善効果の高かった企業を抽出するためのコマンドです。
- 過去履歴データを活用した騰落率のフォワードテスト(シミュレーション)にも対応します(出力一覧のperf列)。
- 指定日(--valuation-date)における持株の評価(--portfolio)にも利用可能です。

ローカルの決算・財務指標・分割調整済み株価から、四半期ベースの銘柄評価一覧を
標準出力します。外部通信やファイル更新は行いません。

```bash
uv run wq quarterly-valuation
uv run wq quarterly-valuation --codes all
uv run wq quarterly-valuation --codes 7203 --codes 6758
uv run wq quarterly-valuation --valuation-date 2026-09-19 \
  --sort-columns dgrp --sort-columns grsl --sort-order desc
```

| オプション | 既定値 | 意味 |
| --- | --- | --- |
| `--valuation-date` | 実行日の今日 | 評価日。`YYYY-MM-DD`形式 |
| `--sort-columns` | `dgrp` | 出力列名。複数指定時はオプションを繰り返し、指定順に優先 |
| `--perf-period` | `quarter` | `perf`・`bm`の期間。`quarter`=今回発表翌取引日→次回発表翌取引日、`announcement`=次回発表当日→翌取引日（始値） |
| `--profit` | `operating` | `prft`・`pr`・`dgrp`・`ngrpr`の利益種別。`operating`=営業利益、`ordinary`=経常利益 |
| `--sort-order` | `desc` | 指定した全列に共通の降順`desc`または昇順`asc` |
| `--codes` | `all` | 全銘柄、または整数の銘柄コード。複数銘柄はオプションを繰り返す |
| `--portfolio` | 無効 | 評価日以前の最新ポートフォリオの個別株に絞る |
| `--start-row` | `1` | 抽出・ソート後の開始行（1始まり） |
| `--end-row` | 末尾まで | 終了行（指定行を含む） |
| `--sls-min` / `--sls-max` | 制限なし | 売上高の下限 / 上限（出力と同じ単位） |
| `--profit-min` / `--profit-max` | 制限なし | `--profit`で選んだ利益の下限 / 上限（出力と同じ単位） |
| `--pr-min` / `--pr-max` | 制限なし | `--profit`で選んだ利益率`pr`の下限 / 上限（%単位） |
| `--dgrp-min` / `--dgrp-max` | 制限なし | `dgrp`の下限 / 上限（表示と同じ%単位） |
| `--grsl-min` / `--grsl-max` | 制限なし | 売上成長率の下限 / 上限（%単位） |

範囲フィルタは境界値を含み、指定されたすべての条件を満たす行を表示します。
`--codes`とも併用できます。銘柄ごとの最新決算を選んだ後、表示用に丸める前の値で
比較します。条件から外れても過去の決算へ戻りません。条件を指定した列がnullの行は
除外しますが、未指定の列のnullは除外しません。上下限が逆の場合は0件になります。

```bash
# 売上高10000以上、売上成長率10%以上50%以下
uv run wq quarterly-valuation --sls-min 10000 --grsl-min 10 --grsl-max 50
# 利益率が5%以上20%以下（片側だけの指定も可能）
uv run wq quarterly-valuation --pr-min 5 --pr-max 20
# dgrpが10%以上30%以下
uv run wq quarterly-valuation --dgrp-min 10 --dgrp-max 30
# 経常利益1000以上5000以下（営業利益の場合は --profit operating）
uv run wq quarterly-valuation --profit ordinary --profit-min 1000 --profit-max 5000
```

Pythonのflow・加工taskには`sls_min`、`sls_max`、`grsl_min`、`grsl_max`、
`profit_min`、`profit_max`、`dgrp_min`、`dgrp_max`、`pr_min`、`pr_max`を渡せます。利益の上下限は`profit`で選んだ`prft`列に適用します。
`dgrp_min`・`dgrp_max`はPythonでも%単位です（25は内部値0.25に対応）。
`--profit`で選んだ利益に基づく`dgrp`へ適用し、表示前の値で比較します。
`pr_min`・`pr_max`はPythonでも%単位で、`profit`で選んだ利益率`pr`に適用します。
各引数は`float | None = None`で、`None`は制限なしです。


`--codes all`または省略時は絞り込みません。個別コードを指定すると、その銘柄だけを
表示します。`all`と個別コードの混在、および整数でないコードは入力エラーです。
Pythonのflow・加工taskでは`codes: Literal["all"] | list[int] = "all"`を受け取り、
空リストは0件、重複コードの指定は行を増やしません。対象外のコードは表示されません。

銘柄名は`meigaralist.parquet`を`MeigaralistPl.from_file()`で読み込み、`code`で結合します。
名前が見つからない行も残し、`name`をnullにします。銘柄名はこのファイルの現在の値であり、
過去の評価日時点の社名を復元するものではありません。完全一致する銘柄名の重複行は
まとめ、同じコードに異なる名前がある場合は結合エラーにします。

同値の場合は`code`昇順（`code`を明示指定した場合はその指定を優先）、nullは末尾です。
不明な列名と列の重複指定はエラーにします。行・列を省略せず表示し、
`sls`・`prft`は表示時に整数へ四捨五入し、3桁区切りのカンマを付けます
（例：`169927.0` → `169,927`。負数の端数0.5も絶対値を切り上げます）。
`pr`・`grsl`・`dgrp`・`ngrpr`・`perf`・`bm`は小数点以下2桁と`%`を表示します。`pr`・`grsl`・`ngrpr`・`perf`・`bm`は既に%単位のため値をそのまま、
`dgrp`は表示時だけ100倍します（内部値`0.25` → `25.00%`）。
`PER`・`divr`は従来どおり小数点以下2桁、nullは`null`と表示します。
内部値・ソート・フィルタは表示の丸めや変換の影響を受けません。

### 利益種別

既定では営業利益（`operating_income`）を使用し、`prft`・`pr`・`dgrp`・`ngrpr`に適用します。
従来の経常利益（`ordinary_profit`）による計算は明示指定できます。

```bash
uv run wq quarterly-valuation --profit operating
uv run wq quarterly-valuation --profit ordinary
```

両期の利益が正であることなどの計算条件は選択した利益に適用します。
`prft`列は選択した利益です。`dgrp`の内部値は倍率のまま、表示時に100倍して`%`を付けます。
旧オプション`--dgrp-profit`は`--profit`へ、旧出力列`ordp`は`prft`へ変更しました。
既定値の変更により`dgrp`の値と既定の並び順も変わります。
Pythonの計算API・task・flowでも`profit: Literal["operating", "ordinary"] = "operating"`
を指定できます。

### 表示する行の範囲

```bash
# フィルタ・ソート後の11件目から30件目まで
uv run wq quarterly-valuation --start-row 11 --end-row 30
```

行番号は見出しを数えず1から始まり、開始・終了の両端を含みます。
`--portfolio`・`--codes`・数値フィルタとソートを適用した後に範囲を切り出します。
終了行が件数を超えた場合は末尾まで、開始行が件数を超えた場合は見出しだけを表示します。
0以下の行番号、または開始行が終了行を超える指定は入力エラーです。
Pythonのflow・加工taskでも`start_row: int = 1`、`end_row: int | None = None`を指定できます。

### ポートフォリオの個別株に絞る

```bash
uv run wq quarterly-valuation --portfolio
uv run wq quarterly-valuation --portfolio --valuation-date 2026-09-01
uv run wq quarterly-valuation --portfolio --grsl-min 10
uv run wq quarterly-valuation --portfolio --codes 5334 --codes 7203
```

`--portfolio`指定時だけ`base_portfolio.parquet`を読み込みます。
`date <= valuation_date`のうちポートフォリオ全体の最新日を選び、
その日の`instrument_type == "個別株"`の銘柄を抽出します。
評価日当日を含みます。ETFは対象外で、銘柄別に古い保有履歴へ遡りません。

`ticker_code`を整数コードへ変換し、複数口座などによる重複はまとめます。
整数に変換できないコード（nullを含む）は除外し、コード重複除去後の除外件数を
標準エラーへ通知します。コードそのものは通知に含めません。

`--codes`と併用した場合は共通する銘柄だけを表示します。`--codes all`や省略時も、
`--portfolio`指定中は保有銘柄に限定されます。既存の決算条件（評価日前93日以内の発表）と
`sls`・`grsl`フィルタはそのまま適用され、条件を満たす決算がない保有銘柄は表示しません。
出力は15列・1銘柄1行を維持します。対象ポートフォリオや対象銘柄がない場合は
見出しのみ表示し、全銘柄表示には戻りません。

Pythonでは`quarterly_valuation_flow(..., portfolio=True)`を使用します。
`portfolio: bool = False`が既定値です。

### 出力列と計算

| 列 | 内容 |
| --- | --- |
| `code` | 銘柄コード |
| `name` | 銘柄名 |
| `setd` | 決算期末日 |
| `annd` | 決算発表日 |
| `qtr` | 選択された決算の`quater`値に`q`を付けて表示（例：`1q`）。各社の決算年度の四半期番号をそのまま使い、日付から再計算しません。欠損は`null` |
| `sls` | 四半期単体の売上高（元データの単位を保持） |
| `prft` | 四半期単体の選択した利益（元データの単位を保持） |
| `pr` | `prft / sls × 100`（%、小数点以下2桁で表示） |
| `grsl` | `(売上高 / 前年同期売上高 - 1) × 100`（%） |
| `dgrp` | `(選択した利益 - 前年同期の同じ利益) / (売上高 - 前年同期売上高)`（内部値は倍率、表示時に100倍して%表記） |
| `ngrpr` | 次四半期の予測利益成長率（%、小数点以下2桁で表示） |
| `PER` | 株価比率で補正した予想PER（倍） |
| `divr` | 株価比率で補正した予想配当利回り（%） |
| `perf` | `--perf-period`で選択した期間の始値による騰落率（%） |
| `bm` | `perf`と同じ開始日・終了日のnh225の始値による騰落率（%） |

`pr`は当期売上が正で、売上・選択利益・計算結果が有限の場合に算出します。
赤字は負値、利益0は0.00%とし、売上0以下・欠損・非有限値はnullです。
前年データの有無に依存せず、丸め前の金額から計算します。

対象は`settlement_type == "四"`かつ
`評価日 - 93日 <= announcement_date < 評価日`の行です。
銘柄ごとに決算期末日が最新の行を選び、同じ決算期なら発表日が最新の行を選びます。
前年同期は同一銘柄の前年同月の四半期単体決算を使います。
前年データも評価日前の発表に限り、93日間の抽出前の履歴から取得します。
既存の`KessanPl`初期化時の日付補正と古い行の除外は適用されます。

### 次四半期の予測利益成長率（ngrpr）

前年同期売上高を`lsls`とし、既存の`grsl = (sls / lsls - 1) × 100`を使用します。
前年同期から3か月後の四半期単体決算の売上高を`nlsls`、
`--profit`で選んだ利益を`nlprofit`として計算します。
例えば最新決算が2026年6月期なら、2025年9月期の売上高・利益を使用します。
年をまたぐ場合も同様で、最新が2026年12月期なら2026年3月期です。
対象年月が欠落していても別の期で補完しません。
評価日より前に発表された履歴から、対象年月の最新決算期・最新訂正を使用します。

```text
nsls    = nlsls × (1 + grsl / 100)
nprofit = nlprofit + (nsls - nlsls) × dgrp
ngrpr   = (nprofit - nlprofit) / nlprofit × 100
```

`dgrp`は表示前の倍率を使用します。`ngrpr`は%単位で保持し、丸め前の値で
`--sort-columns ngrpr`によるソートができます。表示位置は`dgrp`の右隣です。
`grsl`・`dgrp`が計算できない場合、対象データの欠損・非有限値、`nlprofit = 0`、
途中の計算や結果が非有限値の場合はnullとし、銘柄の行は残します。
`nlprofit`が負でも式どおり計算し、結果の負値・0も保持します。

### 決算発表間の株価騰落率（perf）

`--perf-period`で計算期間を選択します。省略時は`quarter`です。

- `quarter`: `perf = (次回発表翌取引日の始値 / 今回発表翌取引日の始値 - 1) × 100`
- `announcement`: `perf = (次回発表翌取引日の始値 / 次回発表当日の始値 - 1) × 100`

`announcement`は同じ次回発表の当日と翌取引日を比較します。場中・引け後の発表を
区別せず、当日の株価は日付の完全一致で取得します。休場日や当日データ欠損時はnullで、
前後の日には置き換えません。`bm`も選択した期間に連動します。

```bash
uv run wq quarterly-valuation --perf-period announcement
```

起点は出力の`annd`で、評価日当日の発表は従来どおり起点に含めません。
次回は、同じ銘柄の次の決算期の四半期実績（`settlement_type == "四"`）の
最初の発表を使います。同一期の訂正発表は次回に含めません。
次回発表と株価の探索には評価日より後のデータも使用します。

始値は`reviced_pricelist.parquet`の分割調整済み`open`を使い、銘柄ごとに
翌取引日には各発表日より後の最初の取引日を選びます（暦上の翌営業日ではなく、株価データ上の取引日）。
次回発表や必要な株価がない場合、始値がnull・0以下・非有限値の場合はnullです。
最初の取引日の始値が無効でも、さらに後日の始値には置き換えません。
内部値は%単位で保持し、表示時に小数点以下2桁へ丸めて`%`を付けます。
`--sort-columns perf`によるソートも可能です。

### 同期間の指数騰落率（bm）

`nh225.parquet`を読み込み、
`bm = (終了日の指数始値 / 開始日の指数始値 - 1) × 100`を計算します。
開始日・終了日は、各銘柄の`perf`に使用する実際の取引日と一致させます。
指数の始値（`p_open`）は日付の完全一致で取得し、前後の日では補完しません。
次回発表や必要な取引日がない場合、指数の始値が欠損・0以下・非有限値の場合はnullです。
個別株の始値が無効で`perf`がnullでも、取引日と有効な指数始値があれば`bm`は計算します。
指数の同日同値の重複はまとめ、同日に異なる始値がある場合はエラーにします。
`perf`の右隣に小数点以下2桁と`%`で表示し、`--sort-columns bm`でソートできます。

### PER・配当利回りの株価補正

銘柄ごとに、評価日より前の最新Financequoteと最新の終値を使用します。
営業日カレンダーは追加せず、株価履歴の`date < 評価日`から最新日を選びます。
Financequoteの日付が指標の基準となる株価日付と一致する前提です。

`reviced_pricelist.parquet`にあるFinancequote基準日と同日の終値を`P0`、
評価日より前の最新終値を`P1`として、次の式を使います。

```text
PER  = expected_PER × P1 / P0
divr = expected_dividend_yield × P0 / P1
```

2つの株価は同じ分割調整済み履歴を使います。基準日と同日の株価がなければ、
別日の株価で代用せずPER・divrをnullにします。
この補正は株価変化だけを反映し、Financequote取得後の業績予想・配当予想の
変更を補完するものではありません。


### null・ゼロ・重複行

- `grsl`は当期・前年の売上高がともに正の場合に計算します。
- `dgrp`はさらに両期の選択した利益がともに正で、売上高前年差が0でない場合に計算します。
- 計算結果の負値・0は有効値として保持します。
- 元のPERが0以下なら`PER`はnull。元の配当利回りが負なら`divr`はnullですが、無配の0%は保持します。
- 株価が0以下、比較データなし、欠損・非有限値の場合は該当指標をnullにし、銘柄の行は残します。
- 同じキーと値の重複行はまとめます。最新発表の同一決算期に異なる決算値、
  最新Financequoteの同日に異なる指標、評価日前の株価の同日に異なる終値があれば、
  値を任意に選ばずエラーにします。
- 対象が0件の場合は列見出しだけを表示します。

## 株価履歴一覧（price-history）

指定した1銘柄・期間の四本値と出来高を標準出力へ表示します。

```bash
uv run wq price-history --code 7203 --start-date 2026-01-01 --end-date 2026-03-31
uv run wq price-history --code 130A --start-date 2026-01-01 --end-date 2026-03-31 --source raw
```

| オプション | 仕様 |
|---|---|
| `--code` | 必須。半角英数字の銘柄コードを1つ指定 |
| `--start-date` | 必須。開始日、YYYY-MM-DD形式 |
| `--end-date` | 必須。終了日、YYYY-MM-DD形式 |
| `--source` | `reviced`（既定）または`raw` |

- `data/reviced_pricelist.parquet`または`data/raw_pricelist.parquet`を読み込みます。
- コードは整数型・文字列型に対応し、文字列化した値と完全一致で照合します。
  英字の大文字・小文字は区別し、先頭のゼロは除去しません。
  英字入りコードの表示には、選択したファイルにそのコードのデータが必要です。
- 期間は両端を含みます。休場日・欠落日は補完しません。
- 表示列は`code date open high low close volume`、日付の昇順です。
- ヘッダー付きの整列表として全行・全列を表示します。追加の丸めは行わず、
  欠損値は`null`と表示します。
- 該当データがなければ標準出力はヘッダーのみ、標準エラーへ「該当データなし」を
  表示し、正常終了します。
- 引数の不正（開始日が終了日より後など）、ファイル不足・読み込み失敗では
  標準エラーへ理由を表示し、非ゼロで終了します。
- 外部通信、ファイル更新、分割調整の再計算は行いません。

## `print-parquet`

`data/`直下の任意のParquetを、全列・表形式で標準出力へ表示します。
管理対象ファイル一覧への登録は不要です。

```bash
uv run wq print-parquet --file base_portfolio
uv run wq print-parquet --file base_portfolio --head 10
uv run wq print-parquet --file base_portfolio --tail 10
```

| オプション | 説明 |
| --- | --- |
| `--file NAME` | 必須。`.parquet`を除いたファイル名。ディレクトリ指定は不可。 |
| `--head N` | 先頭N行。Nは1以上の整数。 |
| `--tail N` | 末尾N行。Nは1以上の整数。 |

行数指定を省略すると全行表示します。`--head`と`--tail`は同時指定できません。
Nが総行数を超える場合は全行を表示します。保存順を保持し、日付によるソートは行いません。
行・列・セル内容は省略せず、nullは`null`と表示します。セル内の改行・タブ・復帰・
バックスラッシュは、それぞれ`\n`・`\t`・`\r`・`\\`にエスケープします。
空データでは列名を表示し、標準エラーへ「該当データなし」と通知して正常終了します。
指定ミスや読み込み失敗は標準エラーへ通知し、非ゼロで終了します。

ファイル全体を読み込んでから行を選ぶため、行数を制限しても全体分のメモリが必要です。
外部通信やファイルの更新は行いません。

## `ds-quarterly-valuation`

機械学習用の全期間の四半期評価データセットを作成して保存します。

```bash
uv run wq ds-quarterly-valuation
uv run wq ds-quarterly-valuation --profit ordinary --perf-period announcement
```

| オプション | 既定値 | 選択肢 |
| --- | --- | --- |
| `--profit` | `operating` | `operating`（営業利益）、`ordinary`（経常利益） |
| `--perf-period` | `quarter` | `quarter`（今回発表翌取引日→次回発表翌取引日）、`announcement`（次回発表当日→翌取引日） |

保存先はリポジトリ直下の`data/datasets/`です。なければ作成します。
ファイル名は`ds-quarterly-valuation-<profit>-<perf-period>-YYYYMMDD_HHMMSS.parquet`で、
実行環境のローカル日時を使います。同名ファイルがあれば上書きせずエラーにします。
全レコードを数値の丸めなしで保存し、標準出力には先頭10行・末尾10行と保存先・件数を表示します。
10行以下の場合も先頭・末尾をそれぞれ表示します。0件の場合もスキーマ付きの空ファイルを保存します。

列順は`code, name, setd, annd, qtr, sls, prft, pr, grsl, dgrp, ngrpr, PER, divr, perf, bm`です。
`code, setd`の昇順で、同じキーは1行です。単位・計算式・無効値の条件は
`quarterly-valuation`と同じで、`dgrp`は倍率、`pr, grsl, ngrpr, divr, perf, bm`は%です。

- 初期化後の`KessanPl.df`の全期間から`settlement_type == "四"`を対象にします。
  既存の初期化による日付補正・2017年1月1日以前の決算除外は維持します。
- 決算期ごとに最初の発表を採用します。同じ銘柄・決算期・発表日に異なる値があればエラーです。
- 評価日は銘柄の株価履歴にある今回発表後最初の取引日です。比較する前年同月・9か月前同月の
  決算は、その評価日より前に判明した最新訂正を使います。将来の訂正を特徴量に使いません。
- `PER`・`divr`には評価日より前の最新Financequoteと終値を使い、指標取得日と同日の
  終値で株価補正します。履歴不足ならnullにし、行を残します。
- `perf`・`bm`は分割調整済み株価と日経平均の始値を使用します。次回は次の決算期の初回発表です。
  いずれかがnullの行だけを除外します。他列のnullは除外理由にしません。
- 株価の取引日判定、指数の日付完全一致、会社名の結合は既存コマンドと同じです。
  銘柄名には読み込んだ銘柄一覧を使い、過去時点の名称は復元しません。

ローカルの既存5ファイルを読み込みます。ダウンロード・外部通信は行いません。
