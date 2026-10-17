# ユーザーガイド

wequantのコマンドを使って、分析用データを準備する手順を説明します。
セットアップは[README](../README.md)、オプションの詳細は
[CLIリファレンス](REFERENCE.md)を参照してください。
以下のコマンドはリポジトリルートで実行します。

## 四半期評価データを機械学習用に準備する

`ds-quarterly-valuation`で作成した原本Parquetから、
`prepare-quarterly-valuation`で対象レコードと特徴量を選び、目的変数を作成します。
加工したParquetと加工条件JSONを読み込み、学習用の説明変数`X`と目的変数`y`を用意します。
この手順はデータの準備までを扱います。モデルの学習・評価は別途行ってください。

### 1. 原本ファイルを用意する

原本がある場合は、そのファイルを次の手順の入力に使ってください。
新しく作成する場合は、必要なローカルParquetを準備したうえで次を実行します。

```bash
uv run wq ds-quarterly-valuation --profit operating --perf-period quarter
```

この例では営業利益を使い、今回の決算発表翌取引日から次回の決算発表翌取引日までの
騰落率を計算します。実際の評価日は銘柄の株価履歴にある発表後最初の取引日です。
経常利益を使う場合は`--profit ordinary`、次回発表当日から翌取引日までを
評価する場合は`--perf-period announcement`を指定します。
利益種別と評価期間は、学習したい内容に合わせて原本作成時に選んでください。

原本作成は決算、Financequote、分割調整済み株価（`reviced_pricelist.parquet`）、
銘柄一覧、日経平均の既存ローカルデータを使います。ダウンロードや外部通信は行いません。
`perf`または`bm`が欠損するレコードは原本に含まれないため、
全期間のすべての決算が揃うとは限りません。

保存先と件数が標準出力に表示されます。保存先の例は次のとおりです。

```text
data/datasets/ds-quarterly-valuation-operating-quarter-20260927_123456.parquet
```

日時部分は実行ごとに変わります。以降の`--input`は実際の保存先に置き換えてください。

### 2. 対象レコード・特徴量・目的変数を選ぶ

ここでは、次の条件で抽出します。条件は操作説明用の例であり、
学習成果や投資成績を保証する選定基準ではありません。

| 項目 | この例の指定 |
| --- | --- |
| 決算期末日 | 2020年1月1日〜2025年12月31日 |
| 四半期 | 各社の決算年度の第1四半期 |
| 売上高 | 10,000以上（元データの単位） |
| PER | 20倍以下 |
| 特徴量 | 売上成長率`grsl`、次四半期の予測利益成長率`ngrpr`、`PER` |
| 目的変数 | 日経平均に対する超過リターン`perf - bm` |

```bash
uv run wq prepare-quarterly-valuation \
  --input data/datasets/ds-quarterly-valuation-operating-quarter-20260927_123456.parquet \
  --setd-from 2020-01-01 \
  --setd-to 2025-12-31 \
  --qtr 1 \
  --sls-min 10000 \
  --per-max 20 \
  --feature grsl \
  --feature ngrpr \
  --feature PER \
  --target excess-return \
  --output data/datasets/prepared/q1-excess-return.parquet
```

日付・数値の範囲は両端を含み、複数の条件をすべて満たす行を残します。
`--qtr 1 --qtr 2`と指定すると第1・第2四半期を対象にします。
四半期番号は各社の決算年度に基づくため、暦年の1〜3月を指定する条件ではありません。
期間・四半期・数値条件を省略すると、その項目による絞り込みは行いません。

フィルターに使う列と学習に使う特徴量は別々に選べます。
この例では`sls`で絞り込みますが、学習の特徴量には含めません。
`--feature`は列ごとに繰り返し指定し、`PER`は大文字で記述してください。
選択できる特徴量は`sls, prft, pr, grsl, dgrp, ngrpr, PER, divr`です。
`ngrpr`は評価時点の情報から計算した予測指標であり、将来の実績利益ではありません。

目的変数は`--target perf`なら株価の騰落率（%）、
`--target excess-return`なら`perf - bm`（パーセントポイント）です。
例えば`perf = 12`、`bm = 5`なら、超過リターンは`7`です。
将来結果の`perf`・`bm`は特徴量やフィルターには指定できません。

指標の単位にも注意してください。`grsl`・`ngrpr`などは%ですが、
この加工コマンドの`dgrp`は倍率です。`--dgrp-min 0.2`は20%相当です。
売上高・利益の単位は元データの単位を確認して指定してください。

### 3. 加工結果を確認する

この例では、次の2ファイルが保存されます。

```text
data/datasets/prepared/q1-excess-return.parquet
data/datasets/prepared/q1-excess-return.json
```

原本は変更しません。同名のParquetまたはJSONがある場合はエラーになるため、
条件を変えて試す際は`--output`に別の名前を指定してください。
`--output`を省略すると、`data/datasets/prepared/`に日時付きの名前で保存します。

Parquetの列は`code, setd, annd, qtr, grsl, ngrpr, PER, target`です。
最初の4列はレコードを識別・確認するための管理用列で、
特徴量は指定した順序で保存されます。`name`、元の`perf`・`bm`、
選ばなかった特徴量は含まれません。

JSONには入力ファイルのパスとSHA-256、加工条件、特徴量の順序、
目的変数の式、加工前後の件数などが記録されます。
ParquetとJSONを組にして保持し、学習結果にも対応する条件を記録してください。

フィルター対象の数値列がnull・NaN・無限値の行は除外されます。
その他の特徴量の欠損・非有限値は残るため、学習前に確認が必要です。
加工後の目的変数に欠損・非有限値がある場合はエラーになります。
抽出結果が0件でもファイルは保存されるため、標準出力の件数を確認してください。

### 4. Pythonで学習用データを読み込む

リポジトリルートで起動したPythonやNotebookで、次のコードを実行します。
Pythonは`uv run python`、JupyterLabは`uv run jupyter lab`で起動できます。
Notebookの作業ディレクトリが異なる場合は、`path`を絶対パスなどに変更してください。

```python
import json
from pathlib import Path

import polars as pl

path = Path("data/datasets/prepared/q1-excess-return.parquet")
metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
df = pl.read_parquet(path)
features = metadata["features"]
target_column = metadata["target"]["column"]

print("加工前:", metadata["rows_before"], "加工後:", metadata["rows_after"])
print(df.head())
print(df.select(features).null_count())

# この例では、特徴量に欠損・NaN・無限値がある行を学習対象から外す。
usable = df.filter(
    pl.all_horizontal([
        pl.col(column).cast(pl.Float64).is_finite().fill_null(False)
        for column in features
    ])
)
print("学習に利用できる件数:", usable.height)
if usable.is_empty():
    raise ValueError("学習対象が0件です。抽出条件と欠損を確認してください。")

# JSONに記録された特徴量だけを、記録された順序で取り出す。
X = usable.select(features).to_pandas()
y = usable.get_column(target_column).to_pandas()
identifiers = usable.select(metadata["identifier_columns"]).to_pandas()
```

`X`には特徴量、`y`には予測したい値が入ります。
`identifiers`は予測結果を元の銘柄・決算に対応付けるために保持します。
管理用列を自動的に`X`へ含めず、特徴量の選択にはJSONの`features`を使ってください。
このコードの欠損除外は学習側の処理で、保存済みParquetを変更しません。
学習対象から外した件数も記録し、特定の期間や銘柄に除外が偏っていないか確認してください。

### 5. 学習・検証の期間と前処理を決める

学習には過去の期間、検証にはそれより後の期間を使う構成を基本にします。
同じ時期のレコードをランダムに混ぜると、将来の情報を使った評価になる可能性があります。
学習期間、検証期間、特徴量、目的変数、欠損処理、モデルの設定を記録してください。

`--setd-from/to`の基準である`setd`は決算期末日です。
決算発表日`annd`、実際の評価日、目的変数の計測終了日とは異なります。
例えば決算期末日が学習期間内でも、発表やリターンの計測終了が検証期間に入る場合があります。
加工結果には実際の評価日・計測終了日の列がないため、`setd`だけで分割して
将来情報の混入を防げたと判断せず、原本作成に使った発表・株価履歴などで境界を確認してください。
学習に使う目的変数の計測が検証開始前に終わっていることと、
期間をまたぐ計測の重複を確認し、必要に応じて境界付近のレコードを除外します。

欠損補完や標準化を行う場合、その設定値は学習データだけから求め、
検証データには同じ設定を適用してください。
`prepare-quarterly-valuation`は期間分割、欠損補完、標準化、モデル学習を行いません。
現在の標準依存関係には学習ライブラリが含まれていないため、
利用するライブラリとモデルは別途選定してください。

コマンドの全オプションと詳細な入出力仕様は、
[原本作成](REFERENCE.md#ds-quarterly-valuation)と
[学習前加工](REFERENCE.md#prepare-quarterly-valuation)を参照してください。
