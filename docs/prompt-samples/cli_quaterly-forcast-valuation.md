# 次期四半期単体決算のフォーキャストを出力するcli

## 概要

--valuation-dataで指定した評価日の次の四半期単体決算のフォーキャスト一覧を出力するcliを作成してください。

## 作成開始条件

- 不明な点はユーザに確認すること
- 作成にとりかかる前に、作成する新規に作成するライブラリのメソッドまたはクラス、task、flowについて、ユーザとすり合わせをしてください。
- 作成にあたり、不明な点や要件があいまいな点は、ユーザに確認してから作成してください。

## 入力パラメータ

### パラメータの種別

cliからの入力パラメータ、および入力パラメータが、内部処理で利用される際のオブジェクトタイプは、以下のとおりです。

| オプション | 既定値 | 意味 |
| --- | --- | --- |
| `--valuation-date` | 実行日の今日 | 評価日。`YYYY-MM-DD`形式 |
| `--sort-columns` | `dgrp` | 出力列名。複数指定時はオプションを繰り返し、指定順に優先 |
| `--profit` | `operating` | `prft`・`pr`・`dgrp`の利益種別。`operating`=営業利益、`ordinary`=経常利益 |
| `--sort-order` | `desc` | 指定した全列に共通の降順`desc`または昇順`asc` |
| `--codes` | `all` | 全銘柄、または整数の銘柄コード。複数銘柄はオプションを繰り返す |
| `--portfolio` | 無効 | 評価日以前の最新ポートフォリオの個別株に絞る |
| `--start-row` | `1` | 抽出・ソート後の開始行（1始まり） |
| `--end-row` | 末尾まで | 終了行（指定行を含む） |
| `--sls-min` / `--sls-max` | 制限なし | 売上高の下限 / 上限（出力と同じ単位） |
| `--profit-min` / `--profit-max` | 制限なし | `--profit`で選んだ利益の下限 / 上限（出力と同じ単位） |
| `--grsl-min` / `--grsl-max` | 制限なし | 売上成長率の下限 / 上限（%単位） |

## 出力

wequant.data_processing.KessanPl.df(以下、KessanPl.df)を加工した一覧表を標準出力する。

### 出力列と計算

| 列 | 内容 |
| --- | --- |
| `code` | 銘柄コード |
| `name` | 銘柄名 |
| `setd` | 決算期末日 |
| `annd` | 決算発表日 |
| `sls` | 四半期単体の売上高（元データの単位を保持） |
| `prft` | 四半期単体の選択した利益（元データの単位を保持） |
| `pr` | `prft / sls × 100`（%、小数点以下2桁で表示） |
| `grsl` | `(売上高 / 前年同期売上高 - 1) × 100`（%） |
| `dgrp` | `(選択した利益 - 前年同期の同じ利益) / (売上高 - 前年同期売上高)`（内部値は倍率、表示時に100倍して%表記） |
| `PER` | 株価比率で補正した予想PER（倍） |
| `divr` | 株価比率で補正した予想配当利回り（%） |
| `perf` | 今回発表翌取引日の始値から次回発表翌取引日の始値までの騰落率（%） |
| `bm` | `perf`と同じ開始日・終了日のnh225の始値による騰落率（%） |

- code: KessanPl.df["code"]
- setd: KessanPl.df["settlement_date"]
- annd: KessanPl.df["announcement_date"]
- sls: KessanPl.df["sales"]
- ordp: KessanPl.df["ordinary_profit"]
- grsl: KessanPl.df["sales"]の前年同期売上高成長率
- dgrp: (経常利益-前年同期の経常利益)/(sales - 前年同期のsales)
- PER: valuation_date前営業日以前の最新終値におけるPER  
FinancequotePl.dfは週に1回程度しかデータを取得していないので、Pricelist.dfも使って算出
- divr: valuation_date前営業日以前の最新終値における配当利回り  
FinancequotePl.dfは週に1回程度しかデータを取得していないので、Pricelist.dfも使って算出