# 次期四半期単体決算のフォーキャストを出力するcli

## 概要

quarterly-valuationコマンドの要領で、機械学習用のベースデータセットを作成して保存するコマンドを作成してください。

quarterly-valuationコマンドは、特定の--valuation-dateを起点に特定期間の四半期決算のみを出力対象してますが、ds-quarterly-valuationは後で機械学習のデータセットとして活用したいので、全期間の四半期決算を出力対象とします。

## 作成開始条件

- 不明な点はユーザに確認すること
- 作成にとりかかる前に、作成する新規に作成するライブラリのメソッドまたはクラス、task、flowについて、ユーザとすり合わせをしてください。
- 作成にあたり、不明な点や要件があいまいな点は、ユーザに確認してから作成してください。

## コマンド名

ds-quaterly-valuation

## 入力パラメータ

### 入力パラメータの種別

cliからの入力パラメータ、および入力パラメータが、内部処理で利用される際のオブジェクトタイプは、以下のとおりです。

| オプション | 既定値 | 意味 |
| --- | --- | --- |
| `--profit` | `operating` | `prft`・`pr`・`dgrp`の利益種別。`operating`=営業利益、`ordinary`=経常利益 |
| `--perf-period` | `quarter` | `perf`・`bm`の期間。`quarter`=今回発表翌取引日→次回発表翌取引日、`announcement`=次回発表当日→翌取引日（始値） |

## 出力

wequant.data_processing.KessanPl.df(以下、KessanPl.df)を加工した一覧表を標準出力する。

### 出力列

- code: KessanPl.df["code"]
- name: codeの会社名
- setd: KessanPl.df["settlement_date"]
- annd: KessanPl.df["announcement_date"]
- qtr: 会計四半期が年決算の第何クォータか。
- sls: KessanPl.df["sales"]
- prft: 入力パラメータ--profitで指定した利益
- grsl: KessanPl.df["sales"]の前年同期売上高成長率
- dgrp: (利益-前年同期の利益)/(sales - 前年同期のsales)
- ngrpr: quarterly-valuationと同じ計算方法で算出
- PER: valuation_date前営業日以前の最新終値におけるPER  
FinancequotePl.dfは週に1回程度しかデータを取得していないので、Pricelist.dfも使って算出
- divr: valuation_date前営業日以前の最新終値における配当利回り  
FinancequotePl.dfは週に1回程度しかデータを取得していないので、Pricelist.dfも使って算出
- perf: --perf-periodで指定した期間の株価騰落率
- bm: --perf-periodで指定した期間の日経平均騰落率

### 出力レコード

以下をすべて満たすレコードを出力する

- KessanPl.df["settlement_type"]=="四"
- ["code","settd"]の列の値の組み合わせがユニーク
- "perf"列、"bm"列がnullの列は除外

### 出力形式

全レコードをparquet型式のファイルとして出力
head10行、tail10行を標準出力

### 出力ファイル名

ds-quarterly-valuation-<--profit>-<--perf-period>-datetime.now().strftime("%Y%m%d_%H%M%S").parquet



### 提案依頼

以下を提案してください。提案内容は、ユーザからの合意をとってください。

- parquetファイル出力先フォルダパス