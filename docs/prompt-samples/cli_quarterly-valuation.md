# 四半期ベースでの銘柄評価一覧を出力するcli

## 概要

四半期ベースの銘柄評価一覧を出力するcliを作成してください。

## 作成開始条件

- 不明な点はユーザに確認すること
- 作成にとりかかる前に、作成する新規に作成するライブラリのメソッドまたはクラス、task、flowについて、ユーザとすり合わせをしてください。
- 作成にあたり、不明な点や要件があいまいな点は、ユーザに確認してから作成してください。


## 入力パラメータ

### 入力パラメータの種別

cliからの入力パラメータ、および入力パラメータが、内部処理で利用される際のオブジェクトタイプは、以下のとおりです。

- valuation_data: date = date.today()
- sort_columns: list = ["dgrp"]
- sort_order: Literal["ask", "desc"] = "desc"

## 出力

wequant.data_processing.KessanPl.df(以下、KessanPl.df)を加工した一覧表を標準出力する。

### 出力列

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

### 出力レコード

以下をすべて満たすレコードを出力する

- KessanPl.df["settlement_type"]=="四"
- KessanPl.df["announcement_date"]が、valuation_dateより93日前以降
- KessanPl.df["code"]が複数ある場合は、新しい方のレコードのみ

## コマンド名の合意

本コマンドのコマンド名を提案してください。  
コマンド名はユーザと合意してください。



