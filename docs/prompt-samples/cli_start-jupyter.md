# start-jupyterコマンドの作成

## 概要

notebooks/からuv run jupyter lab --ip=0.0.0.0 --port=8888 --no-browserでjupyter labを起動する

## 作成開始条件

- 不明な点はユーザに確認すること
- 作成にとりかかる前に、作成する新規に作成するライブラリのメソッドまたはクラス、task、flowについて、ユーザとすり合わせをしてください。
- 作成にあたり、不明な点や要件があいまいな点は、ユーザに確認してから作成してください。

## 仕様

### コマンド名

`start-jupyter`

### コマンドパラメータ

| パラメータ | 規定値 | 内容 |
| --- | --- | --- |
| port | 8888 | jupyter lab起動時の`--port`オプションの値を指定する |
| hostname | なし | hostnameが指定されない場合、urlはhttp://<ホストのLAN IP>:\<port>/lab?token=...、hostnameが指定された場合は、http://\<hostname>:\<port>/lab?token=... |

### 出力

起動したjupyter labのurlを標準出力する
指定ポートが使用中なら、別ポートへ自動変更せず、エラーメッセージを出力してエラー終了する