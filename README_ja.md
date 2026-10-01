# Hapke・Shkuratov反射モデル

反射率の前進計算とShkuratovモデルのスペクトルフィッティング用Python実装。

[English](README.md)

## 環境構築

Gitとuvを導入し、次を実行する。

```sh
git clone https://github.com/ko-tanaami/undergraduate_research.git
cd undergraduate_research
uv sync --locked
```

## 使用方法

```sh
uv run --locked python hapke_model.py --help
uv run --locked python shkuratov_python/shkuratov_model.py --help
```

光学定数、観測データ、設定ファイルはローカルで用意する。研究データ、計算結果、図、発表資料はこのリポジトリで配布しない。一部の計算例と検証スクリプトにはローカルデータが必要であり、cloneだけでは実行できない。

光学定数を使う検証スクリプトは、既定で `shkuratov_python/data/opcon/`
を参照する。この既定パスは作業ディレクトリに依存しない。別の場所を使う場合は
実行前に `OPCON_DIR` 環境変数を指定する。相対指定は作業ディレクトリが基準となる。
計算例のJSONは `../data/opcon` を指定し、JSONファイルの場所を基準に解決する。
別の場所を使う場合はJSONの `opcon_dir` を明示的に変更する。
`OPCON_DIR` は検証スクリプト用であり、モデルのJSON読込には適用されない。
`data/` ディレクトリはGit管理対象外。非公開の入力データをコミットしないこと。
