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
