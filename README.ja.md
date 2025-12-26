# Educational Gemma 3 (270M) 

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/) [![uv](https://img.shields.io/badge/uv-managed-purple)](https://github.com/astral-sh/uv)

![Logo](figs/logo.png)

[English](README.md)

このリポジトリは、Googleの [Gemma 3 270M](https://huggingface.co/google/gemma-3-270m) モデルを、可読性最優先で実装した教育的実装です。
本番環境での推論速度よりも、アルゴリズムがコード上でどう表現されているかの理解を目的としています。

## Beginner Friendly

Hugging Face Transformers などのライブラリは素晴らしいですが、過度な抽象化やあらゆるハードウェアへの対応コードが入り混じり、初学者が「LLMの生の挙動」を追うのは困難です。

そこでこのリポジトリでは以下の点を排除して、シンプルなコードを目指しました。

- ❌ **複雑な抽象化**: 単一ファイルで完結し、各コンポーネントが直接実装されています。
- ❌ **特殊な条件分岐**: 特定のハードウェアや古いバージョンのための分岐はありません。
- ❌ **KV Cache**: 学習用コードとしての可読性を優先し、推論最適化は行っていません。

## 必要要件 & 環境構築

本プロジェクトは、 **[uv](https://github.com/astral-sh/uv)** を前提としています。

### 1. 依存関係の同期
```powershell
uv sync
```

### 2. モデルへのアクセス権
本実装は `google/gemma-3-270m` を使用します。
1. [Hugging Face Hub](https://huggingface.co/google/gemma-3-270m) で利用規約に同意してください。
2. [Access Token](https://huggingface.co/settings/tokens) を取得し、認証を行います。

```powershell
uv run hf auth
```

## 実行方法

以下のコマンドでのサンプルコードを実行します。
ある程度のVRAMが必要です。

```powershell
uv run python main.py
```

## 内部状態の可視化（Logit Lens）

このリポジトリには、**Logit Lens** の実装が含まれており、モデルが層を重ねるごとにどのように予測を洗練していくかを可視化できます。

### 1. 解析データの生成

解析スクリプトを実行して、隠れ状態およびアテンション重みをトレースします。
これにより、`out/` ディレクトリに JSON ファイルが生成されます。

```powershell
uv run python logit_lens.py
```

### 2. エクスプローラの起動

Streamlit アプリを起動し、生成されたデータを対話的に探索します。

```powershell
uv run streamlit run logit_lens_app.py
```

以下の内容を観察できます。

* **Logit Lens**: 次トークンの確率が、レイヤーごとにどのように変化していくか
* **Attention Weights**: 各レイヤー／各ヘッドにおいて、モデルがどのトークンに注意を向けているか

## 開発・貢献について

Issue や PR は歓迎します。

## ライセンス

本コードは Hugging Face Transformers および Google の実装を参照・派生しています。
オリジナルと同様に **Apache-2.0 License** の下で提供されます。
詳細は [LICENSE](LICENSE) および [NOTICE](NOTICE) を参照してください。
