
# CVRP-RL: Transformer ベース強化学習による CVRP ソルバー

本リポジトリは，論文 [Attention, Learn to Solve Routing Problems!](https://arxiv.org/abs/1803.08475) の手法をベースに，**CVRP**（Capacitated Vehicle Routing Problem）を Transformer 構造＋REINFORCE で解くための最小構成実装です。

> Kool, W., van Hoof, H., & Welling, M. (2018). Attention, Learn to Solve Routing Problems!  
> https://arxiv.org/abs/1803.08475

---

## 特徴

- Transformer エンコーダ／デコーダモデルを用いた Pointer Network  
- REINFORCE（モンテカルロ・ポリシーグラディエント）によるエンドツーエンド学習  
- 最小限の依存関係で，動作確認済みのサンプルコードを提供  

---

## ディレクトリ構成

```

.
├── checkpoints/          # 学習済モデルの保存先
├── data/                 # （任意）カスタムデータ置き場
├── env/                  # CVRP 環境・データセット定義
│   ├── vrp.py
│   └── ...
├── nets/                 # Encoder, DecoderStep モジュール
│   ├── Encoder.py
│   └── Decoder.py
├── train.py              # 学習スクリプト
├── inference.py          # 推論スクリプト（1インスタンス出力）
├── option.py             # コマンドライン引数定義
├── requirements.txt      # 必要パッケージ
└── README.md

````

---

## 動作環境

- Python 3.8 以上
- PyTorch 1.12 以上
- CUDA（GPU を使う場合。CPU 実行も可能）

GPU 版 CUDA サポート付き PyTorch を入れる場合は，
公式サイトのコマンド例を参照してください：
[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

---

## インストール方法

```bash
# 仮想環境作成（venv / conda などお好みで）
python -m venv venv
source venv/bin/activate

# 必要パッケージのインストール
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 学習（Training）

```bash
# data は自動生成。引数でデータ数や顧客数を調整可
python train.py 
```

* `--save_dir` 以下に `best_model.pth` が自動生成されます。
* `--resume` を指定しない場合は既存の `checkpoints/best_model.pth` を再利用して学習を再開します。

---

## 推論（Inference）

```bash
python inference.py 
```

* デフォルトで `checkpoints/best_model.pth` を読み込み
* １インスタンス分の環境情報と，モデルが選択したルートを標準出力します

**出力例**

```
環境データ
0 : [0.5, 0.7,   0]    # 0 はデポ (最後の値は需要 0)
1 : [0.1, 0.2,   3]
2 : [0.2, 0.4,   5]
…
解
0 - 2 - 1 - 3 - 0
```

## ライセンス

MIT License

---
