from typing import Optional, Literal, Tuple

import torch
import torch.nn as nn

# 正規化方式の定義
NormalizationType = Literal["batch", "layer"]

class Encoder(nn.Module):
    """
    グラフノードの埋め込みを行うTransformerライクなエンコーダーモジュール
    """

    def __init__(
        self,
        n_heads: int,
        embed_dim: int,
        n_layers: int,
        node_dim: Optional[int] = None,
        normalization: NormalizationType = "batch",
        feed_forward_hidden: int = 512,
    ) -> None:
        """
        Args:
            n_heads: 注意ヘッド数
            embed_dim: 埋め込み次元
            n_layers: レイヤー数
            node_dim: 入力特徴量の次元（Noneの場合はembed_dimと同じ）
            normalization: "batch" または "layer" 正規化方式
            feed_forward_hidden: Feed-Forwardサブレイヤーの隠れ層サイズ
        """
        super().__init__()
        # 入力特徴量次元の決定
        self.node_dim: int = node_dim or embed_dim
        self.embed_dim: int = embed_dim
        self.norm_type: NormalizationType = normalization

        # (1) 入力特徴量から埋め込みへの線形変換
        self.input_proj: nn.Linear = nn.Linear(self.node_dim, embed_dim)

        # (2) マルチヘッド注意機構とFeed-Forward層の準備
        self.attention_layers: nn.ModuleList = nn.ModuleList(
            [nn.MultiheadAttention(embed_dim, n_heads, batch_first=True) for _ in range(n_layers)]
        )
        self.ff_layers: nn.ModuleList = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim),
                )
                for _ in range(n_layers)
            ]
        )

        # (3) 正規化層の準備 (BatchNorm1d または LayerNorm)
        if normalization == "batch":
            self.norm_mha: nn.ModuleList = nn.ModuleList(
                [nn.BatchNorm1d(embed_dim) for _ in range(n_layers)]
            )
            self.norm_ff: nn.ModuleList = nn.ModuleList(
                [nn.BatchNorm1d(embed_dim) for _ in range(n_layers)]
            )
        elif normalization == "layer":
            self.norm_mha: nn.ModuleList = nn.ModuleList(
                [nn.LayerNorm(embed_dim) for _ in range(n_layers)]
            )
            self.norm_ff: nn.ModuleList = nn.ModuleList(
                [nn.LayerNorm(embed_dim) for _ in range(n_layers)]
            )
        else:
            raise ValueError(f"不明な正規化方式: {normalization}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 入力テンソル (batch_size, n_nodes, node_dim)
        Returns:
            node_embeddings: 各ノードの埋め込み (batch_size, n_nodes, embed_dim)
            graph_embedding: グラフ全体の埋め込み (batch_size, embed_dim)
        """
        # 入力から初期ノード埋め込みを生成
        h: torch.Tensor = self.input_proj(x)

        for idx, (attn, ff) in enumerate(zip(self.attention_layers, self.ff_layers)):
            # --- マルチヘッド注意サブレイヤー ---
            residual = h
            attn_out, _ = attn(h, h, h)  # Query, Key, Value 全て h を使用
            h = residual + attn_out      # 残差結合

            # 正規化の適用
            if self.norm_type == "batch":
                b, n, e = h.shape
                h = self.norm_mha[idx](h.view(b * n, e)).view(b, n, e)
            else:
                h = self.norm_mha[idx](h)

            # --- Feed-Forwardサブレイヤー ---
            residual = h
            ff_out: torch.Tensor = ff(h)
            h = residual + ff_out         # 残差結合

            # 正規化の適用
            if self.norm_type == "batch":
                b, n, e = h.shape
                h = self.norm_ff[idx](h.view(b * n, e)).view(b, n, e)
            else:
                h = self.norm_ff[idx](h)

        # --- グラフ全体の埋め込みを平均プーリングで取得 ---
        graph_embedding: torch.Tensor = h.mean(dim=1)

        return h, graph_embedding