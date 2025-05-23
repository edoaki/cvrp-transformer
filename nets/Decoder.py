from typing import Tuple

import math
import torch
import torch.nn as nn

class DecoderStep(nn.Module):
    """
    ICLR2019『Attention, Learn to Solve Routing Problems!』論文の
    Section 3.2 に対応するデコーダーの1ステップ処理モジュール
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        clip_C: float = 10.0,
    ) -> None:
        """
        Args:
            embed_dim: ノードおよびコンテキスト埋め込みの次元 d_h
            n_heads:   マルチヘッド注意のヘッド数 M
            clip_C:    スコアクリップ定数 C (Belloら 2016)
        """
        super().__init__()
        # 埋め込み次元とヘッド数を保持
        self.embed_dim: int = embed_dim
        self.n_heads: int = n_heads
        self.clip_C: float = clip_C

        # --- コンテキスト埋め込みのための射影層 ---
        # 入力: [graph_emb; h_last; h_first] の結合 (3*d_h)
        # 出力: d_h
        self.Wq_c: nn.Linear = nn.Linear(3 * embed_dim, embed_dim)
        # コンテキスト更新用マルチヘッド注意 (Query: 1, Key/Value: N)
        self.attn_c: nn.MultiheadAttention = nn.MultiheadAttention(
            embed_dim, n_heads, batch_first=True
        )

        # --- 出力スコア計算用射影層 ---
        # Query として使う: d_h -> d_h
        self.Wq_out: nn.Linear = nn.Linear(embed_dim, embed_dim)
        # Key として使う: d_h -> d_h
        self.Wk_out: nn.Linear = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        node_emb: torch.Tensor,
        graph_emb: torch.Tensor,
        last_idx: torch.LongTensor,
        first_idx: torch.LongTensor,
        visited_mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        1ステップ分のデコーダー処理を実行し、次ノードの対数確率と
        更新後コンテキストを返す。

        Args:
            node_emb:     各ノードの埋め込み [B, N, d_h]
            graph_emb:    グラフ埋め込み (平均プーリング) [B, d_h]
            last_idx:     直前選択ノードのインデックス [B]
            first_idx:    開始ノード (π₁) のインデックス [B]
            visited_mask: 訪問済みノードをTrueにするマスク [B, N]

        Returns:
            log_p:    次ノード選択の対数確率 [B, N]
            new_ctx:  更新後コンテキスト埋め込み [B, d_h]
        """
        B, N, d_h = node_emb.size()

        # --- コンテキストベクトルの構築 ---
        h_last: torch.Tensor = node_emb[torch.arange(B), last_idx]
        h_first: torch.Tensor = node_emb[torch.arange(B), first_idx]
        ctx: torch.Tensor = torch.cat([graph_emb, h_last, h_first], dim=-1)
        q_c: torch.Tensor = self.Wq_c(ctx).unsqueeze(1)

        # --- マルチヘッド注意でコンテキスト更新 ---
        new_ctx, _ = self.attn_c(q_c, node_emb, node_emb)
        new_ctx = new_ctx.squeeze(1)

        # --- スコア計算 ---
        # Query 用射影
        q_out: torch.Tensor = self.Wq_out(new_ctx)       # [B, d_h]
        # Key 用射影
        k_out: torch.Tensor = self.Wk_out(node_emb)      # [B, N, d_h]
        # 内積によるスコア計算と正規化
        raw_scores: torch.Tensor = torch.matmul(
            q_out.unsqueeze(1),                         # [B,1,d_h]
            k_out.transpose(1, 2)                       # [B,d_h,N]
        ).squeeze(1) / math.sqrt(self.embed_dim)

        # --- スコアクリップとマスク適用 ---
        clipped: torch.Tensor = self.clip_C * torch.tanh(raw_scores)
        scores: torch.Tensor = clipped.masked_fill(visited_mask, float('-inf'))

        # --- 確率化 ---
        log_p: torch.Tensor = torch.log_softmax(scores, dim=-1)

        return log_p, new_ctx
