from typing import Tuple, Literal, TypedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.distributions import Categorical

ModeType = Literal['greedy', 'sampling']

# バッチ入力のキーと型を明示するTypedDict
class BatchDict(TypedDict):
    depot: torch.Tensor
    loc: torch.Tensor
    demand: torch.Tensor


def rollout_sampling(
    encoder: nn.Module,
    decoder: nn.Module,
    env,
    batch: BatchDict,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    確率的ロールアウトによる推論を実行し、
    各インスタンスの合計距離と対数確率の合計を返す。

    Args:
        encoder: ノード・グラフ埋め込みを生成するエンコーダモデル
        decoder: 次ノード選択の対数確率を出力するデコーダモデル
        env: 環境オブジェクト（reset, get_mask, step を提供）
        batch: BatchDict
        device: 使用デバイス

    Returns:
        total_cost: Tensor [B] 各インスタンスの移動距離合計
        logp_sum: Tensor [B] 各ステップの log-prob 和
    """
    batch = {k: v.to(device) for k, v in batch.items()}
    state = env.reset(batch)

    x = state.all_loc
    node_emb, graph_emb = encoder(x)

    B = node_emb.size(0)
    last_idx = torch.zeros(B, dtype=torch.long, device=device)
    first_idx = last_idx.clone()
    visited_mask = ~state.get_mask()

    total_cost = torch.zeros(B, device=device)
    logp_list = []
    done = torch.zeros(B, dtype=torch.bool, device=device)

    while not done.all():
        log_p, _ = decoder(node_emb, graph_emb, last_idx, first_idx, visited_mask)
        dist_cat = Categorical(logits=log_p)
        action = dist_cat.sample()
        logp_list.append(dist_cat.log_prob(action))

        dist_step, _ = state.step(action)
        total_cost += dist_step
        done = state.done

        visited_mask = ~state.get_mask()
        last_idx = action

    logp_sum = torch.stack(logp_list, dim=1).sum(dim=1)
    return total_cost, logp_sum


def rollout_greedy(
    encoder: nn.Module,
    decoder: nn.Module,
    env,
    batch: BatchDict,
    device: torch.device,
) -> torch.Tensor:
    """
    グリーディーロールアウトによる推論を実行し、
    各インスタンスの合計距離を返す。

    Args:
        encoder: エンコーダモデル
        decoder: デコーダモデル
        env: 環境オブジェクト
        batch: BatchDict
        device: 使用デバイス

    Returns:
        total_cost: Tensor [B] 各インスタンスの移動距離合計
    """
    with torch.no_grad():
        batch = {k: v.to(device) for k, v in batch.items()}
        state = env.reset(batch)

        x = state.all_loc
        node_emb, graph_emb = encoder(x)

        B = node_emb.size(0)
        last_idx = torch.zeros(B, dtype=torch.long, device=device)
        first_idx = last_idx.clone()
        visited_mask = ~state.get_mask()

        total_cost = torch.zeros(B, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)

        while not done.all():
            log_p, _ = decoder(node_emb, graph_emb, last_idx, first_idx, visited_mask)
            action = log_p.argmax(dim=-1)

            dist_step, _ = state.step(action)
            total_cost += dist_step
            done = state.done

            visited_mask = ~state.get_mask()
            last_idx = action

    return total_cost


def evaluate_dataset(
    encoder: nn.Module,
    decoder: nn.Module,
    env,
    dataloader: DataLoader,
    device: torch.device,
    mode: ModeType = 'greedy',
) -> torch.Tensor:
    """
    データローダ全インスタンスについて指定モードで推論を実行し、
    合計コストを返す。

    Args:
        encoder: エンコーダモデル
        decoder: デコーダモデル
        env: 環境オブジェクト
        dataloader: 評価用DataLoader
        device: 使用デバイス
        mode: 'greedy' or 'sampling'

    Returns:
        Tensor [N_eval] 全インスタンスの移動距離合計
    """
    fn = rollout_greedy if mode == 'greedy' else rollout_sampling
    all_costs = []
    for batch in dataloader:
        cost = fn(encoder, decoder, env, batch, device)
        # samplingはタプル戻りのため、コストを抽出
        if isinstance(cost, tuple):
            cost = cost[0]
        all_costs.append(cost)
    return torch.cat(all_costs, dim=0)
