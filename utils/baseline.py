from typing import Tuple, List

import copy
import torch
from torch.utils.data import DataLoader
from scipy.stats import ttest_rel

# 環境・ロールアウト関数は外部からインポートされる想定
from rollout import rollout_greedy

class Baseline:
    """
    Rollout-based ベースライン管理クラス

    Attributes:
        encoder: ベースライン用エンコーダモデル
        decoder: ベースライン用デコーダモデル
        env: 環境オブジェクト (CVRPEnv など)
        eval_loader: 評価用データローダ
        eval_costs: 評価セットにおける現在のコスト配列 Tensor [N_eval]
        device: 処理用デバイス
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        env,
        eval_loader: DataLoader,
        device: torch.device,
    ) -> None:
        """
        Args:
            encoder: ベースライン用のエンコーダモデル (深いコピーします)
            decoder: ベースライン用のデコーダモデル (深いコピーします)
            env: 環境オブジェクト (reset, step, get_mask を提供)
            eval_loader: 評価用データ loader
            device: モデル・計算用デバイス
        """
        print("[Info] Baseline initialized.")
        # モデルを複製して eval モードに設定
        self.encoder: torch.nn.Module = copy.deepcopy(encoder).to(device).eval()
        self.decoder: torch.nn.Module = copy.deepcopy(decoder).to(device).eval()
        self.env = env
        self.eval_loader: DataLoader = eval_loader
        self.device: torch.device = device

        # 初期評価セットのコストを算出
        self.eval_costs: torch.Tensor = self._evaluate_eval_set()
        mean = self.eval_costs.mean().item()
        std = self.eval_costs.std().item()
        print(f"[Info] Baseline costs: {mean:.2f} ± {std:.2f}")

    def _evaluate_eval_set(self) -> torch.Tensor:
        """
        評価セット全体をロールアウトしてコストを取得

        Returns:
            Tensor: 評価インスタンスごとのコスト [N_eval]
        """
        costs: List[torch.Tensor] = []
        for batch in self.eval_loader:
            cost_batch = rollout_greedy(self.encoder, self.decoder, self.env, batch, self.device)
            costs.append(cost_batch)
        # バッチ結合
        return torch.cat(costs, dim=0)

    def get_batch_cost(self, batch) -> torch.Tensor:
        """
        指定バッチの推論コストを取得

        Args:
            batch: eval_loader と同形式のバッチ
        Returns:
            Tensor: バッチ内のコスト [B]
        """
        return rollout_greedy(self.encoder, self.decoder, self.env, batch, self.device)

    def evaluate_and_maybe_update(
        self,
        model_encoder: torch.nn.Module,
        model_decoder: torch.nn.Module,
        alpha: float = 0.05,
    ) -> Tuple[bool, float, float]:
        """
        新モデルの評価セット性能をベースラインと比較し、t検定で有意に改善していれば
        ベースラインモデルを更新する

        Args:
            model_encoder: 評価対象のエンコーダモデル
            model_decoder: 評価対象のデコーダモデル
            alpha: 有意水準 (p_val < alpha で改善とみなす)
        Returns:
            improved: bool -- ベースライン更新の可否
            p_val: float -- t検定の p 値
            mean_cost: float -- 新モデルの平均コスト
        """
        # モデルを評価モードに切り替え
        model_encoder.eval()
        model_decoder.eval()

        # 新モデルで評価セットを推論
        new_costs_list: List[torch.Tensor] = []
        for batch in self.eval_loader:
            cost_batch = rollout_greedy(model_encoder, model_decoder, self.env, batch, self.device)
            new_costs_list.append(cost_batch)
        new_costs: torch.Tensor = torch.cat(new_costs_list, dim=0)

        # 対対応 t 検定 (baseline > model の仮説検定)
        t_stat, p_val = ttest_rel(
            self.eval_costs.cpu().numpy(),
            new_costs.cpu().numpy(),
            alternative="greater",
        )
        mean_cost = float(new_costs.mean().item())

        # 有意に改善している場合はモデルを更新
        improved: bool = (p_val < alpha)
        if improved:
            # 重みのみをコピー
            self.encoder.load_state_dict(model_encoder.state_dict())
            self.decoder.load_state_dict(model_decoder.state_dict())
            # コスト記録を更新 (detach して GPU メモリ解放)
            self.eval_costs = new_costs.detach().to(self.device)

        return improved, p_val, mean_cost
