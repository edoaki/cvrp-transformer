import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from option import Args, parse_args
from nets.Encoder import Encoder
from nets.Decoder import DecoderStep
from env.vrp import CVRPEnv, CVRPDataset
from rollout import rollout_sampling, rollout_greedy
from utils.baseline import Baselineb

def train(args: Args) -> None:
    """
    強化学習によるCVRPモデルの訓練ループを実行する

    Args:
        args: 実行時オプション（parse_args() から取得）
    """
    # デバイスの設定
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[Info] Using device: {device}")

    # 1. 訓練用データセットと DataLoader 準備
    train_set = CVRPDataset(
        num_samples=args.num_samples,
        n_customers=args.n_customers,
        demand_max=args.demand_max,
        seed=args.seed
    )
    print(f"[Info] Training set size: {len(train_set)} instances")
    train_loader: DataLoader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )

    # 2. 評価用データセットと DataLoader 準備
    eval_set = CVRPDataset(
        num_samples=args.eval_size,
        n_customers=args.n_customers,
        demand_max=args.demand_max,
        seed=args.eval_seed
    )
    print(f"[Info] Eval set size: {len(eval_set)} instances")
    eval_loader: DataLoader = DataLoader(
        eval_set,
        batch_size=args.eval_batch,
        shuffle=False,
        num_workers=args.num_workers
    )

    # 3. モデル初期化
    encoder = Encoder(
        n_heads=args.n_heads,
        embed_dim=args.embed_dim,
        n_layers=args.n_layers,
        node_dim=2,
        normalization=args.normalization,
        feed_forward_hidden=args.ff_hidden
    ).to(device)
    decoder = DecoderStep(
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        clip_C=args.clip_C
    ).to(device)
    print("[Info] Encoder and Decoder initialized.")

    # 4. ベースライン用モデルと環境初期化
    baseline = Baseline(encoder, decoder, CVRPEnv(capacity=args.capacity, device=device), eval_loader, device)

    # 5. Optimizer と Scheduler の設定
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=args.lr_factor,
        patience=args.lr_patience
    )
    print(f"[Info] Optimizer lr={args.lr}, Scheduler factor={args.lr_factor}, patience={args.lr_patience}")

    # 6. チェックポイント保存ディレクトリ
    os.makedirs(args.save_dir, exist_ok=True)
    best_path = os.path.join(args.save_dir, 'best_model.pth')
    # resume が指定なければ best_model.pth を使う
    ckpt_path = str(args.resume) if args.resume is not None else best_path

     # --- チェックポイントからのロード ---
    start_epoch = 1
    if os.path.isfile(ckpt_path):
        print(f"[Info] Loading checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        optimizer.load_state_dict(checkpoint.get('optimizer', {}))
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"  -> Resumed at epoch {start_epoch-1}")

    print(f"[Info] Training started for {args.epochs} epochs.")

    # 7. 訓練ループ
    for epoch in range(start_epoch, args.epochs + 1):
        encoder.train()
        decoder.train()
        epoch_loss: float = 0.0

        for batch in train_loader:
            # ベースラインコストとサンプリング推論
            cost_base = baseline.get_batch_cost(batch)
            cost_s, logp = rollout_sampling(encoder, decoder, baseline.env, batch, device)

            # 損失計算（REINFORCE）
            advantage = (cost_s - cost_base).detach()
            loss = (advantage * logp).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"[Epoch {epoch}] Train loss: {avg_loss:.4f}")

        # ベースライン評価と更新
        improved, p_val, mean_cost = baseline.evaluate_and_maybe_update(encoder, decoder)
        if improved:
            torch.save({
                'epoch': epoch,
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'p_val': p_val
            }, best_path)
            print(f"  -> Baseline updated: mean_cost={mean_cost:.2f} (p={p_val:.2e}) and model saved")

        # Scheduler へ評価コストを渡す
        scheduler.step(mean_cost)

    print("[Info] Training complete.")

if __name__ == '__main__':
    args = parse_args()
    train(args)