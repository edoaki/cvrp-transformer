from dataclasses import dataclass
import argparse
from typing import Literal
from pathlib import Path

NormalizationType = Literal['batch', 'layer']

@dataclass(frozen=True)
class Args:
    # デバイス／基本設定
    device: str
    epochs: int
    batch_size: int
    num_samples: int
    n_customers: int
    demand_max: int
    capacity: float
    seed: int

    # 評価用データ設定
    eval_seed: int
    eval_size: int
    eval_batch: int

    # DataLoader 並列数
    num_workers: int

    # モデル構成
    n_heads: int
    embed_dim: int
    n_layers: int
    ff_hidden: int
    normalization: NormalizationType
    clip_C: float

    # optimizer／scheduler
    lr: float
    weight_decay: float
    lr_factor: float
    lr_patience: int
    grad_clip: float

    # チェックポイント保存先
    save_dir: Path
    resume: Path | None


def parse_args() -> Args:
    parser = argparse.ArgumentParser("CVRP RL Training")
    parser.add_argument('--device',        type=str,   default='cuda:1')
    parser.add_argument('--epochs',        type=int,   default=100)
    parser.add_argument('--batch_size',    type=int,   default=512)
    parser.add_argument('--num_samples',   type=int,   default=1280000)
    parser.add_argument('--n_customers',   type=int,   default=20)
    parser.add_argument('--demand_max',    type=int,   default=9)
    parser.add_argument('--capacity',      type=float, default=30.0)
    parser.add_argument('--seed',          type=int,   default=42)

    parser.add_argument('--eval_seed',     type=int,   default=123)
    parser.add_argument('--eval_size',     type=int,   default=10000)
    parser.add_argument('--eval_batch',    type=int,   default=1024)

    parser.add_argument('--num_workers',   type=int,   default=4)

    parser.add_argument('--n_heads',       type=int,   default=8)
    parser.add_argument('--embed_dim',     type=int,   default=128)
    parser.add_argument('--n_layers',      type=int,   default=3)
    parser.add_argument('--ff_hidden',     type=int,   default=512)
    parser.add_argument('--normalization', type=str,   choices=['batch','layer'], default='batch')
    parser.add_argument('--clip_C',        type=float, default=10.0)

    parser.add_argument('--lr',            type=float, default=1e-4)
    parser.add_argument('--weight_decay',  type=float, default=1e-5)
    parser.add_argument('--lr_factor',     type=float, default=0.5)
    parser.add_argument('--lr_patience',   type=int,   default=5)
    parser.add_argument('--grad_clip',     type=float, default=1.0)

    parser.add_argument('--save_dir',      type=Path,  default=Path('./checkpoints'))
    parser.add_argument('--resume',        type=Path,  default=None,
                       help='読み込みたいチェックポイントのパス。未指定時は save_dir/best_model.pth を使う')
 

    args = parser.parse_args()
    return Args(
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        n_customers=args.n_customers,
        demand_max=args.demand_max,
        capacity=args.capacity,
        seed=args.seed,
        eval_seed=args.eval_seed,
        eval_size=args.eval_size,
        eval_batch=args.eval_batch,
        num_workers=args.num_workers,
        n_heads=args.n_heads,
        embed_dim=args.embed_dim,
        n_layers=args.n_layers,
        ff_hidden=args.ff_hidden,
        normalization=args.normalization,  # type: ignore
        clip_C=args.clip_C,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_factor=args.lr_factor,
        lr_patience=args.lr_patience,
        grad_clip=args.grad_clip,
        save_dir=args.save_dir,
        resume=args.resume,
    )
