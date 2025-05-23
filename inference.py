# inference.py
import torch
from pathlib import Path
from env.vrp import CVRPEnv, CVRPDataset
from nets.Encoder import Encoder
from nets.Decoder import DecoderStep
from option import parse_args  # 既存の option.py に合わせて

def load_models(args, device):
    # モデル定義
    encoder = Encoder(
        embed_dim=args.embedding_dim,
        n_heads=args.n_heads,
        n_layers=args.n_encoder_layers
    ).to(device)
    decoder = DecoderStep(
        embed_dim=args.embedding_dim,
        n_heads=args.n_heads,
        clip_C=args.clip_C
    ).to(device)
    # チェックポイント読み込み
    ckpt_path = args.resume or (Path(args.save_dir) / 'best_model.pth')
    print(f"[Info] Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(ckpt['encoder'])
    decoder.load_state_dict(ckpt['decoder'])
    encoder.eval(); decoder.eval()
    return encoder, decoder

def infer_one(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder, decoder = load_models(args, device)

    # データセットから1インスタンス取得
    dataset = CVRPDataset(
        num_samples=1,
        n_customers=args.n_customers,
        demand_max=args.demand_max,
        seed=args.seed
    )
    sample = dataset[0]  # dict {'depot': Tensor[2], 'loc': Tensor[N,2], 'demand': Tensor[N]}
    # 環境初期化
    env = CVRPEnv(capacity=args.vehicle_capacity)
    state = env.reset(sample)

    # 環境情報出力
    print("環境データ")
    print(f"0 : {state.depot.tolist()}")
    for i, (coord, dem) in enumerate(zip(state.loc.tolist(), state.demand.tolist()), start=1):
        print(f"{i} : {coord + [dem]}")

    # ---------- デコーダーで順次ノード選択 ----------
    route = [0]  # 最初はデポ（0）
    mask = None
    step_input = None  # 前ステップの埋め込みなど

    # ループ：全顧客を訪問し終えるまで
    for _ in range(args.n_customers):
        # エンコーダーに全ノードを埋め込み
        enc_outputs = encoder(sample['loc'].unsqueeze(0).to(device))  # [1, N, D]
        # デコーダー1ステップ実行
        logp, step_input, mask = decoder(
            enc_outputs,
            step_input,
            mask
        )  # logp: [1, N+1]（デポ含む）
        # マスク反映して最大値を選択
        probs = torch.softmax(logp, dim=-1)
        next_node = torch.argmax(probs, dim=-1).item()  # int
        route.append(next_node)
        # 環境を1ステップ進める（必要なら）
        _ = env.step(torch.tensor([next_node], device=device))

    # 出力ルート
    print("\n解")
    print(" - ".join(str(n) for n in route))

if __name__ == '__main__':
    args = parse_args()
    # デフォルトで resume が None なら best_model.pth を使うように設定しておく
    if args.resume is None:
        args.resume = None
    infer_one(args)
