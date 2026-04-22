"""
MLP: (cross_x, cross_z, t_norm) -> (hand_x, hand_y, hand_z)

Modes
-----
  python train_trajectory.py --pretrain
      Trains on every experiment_*.csv in ../unityproject/Positions/
      Saves:  arm_model_pretrained.pt   (weights checkpoint)
              arm_model.json            (Unity-readable)

  python train_trajectory.py --finetune <csv_path>
      Loads arm_model_pretrained.pt, fine-tunes on the given CSV,
      overwrites arm_model.json. Short run (few hundred epochs).

  python train_trajectory.py
      (legacy) trains from scratch on all CSVs, same as --pretrain.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ------------------------------------------------------------
HERE          = os.path.dirname(os.path.abspath(__file__))
POSITIONS_DIR = os.path.normpath(os.path.join(HERE, "..", "unityproject", "Positions"))
OUT_JSON      = os.path.normpath(os.path.join(HERE, "..", "unityproject",
                                              "Assets", "StreamingAssets", "arm_model.json"))
CKPT_PATH     = os.path.join(HERE, "arm_model_pretrained.pt")

HIDDEN           = 32
PRETRAIN_EPOCHS  = 3000
FINETUNE_EPOCHS  = 400
LR               = 1e-3
FT_LR            = 5e-4
BATCH_SIZE       = 256


def load_csvs(paths: list[str]) -> pd.DataFrame:
    if not paths:
        sys.exit("No CSV paths given.")
    frames = []
    for f in paths:
        df = pd.read_csv(f)
        df["_src"] = os.path.basename(f)
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    print(f"[data] loaded {len(paths)} file(s), {len(data)} rows")
    return data


def build_samples(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    mask = (data["silent"] == 0) & (data["phase"] == "ActiveTrial")
    df = data[mask].copy()
    if df.empty:
        sys.exit("No SOUND/ActiveTrial rows found.")

    X, Y = [], []
    for (src, trial), g in df.groupby(["_src", "trial"]):
        g = g.sort_values("timestamp")
        t0, t1 = g["timestamp"].iloc[0], g["timestamp"].iloc[-1]
        dur = t1 - t0
        if dur <= 1e-3:
            continue
        t_norm = ((g["timestamp"] - t0) / dur).values
        cx = g["cross_x"].values
        cz = g["cross_z"].values
        hx = g["hand_unity_x"].values
        hy = g["hand_unity_y"].values
        hz = g["hand_unity_z"].values
        for i in range(len(g)):
            X.append([cx[i], cz[i], t_norm[i]])
            Y.append([hx[i], hy[i], hz[i]])
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    print(f"[data] training samples: {len(X)}")
    return X, Y


class MLP(nn.Module):
    def __init__(self, in_dim=3, hidden=HIDDEN, out_dim=3):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        return self.l3(x)


def run_training(X, Y, model: MLP, stats: dict, epochs: int, lr: float):
    Xn = (X - stats["in_mean"]) / stats["in_std"]
    Yn = (Y - stats["out_mean"]) / stats["out_std"]
    Xt = torch.from_numpy(Xn.astype(np.float32))
    Yt = torch.from_numpy(Yn.astype(np.float32))

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    n = len(Xt)
    for epoch in range(epochs):
        idx = torch.randperm(n)[:BATCH_SIZE]
        loss = loss_fn(model(Xt[idx]), Yt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if (epoch + 1) % max(1, epochs // 6) == 0:
            with torch.no_grad():
                full = loss_fn(model(Xt), Yt).item()
            print(f"[train] epoch {epoch+1:5d}  batch={loss.item():.5f}  full={full:.5f}")


def compute_stats(X: np.ndarray, Y: np.ndarray) -> dict:
    return dict(
        in_mean  = X.mean(0),
        in_std   = X.std(0)  + 1e-6,
        out_mean = Y.mean(0),
        out_std  = Y.std(0)  + 1e-6,
    )


def save_checkpoint(model: MLP, stats: dict, path: str) -> None:
    torch.save({
        "state_dict": model.state_dict(),
        "in_mean":  stats["in_mean"].tolist(),
        "in_std":   stats["in_std"].tolist(),
        "out_mean": stats["out_mean"].tolist(),
        "out_std":  stats["out_std"].tolist(),
    }, path)
    print(f"[ckpt] saved {path}")


def load_checkpoint(path: str) -> tuple[MLP, dict]:
    if not os.path.exists(path):
        sys.exit(f"Checkpoint not found: {path}. Run --pretrain first.")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = MLP()
    model.load_state_dict(ck["state_dict"])
    stats = dict(
        in_mean  = np.asarray(ck["in_mean"],  dtype=np.float32),
        in_std   = np.asarray(ck["in_std"],   dtype=np.float32),
        out_mean = np.asarray(ck["out_mean"], dtype=np.float32),
        out_std  = np.asarray(ck["out_std"],  dtype=np.float32),
    )
    return model, stats


def export_json(model: MLP, stats: dict, path: str) -> None:
    def layer(lin: nn.Linear, act: str) -> dict:
        W = lin.weight.detach().cpu().numpy().astype(np.float32)
        b = lin.bias.detach().cpu().numpy().astype(np.float32)
        return dict(in_dim=int(W.shape[1]), out_dim=int(W.shape[0]),
                    W=W.flatten().tolist(), b=b.tolist(), act=act)

    obj = dict(
        layers=[layer(model.l1, "tanh"),
                layer(model.l2, "tanh"),
                layer(model.l3, "linear")],
        in_mean = np.asarray(stats["in_mean"]).tolist(),
        in_std  = np.asarray(stats["in_std"]).tolist(),
        out_mean= np.asarray(stats["out_mean"]).tolist(),
        out_std = np.asarray(stats["out_std"]).tolist(),
        input_names =["cross_x", "cross_z", "t_norm"],
        output_names=["hand_x", "hand_y", "hand_z"],
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)
    print(f"[out]  wrote {path}")


# ------------------------------------------------------------
def cmd_pretrain():
    csvs = sorted(glob.glob(os.path.join(POSITIONS_DIR, "experiment_*.csv")))
    data = load_csvs(csvs)
    X, Y = build_samples(data)
    stats = compute_stats(X, Y)
    model = MLP()
    run_training(X, Y, model, stats, PRETRAIN_EPOCHS, LR)
    save_checkpoint(model, stats, CKPT_PATH)
    export_json(model, stats, OUT_JSON)


def cmd_finetune(csv_path: str):
    if not os.path.exists(csv_path):
        sys.exit(f"CSV not found: {csv_path}")
    model, stats = load_checkpoint(CKPT_PATH)
    data = load_csvs([csv_path])
    X, Y = build_samples(data)
    # Keep pretrain stats for normalization so the feature space matches
    run_training(X, Y, model, stats, FINETUNE_EPOCHS, FT_LR)
    export_json(model, stats, OUT_JSON)
    print("[done] personalized model ready.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain", action="store_true",
                    help="train from scratch on all CSVs in Positions/")
    ap.add_argument("--finetune", type=str, default=None,
                    help="fine-tune the pretrained checkpoint on ONE CSV")
    args = ap.parse_args()

    if args.finetune:
        cmd_finetune(args.finetune)
    else:
        # default = pretrain (also covers legacy no-arg run)
        cmd_pretrain()


if __name__ == "__main__":
    main()
