"""Train discrete operand/operator routing and a classification head."""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model import OSLGN
from utils import evaluate, load_config, load_dataset, set_seed, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/oslgn.yaml")
    parser.add_argument("--data", help="NPZ with x_train/y_train/x_val/y_val")
    parser.add_argument("--mnist", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = load_config("configs/smoke.yaml" if args.smoke else args.config)
    set_seed(cfg["train"]["seed"])
    torch.set_num_threads(cfg["train"].get("num_threads", 1))
    model = OSLGN(**cfg["model"]).to(args.device)
    train_loader = DataLoader(load_dataset(cfg, "train", args.data, args.mnist, args.download),
                              batch_size=cfg["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(load_dataset(cfg, "val", args.data, args.mnist, args.download), batch_size=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["learning_rate"])
    initial = evaluate(model, train_loader, args.device)
    best, history = float("inf"), []
    save_dir = Path(cfg["train"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x.to(args.device))
            loss = torch.nn.functional.cross_entropy(logits, y.to(args.device))
            loss.backward()
            if cfg["train"].get("clip_max_norm", 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["clip_max_norm"])
            optimizer.step()
        scores = evaluate(model, val_loader, args.device)
        history.append({"epoch": epoch + 1, **scores})
        checkpoint = {"config": cfg, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "epoch": epoch + 1, "metrics": scores}
        torch.save(checkpoint, save_dir / "last.pth")
        if scores["loss"] < best:
            best = scores["loss"]
            torch.save(checkpoint, save_dir / "best.pth")
        print(f"epoch {epoch + 1}: {scores}")
    final = evaluate(model, train_loader, args.device)
    write_json(save_dir / "metrics.json", {"task": "mnist" if args.mnist else "npz" if args.data else "synthetic_boolean",
               "initial_train": initial, "final_train": final, "history": history})
    if args.smoke and final["loss"] >= initial["loss"]:
        raise RuntimeError("smoke optimization did not decrease training loss")


if __name__ == "__main__":
    main()
