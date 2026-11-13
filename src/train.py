import os
import time
import copy
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


def train_single_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs=35,
    patience=10,
    save_path=None
):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    start_time = time.perf_counter()
    best_f1 = -1.0
    best_loss = float("inf")
    best_epoch = 0
    best_weights = copy.deepcopy(model.state_dict())
    patience_cnt = 0

    history = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "epoch_times": []
    }

    model.to(device)

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()

        # treino
        model.train()
        train_loss = 0.0
        y_train_pred, y_train_true = [], []

        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            preds = torch.argmax(out, dim=1)
            y_train_pred.extend(preds.detach().cpu().numpy().tolist())
            y_train_true.extend(y.detach().cpu().numpy().tolist())

        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = accuracy_score(y_train_true, y_train_pred)
        epoch_train_f1 = f1_score(y_train_true, y_train_pred, average="macro", zero_division=0)

        # validacao
        model.eval()
        val_loss = 0.0
        y_val_pred, y_val_true = [], []

        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)

                val_loss += loss.item() * x.size(0)
                preds = torch.argmax(out, dim=1)
                y_val_pred.extend(preds.detach().cpu().numpy().tolist())
                y_val_true.extend(y.detach().cpu().numpy().tolist())

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = accuracy_score(y_val_true, y_val_pred)
        epoch_val_f1 = f1_score(y_val_true, y_val_pred, average="macro", zero_division=0)

        dt = time.perf_counter() - t0
        history["train_loss"].append(float(epoch_train_loss))
        history["train_acc"].append(float(epoch_train_acc))
        history["train_f1"].append(float(epoch_train_f1))
        history["val_loss"].append(float(epoch_val_loss))
        history["val_acc"].append(float(epoch_val_acc))
        history["val_f1"].append(float(epoch_val_f1))
        history["epoch_times"].append(float(dt))

        print(
            f"Epoca [{epoch:02d}/{epochs:02d}] "
            f"Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.3f} | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.3f} | "
            f"Val F1: {epoch_val_f1:.3f} ({dt:.2f}s)",
            flush=True
        )

        # salva melhor modelo por F1 macro de validacao
        if epoch_val_f1 > best_f1:
            best_f1 = epoch_val_f1
            best_loss = epoch_val_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            patience_cnt = 0
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(best_weights, save_path)
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"Early stopping acionado na epoca {epoch}. Melhor epoca: {best_epoch} (Val F1: {best_f1:.4f})")
                break

    total_time = time.perf_counter() - start_time
    avg_epoch_time = float(np.mean(history["epoch_times"]))

    peak_vram = 0.0
    if torch.cuda.is_available() and device.type == "cuda":
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    # restaura os pesos da melhor epoca salva
    model.load_state_dict(best_weights)

    model_size_mb = 0.0
    if save_path and os.path.exists(save_path):
        model_size_mb = os.path.getsize(save_path) / (1024 * 1024)

    return {
        "best_epoch": best_epoch,
        "best_val_f1": float(best_f1),
        "best_val_loss": float(best_loss),
        "total_epochs_trained": len(history["train_loss"]),
        "total_time_seconds": float(total_time),
        "avg_epoch_time_seconds": avg_epoch_time,
        "peak_gpu_mem_mb": float(peak_vram),
        "model_size_mb": float(model_size_mb),
        "history": history
    }
