import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import torch

from src.data_loader import CLASS_NAMES, IDX_TO_CLASS


def evaluate_model(model, data_loader, device):
    model.eval()
    preds_all = []
    targets_all = []
    probs_all = []
    paths_all = []

    with torch.no_grad():
        for x, y, paths in data_loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            probs = torch.softmax(out, dim=1)
            preds = torch.argmax(probs, dim=1)

            preds_all.extend(preds.cpu().numpy().tolist())
            targets_all.extend(y.cpu().numpy().tolist())
            probs_all.extend(probs.cpu().numpy().tolist())
            paths_all.extend(paths)

    preds_all = np.array(preds_all)
    targets_all = np.array(targets_all)
    probs_all = np.array(probs_all)

    acc = float(accuracy_score(targets_all, preds_all))
    f1_macro = float(f1_score(targets_all, preds_all, average="macro", zero_division=0))

    report = classification_report(
        targets_all,
        preds_all,
        target_names=CLASS_NAMES,
        labels=list(range(len(CLASS_NAMES))),
        output_dict=True,
        zero_division=0
    )

    cm = confusion_matrix(
        targets_all,
        preds_all,
        labels=list(range(len(CLASS_NAMES)))
    )

    # lista os erros de predicao no teste
    errors = []
    for i in range(len(targets_all)):
        if preds_all[i] != targets_all[i]:
            errors.append({
                "image_path": paths_all[i],
                "true_class": IDX_TO_CLASS[targets_all[i]],
                "pred_class": IDX_TO_CLASS[preds_all[i]],
                "confidence": float(probs_all[i][preds_all[i]])
            })

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "errors": errors,
        "predictions": preds_all.tolist(),
        "targets": targets_all.tolist(),
        "probabilities": probs_all.tolist(),
        "paths": paths_all
    }


def plot_confusion_matrix(cm, title, output_path, class_names=CLASS_NAMES):
    cm = np.array(cm)
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = np.true_divide(cm, row_sums)
        cm_norm[~np.isfinite(cm_norm)] = 0.0

    # anota quantidade e porcentagem na mesma celula
    labels = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            labels[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"

    plt.figure(figsize=(7.5, 6))
    sns.heatmap(
        cm_norm,
        annot=labels,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True
    )
    plt.title(title, fontsize=12, pad=10)
    plt.ylabel("Classe Real")
    plt.xlabel("Classe Prevista")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_learning_curves(history, title, output_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # perda
    ax1.plot(epochs, history["train_loss"], label="Treino", color="#1f77b4")
    ax1.plot(epochs, history["val_loss"], label="Validação", color="#ff7f0e")
    ax1.set_title("Curva de Perda")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    # acuracia
    ax2.plot(epochs, history["train_acc"], label="Treino", color="#1f77b4")
    ax2.plot(epochs, history["val_acc"], label="Validação", color="#ff7f0e")
    ax2.set_title("Curva de Acurácia")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Acurácia")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
