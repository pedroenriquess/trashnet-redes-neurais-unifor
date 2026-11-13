import os
import sys
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import get_dataloaders, CLASS_NAMES
from src.models import TrashMLP, TrashCNN, count_parameters
from src.train import train_single_model
from src.evaluate import evaluate_model, plot_confusion_matrix, plot_learning_curves


def fix_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_experiment(cfg, device):
    exp_id = cfg["id"]
    name = cfg["name"]
    model_type = cfg["model_type"]
    res = cfg["resolution"] # (largura, altura)
    use_weights = cfg["use_weights"]
    batch_size = cfg.get("batch_size", 32)
    lr = cfg.get("lr", 0.001)
    epochs = cfg.get("epochs", 35)
    patience = cfg.get("patience", 10)
    seed = cfg.get("seed", 42)

    print(f"\n--- Experimento {exp_id}: {name} ({res[0]}x{res[1]}) | Pesos: {use_weights} ---", flush=True)

    fix_seeds(seed)

    train_loader, val_loader, test_loader, class_weights = get_dataloaders(
        target_size=res,
        is_mlp=(model_type == "mlp"),
        batch_size=batch_size
    )

    if model_type == "mlp":
        model = TrashMLP(input_size=res, num_classes=len(CLASS_NAMES))
    else:
        model = TrashCNN(num_classes=len(CLASS_NAMES))

    n_params = count_parameters(model)
    print(f"Parametros: {n_params:,}")

    weights_tensor = class_weights.to(device) if use_weights else None
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    weights_path = str(PROJECT_ROOT / "models" / f"exp_{exp_id}_{model_type}.pt")

    # loop de treino
    train_res = train_single_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        patience=patience,
        save_path=weights_path
    )

    # avaliacao final no teste
    test_eval = evaluate_model(model, test_loader, device=device)
    print(f"Teste -> Acc: {test_eval['accuracy']:.4f} | F1-Macro: {test_eval['f1_macro']:.4f}\n")

    # salva figuras
    plot_learning_curves(
        train_res["history"],
        title=f"Exp {exp_id} - {name}",
        output_path=str(PROJECT_ROOT / "results" / f"curves_exp_{exp_id}.png")
    )

    plot_confusion_matrix(
        test_eval["confusion_matrix"],
        title=f"Matriz de Confusao - Exp {exp_id}",
        output_path=str(PROJECT_ROOT / "results" / f"cm_exp_{exp_id}.png")
    )

    summary = {
        "id": exp_id,
        "name": name,
        "model_type": model_type,
        "resolution": f"{res[0]}x{res[1]}",
        "resolution_tuple": list(res),
        "use_class_weights": use_weights,
        "parameters": n_params,
        "model_size_mb": train_res["model_size_mb"],
        "epochs_trained": train_res["total_epochs_trained"],
        "best_epoch": train_res["best_epoch"],
        "best_val_f1": train_res["best_val_f1"],
        "best_val_loss": train_res["best_val_loss"],
        "total_time_seconds": train_res["total_time_seconds"],
        "avg_epoch_time_seconds": train_res["avg_epoch_time_seconds"],
        "peak_gpu_mem_mb": train_res["peak_gpu_mem_mb"],
        "test_accuracy": test_eval["accuracy"],
        "test_f1_macro": test_eval["f1_macro"],
        "test_report": test_eval["classification_report"],
        "confusion_matrix": test_eval["confusion_matrix"],
        "errors": test_eval["errors"]
    }

    with open(PROJECT_ROOT / "results" / f"summary_exp_{exp_id}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(PROJECT_ROOT / "results" / f"history_exp_{exp_id}.json", "w", encoding="utf-8") as f:
        json.dump(train_res["history"], f, indent=2)

    return summary


def plot_resolution_effect(results):
    # compara apenas as CNNs sem pesos (exp 3, 5 e 6)
    cnn_exps = [r for r in results if r["model_type"] == "cnn" and not r["use_class_weights"]]
    cnn_exps.sort(key=lambda x: x["resolution_tuple"][0])

    labels = [r["resolution"] for r in cnn_exps]
    accs = [r["test_accuracy"] * 100 for r in cnn_exps]
    f1s = [r["test_f1_macro"] * 100 for r in cnn_exps]
    avg_t = [r["avg_epoch_time_seconds"] for r in cnn_exps]
    tot_t = [r["total_time_seconds"] for r in cnn_exps]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(labels))
    w = 0.35

    # metricas por resolucao
    ax1.bar(x - w/2, accs, w, label="Acurácia (%)", color="#2ca02c")
    ax1.bar(x + w/2, f1s, w, label="F1-Macro (%)", color="#1f77b4")
    ax1.set_xlabel("Resolução")
    ax1.set_ylabel("%")
    ax1.set_title("Desempenho no Teste vs Resolução")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 100)
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")

    # tempo por resolucao
    ax2.bar(x - w/2, avg_t, w, label="Tempo/Época (s)", color="#ff7f0e")
    ax2.bar(x + w/2, tot_t, w, label="Tempo Total (s)", color="#d62728")
    ax2.set_xlabel("Resolução")
    ax2.set_ylabel("Segundos")
    ax2.set_title("Custo Computacional vs Resolução")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.4, axis="y")

    plt.tight_layout()
    out = str(PROJECT_ROOT / "results" / "resolution_comparison.png")
    plt.savefig(out, dpi=300)
    plt.close()


def save_error_panel(best_exp):
    errors = best_exp.get("errors", [])
    if not errors:
        return

    samples = errors[:6]
    fig, axes = plt.subplots(1, len(samples), figsize=(3.5 * len(samples), 3.8))
    if len(samples) == 1:
        axes = [axes]

    for err, ax in zip(samples, axes):
        im = Image.open(err["image_path"]).convert("RGB")
        ax.imshow(im)
        ax.set_title(
            f"Real: {err['true_class']}\nPrev: {err['pred_class']} ({err['confidence']*100:.1f}%)",
            fontsize=9
        )
        ax.axis("off")

    plt.suptitle(f"Erros de Teste — {best_exp['name']}", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "results" / "best_model_errors.png", dpi=300)
    plt.close()

    with open(PROJECT_ROOT / "results" / "error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # os 6 experimentos solicitados na atividade
    experiments = [
        {"id": 1, "name": "MLP (128x96) Sem Pesos", "model_type": "mlp", "resolution": (128, 96), "use_weights": False},
        {"id": 2, "name": "MLP (128x96) Com Pesos", "model_type": "mlp", "resolution": (128, 96), "use_weights": True},
        {"id": 3, "name": "CNN (128x96) Sem Pesos", "model_type": "cnn", "resolution": (128, 96), "use_weights": False},
        {"id": 4, "name": "CNN (128x96) Com Pesos", "model_type": "cnn", "resolution": (128, 96), "use_weights": True},
        {"id": 5, "name": "CNN (256x192) Sem Pesos", "model_type": "cnn", "resolution": (256, 192), "use_weights": False},
        {"id": 6, "name": "CNN (512x384) Sem Pesos", "model_type": "cnn", "resolution": (512, 384), "use_weights": False},
    ]

    all_results = []
    for exp in experiments:
        res = run_experiment(exp, device=device)
        all_results.append(res)

    best_exp = max(all_results, key=lambda x: x["best_val_f1"])
    print(f"\nMelhor modelo (Val F1): {best_exp['name']} (F1={best_exp['best_val_f1']:.4f})")

    save_error_panel(best_exp)
    plot_resolution_effect(all_results)

    # tabela comparativa resumida
    rows = []
    for r in all_results:
        rows.append({
            "Experimento": r["id"],
            "Modelo": r["name"],
            "Resolução": r["resolution"],
            "Pesos": "Sim" if r["use_class_weights"] else "Não",
            "Parâmetros": r["parameters"],
            "Tamanho (MB)": round(r["model_size_mb"], 2),
            "Épocas": r["epochs_trained"],
            "Tempo Total (s)": round(r["total_time_seconds"], 1),
            "Tempo/Época (s)": round(r["avg_epoch_time_seconds"], 2),
            "Pico VRAM (MB)": round(r["peak_gpu_mem_mb"], 1),
            "F1 Validação": round(r["best_val_f1"], 4),
            "Acurácia Teste": round(r["test_accuracy"], 4),
            "F1 Macro Teste": round(r["test_f1_macro"], 4)
        })

    df = pd.DataFrame(rows)
    df.to_csv(PROJECT_ROOT / "results" / "comparative_table.csv", index=False)
    with open(PROJECT_ROOT / "results" / "comparative_summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print("\nTabela Comparativa:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
