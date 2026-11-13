import os
import sys
import json
import shutil
import time
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import CLASS_NAMES, SPLITS_JSON_PATH
from src.evaluate import plot_confusion_matrix


def prepare_dataset_folders(splits_file, out_dir):
    # organiza train, val e test em pastas separadas respeitando os splits
    with open(splits_file, "r", encoding="utf-8") as f:
        splits = json.load(f)

    for split in ["train", "val", "test"]:
        for item in splits[split]:
            src = Path(item["path"])
            dst_dir = out_dir / split / item["class_name"]
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_file = dst_dir / src.name
            if not dst_file.exists():
                shutil.copy2(src, dst_file)


def main():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Ultralytics nao instalado. Execute: pip install ultralytics")
        return

    splits_file = PROJECT_ROOT / "data" / "splits.json"
    yolo_dir = PROJECT_ROOT / "data" / "yolo_dataset"
    prepare_dataset_folders(splits_file, yolo_dir)

    print("\nTreinando YOLOv8n-cls com pesos pre-treinados do ImageNet...")
    model = YOLO("yolov8n-cls.pt")

    t0 = time.perf_counter()
    epochs = 25
    imgsz = 224

    # fine-tuning
    model.train(
        data=str(yolo_dir),
        epochs=epochs,
        imgsz=imgsz,
        batch=32,
        workers=0,
        project=str(PROJECT_ROOT / "models" / "yolo_run"),
        name="trashnet_yolov8n",
        exist_ok=True,
        seed=42
    )

    total_time = time.perf_counter() - t0
    avg_epoch = total_time / epochs

    # avalia no subconjunto de teste
    best_pt = PROJECT_ROOT / "models" / "yolo_run" / "trashnet_yolov8n" / "weights" / "best.pt"
    eval_model = YOLO(str(best_pt))

    test_dir = yolo_dir / "test"
    targets = []
    preds = []
    probs = []

    for idx, cls_name in enumerate(CLASS_NAMES):
        cls_path = test_dir / cls_name
        if not cls_path.exists():
            continue
        for img_path in cls_path.glob("*.*"):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                res = eval_model(str(img_path), verbose=False)[0]
                top1_idx = res.probs.top1
                pred_name = res.names[top1_idx]
                
                # garante que o indice bate com a ordem oficial
                pred_idx = CLASS_NAMES.index(pred_name) if pred_name in CLASS_NAMES else top1_idx
                targets.append(idx)
                preds.append(pred_idx)
                probs.append(float(res.probs.top1conf.cpu().numpy()))

    acc = float(accuracy_score(targets, preds))
    f1_macro = float(f1_score(targets, preds, average="macro", zero_division=0))
    report = classification_report(
        targets,
        preds,
        target_names=CLASS_NAMES,
        labels=list(range(len(CLASS_NAMES))),
        output_dict=True,
        zero_division=0
    )
    cm = confusion_matrix(targets, preds, labels=list(range(len(CLASS_NAMES))))

    plot_confusion_matrix(
        cm,
        title="Matriz de Confusao - YOLOv8n-cls",
        output_path=str(PROJECT_ROOT / "results" / "cm_yolo.png"),
        class_names=CLASS_NAMES
    )

    summary = {
        "model": "YOLOv8n-cls",
        "pretraining": "ImageNet-1k",
        "resolution": f"{imgsz}x{imgsz}",
        "epochs": epochs,
        "total_time_seconds": round(total_time, 1),
        "avg_epoch_time_seconds": round(avg_epoch, 2),
        "test_accuracy": round(acc, 4),
        "test_f1_macro": round(f1_macro, 4),
        "test_report": report,
        "confusion_matrix": cm.tolist()
    }

    with open(PROJECT_ROOT / "results" / "yolo_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nYOLO -> Acurácia Teste: {acc*100:.2f}% | F1-Macro: {f1_macro*100:.2f}%")
    print(f"Tempo total: {total_time:.1f}s")


if __name__ == "__main__":
    main()
