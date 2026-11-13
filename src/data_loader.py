import os
import json
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

# ordem exigida no documento da atividade
CLASS_NAMES = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASS_NAMES)}

DATASET_DEFAULT_PATH = r"C:\Users\ezequ\Downloads\dataset-resized\dataset-resized"
SPLITS_JSON_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "splits.json")


def create_or_load_splits(dataset_dir=DATASET_DEFAULT_PATH, splits_file=SPLITS_JSON_PATH, seed=42):
    # se ja tiver o json gerado, carrega direto pra nao mudar os conjuntos
    if os.path.exists(splits_file):
        with open(splits_file, "r", encoding="utf-8") as f:
            return json.load(f)

    samples = []
    for cls_name in CLASS_NAMES:
        cls_dir = os.path.join(dataset_dir, cls_name)
        if not os.path.isdir(cls_dir):
            raise FileNotFoundError(f"Pasta nao encontrada: {cls_dir}")
        
        files = sorted([
            f for f in os.listdir(cls_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        for f in files:
            samples.append({
                "path": os.path.join(cls_dir, f),
                "relative_path": os.path.join(cls_name, f),
                "class_name": cls_name,
                "label": CLASS_TO_IDX[cls_name]
            })

    labels = [s["label"] for s in samples]

    # split estratificado: 70% treino, 30% resto
    train_data, temp_data, _, temp_labels = train_test_split(
        samples,
        labels,
        test_size=0.30,
        random_state=seed,
        stratify=labels
    )

    # 15% validacao e 15% teste (metade dos 30%)
    val_data, test_data = train_test_split(
        temp_data,
        test_size=0.50,
        random_state=seed,
        stratify=temp_labels
    )

    splits = {
        "metadata": {
            "random_seed": seed,
            "total_samples": len(samples),
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "test_samples": len(test_data),
            "classes": CLASS_NAMES
        },
        "train": train_data,
        "val": val_data,
        "test": test_data
    }

    os.makedirs(os.path.dirname(splits_file), exist_ok=True)
    with open(splits_file, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2, ensure_ascii=False)

    return splits


def compute_class_weights(train_samples):
    # calcula pesos inversos apenas com o split de treino
    counts = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    for s in train_samples:
        counts[s["label"]] += 1.0

    total = len(train_samples)
    n_classes = len(CLASS_NAMES)
    w = total / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32)


class TrashNetDataset(Dataset):
    def __init__(self, samples, target_size=(128, 96), is_flatten=False):
        self.samples = samples
        self.target_size = target_size
        self.is_flatten = is_flatten

        width, height = self.target_size
        tensors = []
        labels = []
        paths = []

        # pre-carrega e redimensiona em uint8 pra acelerar o loop de treino
        for s in samples:
            with Image.open(s["path"]) as img:
                img_rgb = img.convert("RGB")
                if img_rgb.size != (width, height):
                    img_rgb = img_rgb.resize((width, height), Image.BILINEAR)
                arr = np.array(img_rgb, dtype=np.uint8)
                t = torch.from_numpy(arr).permute(2, 0, 1)
                tensors.append(t)

            labels.append(s["label"])
            paths.append(s["path"])

        self.tensors = torch.stack(tensors)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.paths = paths

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # normaliza para [0, 1] dividindo por 255
        x = self.tensors[idx].float() / 255.0

        if self.is_flatten:
            x = x.view(-1)

        y = self.labels[idx].item()
        path = self.paths[idx]
        return x, y, path


def get_dataloaders(target_size=(128, 96), is_mlp=False, batch_size=32, num_workers=0):
    splits = create_or_load_splits()

    train_ds = TrashNetDataset(splits["train"], target_size=target_size, is_flatten=is_mlp)
    val_ds = TrashNetDataset(splits["val"], target_size=target_size, is_flatten=is_mlp)
    test_ds = TrashNetDataset(splits["test"], target_size=target_size, is_flatten=is_mlp)

    weights = compute_class_weights(splits["train"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, weights
