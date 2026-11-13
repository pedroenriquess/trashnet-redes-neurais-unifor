import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import CLASS_NAMES, create_or_load_splits

def main():
    splits = create_or_load_splits()
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. grafico de distribuicao
    train_df = pd.DataFrame(splits['train'])
    val_df = pd.DataFrame(splits['val'])
    test_df = pd.DataFrame(splits['test'])

    df_splits = pd.DataFrame({
        'Treino': train_df['class_name'].value_counts(),
        'Validação': val_df['class_name'].value_counts(),
        'Teste': test_df['class_name'].value_counts()
    }).loc[CLASS_NAMES]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    df_splits.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c'], width=0.8)
    ax.set_title('Distribuição das Classes (TrashNet)', fontsize=12)
    ax.set_ylabel('Quantidade de Imagens')
    ax.set_xlabel('Classe')
    ax.set_xticklabels(CLASS_NAMES, rotation=0)
    ax.grid(True, linestyle='--', alpha=0.4, axis='y')
    plt.tight_layout()
    plt.savefig(results_dir / "dataset_distribution.png", dpi=300)
    plt.close()

    # 2. exemplos por classe (2 de cada)
    fig, axes = plt.subplots(len(CLASS_NAMES), 2, figsize=(6, 2.4 * len(CLASS_NAMES)))
    for i, cls in enumerate(CLASS_NAMES):
        samples = [s['path'] for s in splits['train'] if s['class_name'] == cls][:2]
        for j, p in enumerate(samples):
            im = Image.open(p).convert('RGB')
            axes[i, j].imshow(im)
            axes[i, j].set_title(f"{cls} ({j+1})", fontsize=9)
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.savefig(results_dir / "class_samples.png", dpi=300)
    plt.close()

    # 3. comparativo de resolucoes de uma mesma imagem
    orig = Image.open(splits['train'][0]['path']).convert('RGB')
    res_list = [(128, 96), (256, 192), (512, 384)]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (w, h) in zip(axes, res_list):
        resized = orig.resize((w, h))
        ax.imshow(resized)
        ax.set_title(f"{w} × {h}", fontsize=11)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(results_dir / "resolution_samples.png", dpi=300)
    plt.close()

    print("Figuras de EDA geradas com sucesso.")

if __name__ == "__main__":
    main()
