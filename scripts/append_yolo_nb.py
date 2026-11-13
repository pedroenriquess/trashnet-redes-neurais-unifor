import json
from pathlib import Path

nb_path = Path("notebooks/03_resultados.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

yolo_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Extensão Opcional: Transferência de Aprendizado com YOLOv8 Classify (Bônus)\n",
            "Comparação direta no mesmo conjunto de teste entre a melhor CNN desenvolvida do zero e o modelo pré-treinado no ImageNet (YOLOv8n-cls)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "yolo_json = RESULTS_DIR / 'yolo_summary.json'\n",
            "if yolo_json.exists():\n",
            "    with open(yolo_json, 'r', encoding='utf-8') as f:\n",
            "        yd = json.load(f)\n",
            "    print(f\"Modelo: {yd['model']} | Pré-treinamento: {yd['pretraining']}\")\n",
            "    print(f\"Acurácia no Teste: {yd['test_accuracy']*100:.2f}% | F1-Macro no Teste: {yd['test_f1_macro']*100:.2f}%\")\n",
            "    print(f\"Tempo de Treinamento: {yd['total_time_seconds']}s ({yd['avg_epoch_time_seconds']}s/época)\")\n",
            "    rep_df = pd.DataFrame(yd['test_report']).transpose()\n",
            "    display(rep_df.loc[['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash', 'macro avg', 'accuracy']])\n",
            "else:\n",
            "    print('yolo_summary.json não encontrado.')"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))\n",
            "cm_yolo_path = RESULTS_DIR / 'cm_yolo.png'\n",
            "curves_yolo_path = RESULTS_DIR / 'curves_yolo.png'\n",
            "if cm_yolo_path.exists():\n",
            "    ax1.imshow(mpimg.imread(str(cm_yolo_path)))\n",
            "    ax1.axis('off')\n",
            "    ax1.set_title('Matriz de Confusão (YOLOv8n-cls)', fontsize=12, fontweight='bold')\n",
            "if curves_yolo_path.exists():\n",
            "    ax2.imshow(mpimg.imread(str(curves_yolo_path)))\n",
            "    ax2.axis('off')\n",
            "    ax2.set_title('Curvas de Treino e Validação (Ultralytics)', fontsize=12, fontweight='bold')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    }
]

nb["cells"].extend(yolo_cells)
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("YOLO cells appended to 03_resultados.ipynb!")
