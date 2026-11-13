# Classificação de Resíduos por Imagens (TrashNet)
### Atividade Final de Redes Neurais Artificiais — UNIFOR

Este repositório contém os códigos, experimentos e relatórios da atividade final de Redes Neurais Artificiais (UNIFOR), cujo objetivo é comparar arquiteturas MLP e CNN, avaliar o impacto do desbalanceamento entre classes com pesos inversos e investigar se resoluções espaciais maiores compensam o custo computacional no dataset TrashNet. Também inclui a extensão com YOLOv8 no modo classify.

---

##  Resumo dos Resultados

Os experimentos foram treinados e avaliados mantendo a mesma partição estratificada (**70% treino, 15% validação e 15% teste**, semente `seed=42`) em uma GPU NVIDIA GeForce RTX 3050:

| Exp | Modelo | Resolução | Pesos | Parâmetros | Épocas | Tempo Total | F1 Val. | Acc Teste | F1 Teste |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | MLP Sem Pesos | 128×96 | Não | 18.941.318 | 17 | 85,5 s | 0,3733 | **38,42%** | 0,2938 |
| **2** | MLP Com Pesos | 128×96 | Sim | 18.941.318 | 25 | 125,1 s | 0,1194 | **28,42%** | 0,1271 |
| **3** | CNN Sem Pesos | 128×96 | Não | 94.470 | 28 | 376,2 s | 0,6280 | **67,89%** | **0,6484** |
| **4** | CNN Com Pesos *(Melhor Val.)* | 128×96 | Sim | 94.470 | 35 | 444,8 s | **0,6646** | **66,84%** | **0,6303** |
| **5** | CNN Sem Pesos | 256×192 | Não | 94.470 | 21 | 1.092,4 s | 0,5802 | **57,89%** | 0,5185 |
| **6** | CNN Sem Pesos | 512×384 | Não | 94.470 | 28 | 6.134,1 s | 0,5939 | **58,95%** | 0,5402 |
| **Bônus** | **YOLOv8n-cls** | 224×224 | Pré-treinado | 1.442.566 | 25 | 377,5 s | 0,9100 | **92,89%** | **0,9165** |

---

##  Estrutura do Projeto

```text
├── data/
│   ├── splits.json               # divisao oficial (70/15/15) salva em json
│   └── yolo_dataset/             # estrutura em pastas train/val/test para o YOLO
├── docs/
│   ├── ARTIGO_TECNICO.docx       # artigo formatado em Word para edicao da equipe
│   └── ARTIGO_TECNICO.pdf        # versao final em PDF para submissao no AVA
├── models/                       # checkpoints salvos (.pt)
│   ├── exp_1_mlp.pt
│   ├── ...
│   ├── exp_6_cnn.pt
│   └── yolo_run/                 # pesos salvos do YOLOv8n-cls
├── notebooks/
│   ├── 01_analise_exploratoria.ipynb  # inspecao dos dados, distribuicao e amostras
│   └── 03_resultados.ipynb            # visualizacao de metricas, curvas e matrizes
├── results/                      # figuras geradas, matrizes de confusao e relatorios json/csv
├── scripts/
│   ├── run_all_experiments.py    # executa os 6 experimentos obrigatorios
│   ├── run_yolo_bonus.py         # executa o treino e teste do YOLOv8
│   └── generate_pdf_article.py   # exporta o PDF do artigo
├── requirements.txt
└── README.md
```

---

## ️ Como Executar

### 1. Dependências
```bash
pip install -r requirements.txt
```

### 2. Avaliação sem novo treino
Os modelos já treinados estão salvos na pasta `models/`. É possível visualizar todas as métricas, curvas de perda e matrizes de confusão abrindo o notebook `notebooks/03_resultados.ipynb`.

### 3. Reexecutar os experimentos
```bash
# Executa os 6 treinos em sequencia (MLP e CNNs)
python scripts/run_all_experiments.py

# Treina e avalia o modelo YOLOv8 classify
python scripts/run_yolo_bonus.py
```
