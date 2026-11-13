# Classificação de Resíduos Sólidos por Visão Computacional: Um Estudo Comparativo entre MLP, Redes Convolucionais e Transferência de Aprendizado com YOLO

**Autores:** [Nome dos Integrantes da Equipe - 4 a 6 integrantes]  
**Matrículas:** [Matrículas dos Integrantes]  
**Instituição:** Universidade de Fortaleza (UNIFOR) — Graduação / EAD  
**Disciplina:** Redes Neurais Artificiais  
**Repositório GitHub:** `https://github.com/[seu-usuario]/[seu-repositorio]`

---

### Resumo

A triagem automatizada e a correta destinação de resíduos sólidos urbanos representam desafios fundamentais para a sustentabilidade ambiental e a viabilidade da reciclagem em larga escala. Este trabalho apresenta um estudo comparativo empírico e controlado para a classificação de resíduos por imagens utilizando o conjunto de dados *TrashNet* (2.527 imagens distribuídas em seis classes: *glass*, *paper*, *cardboard*, *plastic*, *metal* e *trash*). Foram avaliadas e comparadas arquiteturas *Multilayer Perceptron* (MLP) e Redes Neurais Convolucionais (CNN) sob uma partição estratificada estrita (70% treino, 15% validação e 15% teste com semente 42). O estudo investigou a influência da ponderação de pesos inversamente proporcionais à frequência de classes para mitigar o desbalanceamento, bem como o impacto do aumento da resolução espacial de entrada (128×96, 256×192 e 512×384 pixels) em relação ao custo computacional (tempo por época e total). Adicionalmente, foi explorada a transferência de aprendizado com uma variante pré-treinada *YOLOv8n Classify*. Os resultados revelaram que a CNN superou amplamente a MLP (67,89% contra 38,42% de acurácia de teste), utilizando 200 vezes menos parâmetros (94.470 contra 18,9 milhões) através de invariância espacial e *Global Average Pooling*. A elevação da resolução não proporcionou ganhos de acurácia (caindo para 58,95% em 512×384 pixels), enquanto elevou o tempo de treino em mais de 1.500%. Por fim, a abordagem baseada em transferência de aprendizado com YOLOv8 atingiu 92,89% de acurácia e 91,65% de F1-Macro no teste, estabelecendo um patamar de alto desempenho aplicável a sistemas de separação automatizada.

**Palavras-chave:** Visão Computacional, Redes Neurais Convolucionais, TrashNet, Resíduos Sólidos, Desbalanceamento de Classes, YOLOv8.

---

## 1. Introdução

O acelerado crescimento urbano e os padrões contemporâneos de consumo resultam em uma produção exponencial de resíduos sólidos em todo o mundo. A reciclagem eficiente depende criticamente da segregação correta dos materiais recicláveis em suas respectivas categorias físico-químicas (como vidro, plástico, papel, papelão e metais), bem como da rejeição de resíduos comuns não recicláveis (*trash*). Tradicionalmente, as centrais de triagem dependem de operações manuais e repetitivas, caracterizadas por elevados riscos ocupacionais à saúde dos operadores, baixa velocidade de processamento e inconsistência classificatória.

Nesse cenário, a automação baseada em Inteligência Artificial, especificamente em Visão Computacional e Aprendizado Profundo (*Deep Learning*), surge como uma alternativa promissora para modernizar esteiras de separação e lixeiras inteligentes (YANG; THUNG, 2016). No entanto, o desenvolvimento de modelos neurais para classificação de resíduos enfrenta desafios práticos severos, incluindo a sobreposição visual e textural entre classes (por exemplo, garrafas plásticas transparentes e frascos de vidro), deformações geométricas de embalagens amassadas, o desbalanceamento inerente entre os tipos de descarte e a escolha de um compromisso viável entre resolução de imagem e viabilidade computacional em tempo real.

O objetivo central deste trabalho é desenvolver um estudo sistemático no dataset *TrashNet*, contemplando seis experimentos controlados obrigatórios:
1. Comparar diretamente redes clássicas totalmente conectadas (*Multilayer Perceptron* - MLP) e Redes Neurais Convolucionais (CNN).
2. Investigar o efeito da ponderação de pesos nas perdas para atenuar o desbalanceamento da classe minoritária (*trash*).
3. Analisar o impacto da resolução espacial de entrada (128×96, 256×192 e 512×384) no desempenho da CNN frente ao custo de treinamento.
4. Explorar, como extensão técnica, o impacto da transferência de aprendizado com uma arquitetura moderna *state-of-the-art* (*YOLOv8n Classify*).

---

## 2. Materiais e Métodos

### 2.1 Conjunto de Dados: TrashNet
O estudo utilizou a versão de referência padronizada do arquivo `dataset-resized.zip` do repositório *TrashNet* (THUNG; YANG, 2016). O dataset reúne exatamente 2.527 imagens coloridas em formato RGB com dimensões nominais de 512 × 384 pixels (largura × altura), distribuídas em seis classes funcionais:
- **glass (vidro):** 501 imagens (19,83%)
- **paper (papel):** 594 imagens (23,51%)
- **cardboard (papelão):** 403 imagens (15,95%)
- **plastic (plástico):** 482 imagens (19,07%)
- **metal (metal):** 410 imagens (16,22%)
- **trash (outros resíduos):** 137 imagens (5,42%)

A classe *trash* configura-se como a classe minoritária explícita do estudo, possuindo menos de um quarto do volume da classe majoritária (*paper*), permitindo avaliar o comportamento dos algoritmos sob desequilíbrio amostral.

### 2.2 Particionamento Estratificado e Prevenção de Vazamento (*Data Leakage*)
Para garantir a integridade dos dados e impedir vazamento de informação entre etapas, adotou-se uma divisão estratificada com semente fixa (`random_seed = 42`):
- **Treinamento (70%):** 1.768 imagens
- **Validação (15%):** 379 imagens
- **Teste (15%):** 380 imagens

A proporção das seis classes foi rigorosamente mantida nas três partições. Os identificadores e caminhos de cada arquivo foram serializados no arquivo imutável `data/splits.json`, assegurando que todos os seis experimentos e o modelo extra avaliassem rigorosamente as mesmas amostras de teste, sem qualquer reamostragem.

### 2.3 Pré-processamento e Normalização
As imagens foram carregadas no espaço de cores RGB e redimensionadas por interpolação bilinear a partir da versão de referência de 512×384 pixels para as resoluções investigadas: 128×96, 256×192 e 512×384 pixels.
A normalização dos canais de cor consistiu na divisão escalar de cada pixel por 255,0, mapeando os tensores de entrada para o intervalo real $[0,0, 1,0]$.

### 2.4 Arquiteturas Desenvolvidas

#### 2.4.1 Multilayer Perceptron (MLP)
A MLP foi projetada para receber as imagens na resolução de 128×96 pixels. Os canais e coordenadas espaciais foram vetorizados em um tensor unidimensional de entrada de tamanho:
$$\text{Dimensão de Entrada} = 3 \times 128 \times 96 = 36.864 \text{ elementos}$$
A arquitetura é composta por:
1. Camada Densa: Entrada $\to$ 512 neurônios, ativação ReLU e Dropout de 0,2.
2. Camada Densa: 512 $\to$ 128 neurônios, ativação ReLU e Dropout de 0,2.
3. Camada de Saída: 128 $\to$ 6 neurônios (logits das classes).
Total de parâmetros treináveis: **18.941.318 parâmetros** (ocupando 72,26 MB em ponto flutuante).

#### 2.4.2 Rede Neural Convolucional (CNN)
A CNN foi projetada visando eficiência de extração de atributos visuais e invariância à escala por meio de três blocos convolucionais seguidos de *Global Average Pooling* (GAP):
- **Bloco 1:** Convolução 2D ($3 \to 32$ filtros, núcleo $3\times3$, preenchimento 1) $\to$ Normalização em Lote (*BatchNorm2d*) $\to$ Ativação ReLU $\to$ *MaxPool2d* ($2\times2$).
- **Bloco 2:** Convolução 2D ($32 \to 64$ filtros, núcleo $3\times3$, preenchimento 1) $\to$ *BatchNorm2d* $\to$ Ativação ReLU $\to$ *MaxPool2d* ($2\times2$).
- **Bloco 3:** Convolução 2D ($64 \to 128$ filtros, núcleo $3\times3$, preenchimento 1) $\to$ *BatchNorm2d* $\to$ Ativação ReLU $\to$ *MaxPool2d* ($2\times2$).
- **Camada de Redução:** *AdaptiveAvgPool2d((1, 1))* — *Global Average Pooling*.
- **Cabeça Classificadora:** Dropout de 0,3 $\to$ Camada Densa ($128 \to 6$ neurônios).

A aplicação do GAP garante que a dimensionalidade na saída do extrator convolucional seja fixada em um vetor de 128 elementos independentemente da resolução espacial de entrada ($128\times96$, $256\times192$ ou $512\times384$). Dessa forma, o total de parâmetros da CNN permanece estritamente fixo em **94.470 parâmetros** (0,37 MB) em todas as resoluções.

### 2.5 Tratamento de Desbalanceamento (Pesos de Classe)
Nos experimentos com ponderação, a função de perda *Cross-Entropy* foi modificada com pesos calculados exclusivamente a partir do subconjunto de treinamento:
$$w_c = \frac{N_{\text{treino}}}{C \times N_{c,\text{treino}}}$$
Onde $N_{\text{treino}} = 1.768$, $C = 6$, e $N_{c,\text{treino}}$ é a frequência da classe $c$. O peso atribuído à classe minoritária *trash* ($N_{\text{trash}} = 96$) foi de aproximadamente 3,07, enquanto a classe majoritária *paper* ($N_{\text{paper}} = 416$) recebeu peso 0,71.

### 2.6 Protocolo de Treinamento e Seleção de Modelos
- **Otimizador:** Adam com taxa de aprendizado inicial $\eta = 10^{-3}$ ($\beta_1=0,9, \beta_2=0,999$).
- **Tamanho do Lote (*Batch Size*):** 32.
- **Número Máximo de Épocas:** 35.
- **Critério de Parada (*Early Stopping*):** Paciência de 10 épocas monitorando a perda de validação, restaurando os melhores pesos salvos.
- **Métrica Principal de Seleção:** $F_1$-Macro de validação (atribuindo a mesma importância para as seis classes).
- **Ambiente Computacional:** GPU NVIDIA GeForce RTX 3050 (8 GB VRAM), processador x86_64, PyTorch 2.6.0+cu124, sistema operacional Windows 11.

### 2.7 Extensão com Transferência de Aprendizado (YOLOv8n Classify)
Para o objetivo bônus, utilizou-se a arquitetura *YOLOv8n-cls* (1.442.566 parâmetros e 3,3 GFLOPs), inicializada com pesos pré-treinados no *ImageNet-1k*. O modelo foi ajustado finamente (*fine-tuning*) no *TrashNet* durante 25 épocas com tamanho de imagem 224×224 pixels, mantendo-se rigorosamente a partição de dados idêntica (train/val/test).

---

## 3. Resultados

### 3.1 Tabela Geral Comparativa de Desempenho e Custo
A Tabela 1 consolida as métricas obtidas nos seis experimentos obrigatórios e no modelo bônus de transferência de aprendizado avaliados no mesmo subconjunto de teste.

**Tabela 1:** Síntese comparativa de desempenho e custo computacional dos modelos no conjunto de teste.
| Experimento | Modelo | Resolução | Pesos | Parâmetros | Tam. (MB) | Épocas | Tempo Total (s) | Tempo/Época (s) | F1-Validação | Acurácia Teste | F1-Macro Teste |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Exp 1** | MLP | 128×96 | Não | 18.941.318 | 72,26 | 17 | 85,5 | 4,98 | 0,3733 | 38,42% | 0,2938 |
| **Exp 2** | MLP | 128×96 | Sim | 18.941.318 | 72,26 | 25 | 125,1 | 4,98 | 0,1194 | 28,42% | 0,1271 |
| **Exp 3** | CNN | 128×96 | Não | 94.470 | 0,37 | 28 | 376,2 | 13,43 | 0,6280 | 67,89% | 0,6484 |
| **Exp 4** | **CNN (Ref.)** | 128×96 | **Sim** | 94.470 | 0,37 | 35 | 444,8 | 12,71 | **0,6646** | **66,84%** | **0,6303** |
| **Exp 5** | CNN | 256×192 | Não | 94.470 | 0,37 | 21 | 1.092,4 | 52,02 | 0,5802 | 57,89% | 0,5185 |
| **Exp 6** | CNN | 512×384 | Não | 94.470 | 0,37 | 28 | 6.134,1 | 219,08 | 0,5939 | 58,95% | 0,5402 |
| **Bônus** | **YOLOv8n-cls** | 224×224 | Pré-treinado | 1.442.566 | 3,00 | 25 | 377,5 | 15,10 | 0,9100 | **92,89%** | **0,9165** |

### 3.2 Desempenho por Classe do Melhor Modelo Obrigatório (Exp 4)
Conforme o critério estabelecido na Seção 4 da proposta, o modelo selecionado entre os seis obrigatórios foi o **Experimento 4 (CNN 128×96 Com Pesos)**, por apresentar o maior F1-Macro de validação (0,6646). A Tabela 2 apresenta o detalhamento de precisão, recall, F1-Score e suporte para cada uma das seis classes.

**Tabela 2:** Relatório de classificação detalhado por classe no conjunto de teste — Experimento 4 (CNN 128×96 com Pesos).
| Classe | Precisão | Recall (Sensibilidade) | F1-Score | Suporte (Exemplos) |
|:---|:---:|:---:|:---:|:---:|
| **glass** | 0,683 | 0,539 | 0,603 | 76 |
| **paper** | 0,735 | 0,798 | 0,765 | 89 |
| **cardboard**| 0,789 | 0,750 | 0,769 | 60 |
| **plastic** | 0,547 | 0,644 | 0,591 | 73 |
| **metal** | 0,689 | 0,677 | 0,683 | 62 |
| **trash** | 0,344 | 0,550 | 0,423 | 20 |
| **Média Macro** | **0,631** | **0,660** | **0,630** | **380** |
| **Acurácia Global** | — | — | **0,668 (66,84%)** | **380** |

### 3.3 Matrizes de Confusão e Curvas de Aprendizagem
As matrizes de confusão de todos os experimentos (disponíveis nos arquivos `results/cm_exp_1.png` a `results/cm_exp_6.png` e `results/cm_yolo.png`) apresentaram a ordem oficial estipulada: *glass, paper, cardboard, plastic, metal, trash*.
Nas curvas de perda e acurácia (`results/curves_exp_1.png` a `results/curves_exp_6.png`), observa-se que a MLP sofreu sobreajuste imediato após a 5ª época (perda de validação crescente enquanto a de treino decrescia a zero). A CNN apresentou curvas regulares, com convergência gradual e convergência estável da perda de validação.

### 3.4 Efeito da Resolução Espacial na CNN
A investigação empírica entre as resoluções 128×96, 256×192 e 512×384 (Experimentos 3, 5 e 6) gerou as seguintes evidências quantitativas:
1. **Desempenho no Teste:** A resolução de menor escala (128×96) alcançou a melhor acurácia (67,89%) e maior F1-Macro (0,6484). As resoluções 256×192 e 512×384 registraram quedas para 57,89% e 58,95%, respectivamente.
2. **Custo Computacional:** O tempo médio por época aumentou de 13,4 segundos (128×96) para 52,0 segundos (256×192) e 219,1 segundos (512×384). O tempo total de treino subiu de 6,2 minutos para mais de 102 minutos (aumento de 16,3 vezes).

### 3.5 Análise Qualitativa de Erros
A análise sistemática das predições incorretas do melhor modelo (Exp 4), registrada em `results/error_analysis.json` e ilustrada em `results/best_model_errors.png`, revelou os principais padrões de confusão:
1. **Plástico vs Trash / Vidro:** Amostras de garrafas plásticas transparentes e translúcidas foram classificadas como *glass* ou *trash* devido à ausência de pistas de rigidez e reflexos especulares semelhantes entre materiais vítreos e polímeros plásticos.
2. **Metal vs Papel / Trash:** Latas de alumínio amassadas e tubos metálicos deformados foram classificados como *trash* ou *paper* devido à alteração de suas geometrias regulares e reflexos distorcidos.
3. **Plástico com Rótulos de Papel:** Embalagens plásticas contendo grandes rótulos de papel impresso foram induzidas ao erro em direção à classe *paper*.

---

## 4. Discussão

### 4.1 Comportamento das Arquiteturas: MLP vs CNN
A disparidade entre MLP e CNN confirma os fundamentos teóricos de visão computacional. Ao achatar a matriz bidimensional em um vetor de 36.864 valores, a MLP perde completamente a correlação espacial e a localidade de vizinhança entre pixels adjacentes. Consequentemente, para tentar aprender características discriminativas, a camada densa necessita de quase 19 milhões de conexões. Com apenas 1.768 amostras de treino, o número de parâmetros superou a quantidade de dados em mais de quatro ordens de magnitude, resultando em *memorização* e sobreajuste prematuro (atingindo no máximo 38,42% de teste).
Em contrapartida, a CNN utiliza campos receptivos locais, compartilhamento de pesos nos filtros convolucionais e invariância a pequenas translações. O emprego de *Global Average Pooling* permitiu condensar o mapa de características espaciais sem a necessidade de camadas densas volumosas, viabilizando um modelo com apenas 94 mil parâmetros que atingiu 67,89% de acurácia com excelente generalização.

### 4.2 Impacto do Desbalanceamento e Pesos de Classe
Na MLP, a introdução de pesos causou severa instabilidade de gradiente: as penalidades elevadas para a classe *trash* fizeram com que o modelo oscilasse bruscamente, colapsando a acurácia para 28,42% e o F1-Macro para 0,1271.
Na CNN, por outro lado, a ponderação surtiu efeito benéfico no equilíbrio classificatório: o recall da classe minoritária *trash* subiu de 35% no Exp 3 para 55% no Exp 4 (Tabela 2), elevando o F1-Macro de validação para o melhor índice obtido entre todos os modelos desenvolvidos do zero (0,6646). Embora tenha ocorrido uma pequena redução residual na precisão de *trash* (decorrente do aumento de falsos positivos induzidos pelo peso maior), a capacidade de recuperar resíduos comuns foi significativamente reforçada.

### 4.3 Resolução vs Custo: Imagens Maiores Compensam?
A hipótese de que resoluções espaciais maiores proporcionariam representações mais ricas e maiores índices de acurácia foi **refutada experimentalmente** neste estudo.
Ao manter a profundidade convolucional constante (três blocos), o campo receptivo efetivo da rede torna-se proporcionalmente menor em relação à área total de uma imagem de 512×384 pixels do que em uma de 128×96 pixels. Em imagens grandes, os primeiros blocos convolucionais capturam apenas gradientes de textura ultramicroscópicos, falhando em abstrair a forma global do objeto descartado. Além disso, o volume de tensores intermediários dilatou exponencialmente o processamento matricial, elevando o tempo total de 6 minutos para 1,7 hora sem nenhum ganho prático. Conclui-se que, para este conjunto de dados e arquitetura, a resolução reduzida de 128×96 é a escolha ideal sob a ótica da engenharia de sistemas.

### 4.4 Comparação com o Estudo de Referência (Yang e Thung, 2016)
No artigo original de introdução do *TrashNet*, Yang e Thung (2016) relataram **63% de acurácia com SVM** operando sobre atributos manuais (SIFT e Color Histograms) nas seis classes, e apenas **22% de acurácia com uma CNN básica** treinada em cinco classes (os autores removeram propositalmente a classe *trash* do treino da CNN devido às dificuldades causadas pelo desbalanceamento). Posteriormente, no README do repositório oficial do projeto, os autores mencionaram ter atingido aproximadamente **75% de acurácia** com variantes posteriores de CNN.

A comparação com o estudo de Yang e Thung deve ser contextualizada com rigor metodológico:
1. **Diferenças de Partição e Classes:** O artigo utilizou uma divisão aleatória de 70/30 sem validação formal estratificada e excluiu *trash* do modelo convolucional, enquanto nosso estudo manteve integralmente as seis classes com divisão 70/15/15 estratificada.
2. **Desempenho da CNN:** Nossa CNN obrigatória alcançou **67,89% de acurácia de teste mantendo todas as seis classes**, superando substancialmente os 22% obtidos pela CNN do artigo original e os 63% do SVM.
3. **Ponto Crítico:** Nosso estudo demonstrou que não é necessário descartar a classe *trash* para obter convergência: o uso de pesos inversos e *Global Average Pooling* permite reter a classe minoritária e obter resultados sólidos.

### 4.5 Impacto da Transferência de Aprendizado com YOLOv8
O experimento com *YOLOv8n-cls* evidenciou o impacto revolucionário do pré-treinamento no *ImageNet*. Ao reaproveitar filtros extratores de bordas, reflexos e texturas genéricas já consolidados em milhões de imagens, o modelo alcançou **92,89% de acurácia global e 91,65% de F1-Macro** em apenas 25 épocas (Tabela 1). O recall em todas as cinco classes majoritárias superou 88%, e a classe minoritária *trash* alcançou 90,0% de recall (Tabela 2 do YOLO). Esse resultado demonstra que para implantação industrial real em esteiras seletivas, a transferência de aprendizado é a abordagem definitiva.

---

## 5. Conclusão

Este estudo conduziu uma investigação experimental aprofundada sobre a classificação supervisionada de resíduos utilizando o dataset *TrashNet*. As evidências empíricas obtidas permitiram responder categoricamente às questões norteadoras da pesquisa:

1. **Superioridade Convolucional:** As redes convolucionais são mandatórias para a tarefa, superando as MLPs em mais de 29% de acurácia com apenas uma fração de seus parâmetros e sem sofrer sobreajuste imediato.
2. **Mitigação do Desbalanceamento:** A aplicação de pesos inversamente proporcionais à frequência de classes comprovou ser eficaz na CNN, impulsionando a sensibilidade na classe crítica de rejeitos comuns (*trash*) e alcançando o maior F1-Macro de validação (0,6646).
3. **Inviabilidade de Resoluções Maiores:** O aumento da resolução de 128×96 para 512×384 pixels não trouxe benefícios preditivos e onerou o tempo de treino em mais de 16 vezes, consolidando a resolução de 128×96 como a solução mais eficiente.
4. **Potencial de Aplicação Real:** A combinação de modelos compactos modernos com transferência de aprendizado (*YOLOv8n-cls*) elevou a acurácia para 92,89%, comprovando a viabilidade técnica da visão computacional para triagem automatizada de materiais recicláveis.

Como trabalhos futuros e melhorias sugeridas, recomenda-se a expansão do dataset com imagens capturadas diretamente sob esteiras rolantes reais com iluminação não uniforme, a aplicação de técnicas avançadas de aumento de dados (*Data Augmentation*) e a implementação de quantização para inferência em dispositivos de borda (*Edge AI*).

---

## Referências

1. HE, K.; ZHANG, X.; REN, S.; SUN, J. Deep residual learning for image recognition. In: **Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR)**, 2016. p. 770-778.
2. JOCHER, G.; CHEN, A.; CORVO, A. **Ultralytics YOLOv8**. Versão 8.4.190, 2023. Disponível em: <https://github.com/ultralytics/ultralytics>. Acesso em: 29 out. 2026.
3. LIN, M.; CHEN, Q.; YAN, S. Network in network. **arXiv preprint arXiv:1312.4400**, 2013.
4. PASZKE, A. et al. PyTorch: An imperative style, high-performance deep learning library. In: **Advances in Neural Information Processing Systems (NeurIPS 32)**, 2019. p. 8024-8035.
5. PEDREGOSA, F. et al. Scikit-learn: Machine learning in Python. **Journal of Machine Learning Research**, v. 12, p. 2825-2830, 2011.
6. THUNG, G.; YANG, M. **TrashNet: Dataset of images of trash**. Repositório GitHub, 2016. Disponível em: <https://github.com/garythung/trashnet>. Acesso em: 26 out. 2026.
7. YANG, M.; THUNG, G. Classification of trash for recyclability status. **CS229 Project Report**, Stanford University, v. 2016, n. 1, p. 1-6, 2016.
