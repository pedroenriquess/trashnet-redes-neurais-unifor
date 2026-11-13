import os
import sys
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def set_cell_border(cell, **kwargs):
    """
    Define bordas personalizadas para uma célula de tabela.
    kwargs: top, bottom, left, right (ex: val="single", sz="4", color="CCCCCC")
    """
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = parse_xml(r'<w:tcBorders xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"/>')
    for edge in ('top', 'left', 'bottom', 'right'):
        edge_data = kwargs.get(edge)
        if edge_data:
            tag = f'w:{edge}'
            element = parse_xml(f'<{tag} xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
                                f'w:val="{edge_data.get("val", "single")}" '
                                f'w:sz="{edge_data.get("sz", "4")}" '
                                f'w:space="0" '
                                f'w:color="{edge_data.get("color", "auto")}"/>')
            tcBorders.append(element)
    tcPr.append(tcBorders)

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def create_word_document():
    doc = Document()

    # 1. Configuração das Margens ABNT (Superior 3cm, Esquerda 3cm, Direita 2cm, Inferior 2cm)
    for section in doc.sections:
        section.top_margin = Cm(3.0)
        section.left_margin = Cm(3.0)
        section.right_margin = Cm(2.0)
        section.bottom_margin = Cm(2.0)
        section.page_width = Cm(21.0)
        section.page_height = Cm(29.7)

    # Configuração de Estilo Normal
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Arial'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(0, 0, 0)
    normal_style.paragraph_format.line_spacing = 1.35
    normal_style.paragraph_format.space_after = Pt(6)

    # --- CABEÇALHO / TÍTULO ---
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_after = Pt(14)
    run_title = p_title.add_run("CLASSIFICAÇÃO DE RESÍDUOS SÓLIDOS POR VISÃO COMPUTACIONAL: UM ESTUDO COMPARATIVO ENTRE MLP, REDES CONVOLUCIONAIS E TRANSFERÊNCIA DE APRENDIZADO COM YOLO")
    run_title.font.size = Pt(14)
    run_title.font.bold = True

    # Informações dos Autores (Editável)
    p_auth = doc.add_paragraph()
    p_auth.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_auth.paragraph_format.space_after = Pt(4)
    r1 = p_auth.add_run("Autores: [Nome dos Integrantes da Equipe — 4 a 6 integrantes]\n")
    r1.font.bold = True
    p_auth.add_run("Matrículas: [Matrícula 1, Matrícula 2, Matrícula 3, ...]\n")
    p_auth.add_run("Curso de Graduação — Universidade de Fortaleza (UNIFOR)\n")
    p_auth.add_run("Disciplina: Redes Neurais Artificiais\n")
    r_git = p_auth.add_run("Repositório no GitHub: https://github.com/[seu-usuario]/[seu-repositorio]\n")
    r_git.font.italic = True
    r_git.font.color.rgb = RGBColor(0, 102, 204)

    # --- RESUMO ---
    p_res = doc.add_paragraph()
    p_res.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_res.paragraph_format.space_before = Pt(10)
    p_res.paragraph_format.space_after = Pt(6)
    p_res.paragraph_format.left_indent = Cm(0)
    run_res_bold = p_res.add_run("RESUMO: ")
    run_res_bold.font.bold = True
    p_res.add_run(
        "A triagem automatizada e a correta destinação de resíduos sólidos urbanos representam desafios fundamentais "
        "para a sustentabilidade ambiental e a viabilidade da reciclagem em larga escala. Este trabalho apresenta um estudo "
        "comparativo empírico e controlado para a classificação de resíduos por imagens utilizando o conjunto de dados TrashNet "
        "(2.527 imagens distribuídas em seis classes: glass, paper, cardboard, plastic, metal e trash). Foram avaliadas e comparadas "
        "arquiteturas Multilayer Perceptron (MLP) e Redes Neurais Convolucionais (CNN) sob uma partição estratificada estrita "
        "(70% treino, 15% validação e 15% teste com semente 42). O estudo investigou a influência da ponderação de pesos inversamente "
        "proporcionais à frequência de classes para mitigar o desbalanceamento, bem como o impacto do aumento da resolução espacial "
        "de entrada (128×96, 256×192 e 512×384 pixels) em relação ao custo computacional (tempo por época e total). Adicionalmente, "
        "foi explorada a transferência de aprendizado com uma variante pré-treinada YOLOv8n Classify. Os resultados revelaram que "
        "a CNN superou amplamente a MLP (67,89% contra 38,42% de acurácia de teste), utilizando 200 vezes menos parâmetros "
        "(94.470 contra 18,9 milhões) através de invariância espacial e Global Average Pooling. A elevação da resolução não "
        "proporcionou ganhos de acurácia (caindo para 58,95% em 512×384 pixels), enquanto elevou o tempo de treino em mais de 1.500%. "
        "Por fim, a abordagem baseada em transferência de aprendizado com YOLOv8 atingiu 92,89% de acurácia e 91,65% de F1-Macro no teste, "
        "estabelecendo um patamar de alto desempenho aplicável a sistemas de separação automatizada."
    )

    p_kw = doc.add_paragraph()
    p_kw.paragraph_format.space_after = Pt(16)
    r_kw_bold = p_kw.add_run("Palavras-chave: ")
    r_kw_bold.font.bold = True
    p_kw.add_run("Visão Computacional. Redes Neurais Convolucionais. TrashNet. Resíduos Sólidos. YOLOv8.")

    def add_heading_1(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(4)
        run = p.add_run(text)
        run.font.bold = True
        run.font.size = Pt(12.5)

    def add_heading_2(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(3)
        run = p.add_run(text)
        run.font.bold = True
        run.font.size = Pt(11.5)

    def add_body(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.first_line_indent = Cm(1.25)
        p.paragraph_format.space_after = Pt(6)
        p.add_run(text)

    def add_figure(img_path, caption_text, width_in=5.8):
        if os.path.exists(img_path):
            p_img = doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_img.paragraph_format.space_before = Pt(8)
            p_img.paragraph_format.space_after = Pt(2)
            p_img.add_run().add_picture(str(img_path), width=Inches(width_in))

            p_cap = doc.add_paragraph()
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cap.paragraph_format.space_after = Pt(10)
            run_cap = p_cap.add_run(caption_text)
            run_cap.font.size = Pt(9.5)
            run_cap.font.italic = True

    # --- SEÇÃO 1: INTRODUÇÃO ---
    add_heading_1("1. INTRODUÇÃO")
    add_body(
        "O acelerado crescimento urbano e os padrões contemporâneos de consumo resultam em uma produção exponencial "
        "de resíduos sólidos em todo o mundo. A reciclagem eficiente depende criticamente da segregação correta dos materiais "
        "recicláveis em suas respectivas categorias físico-químicas (como vidro, plástico, papel, papelão e metais), bem como "
        "da rejeição de resíduos comuns não recicláveis (trash). Tradicionalmente, as centrais de triagem dependem de operações "
        "manuais e repetitivas, caracterizadas por elevados riscos ocupacionais à saúde dos operadores, baixa velocidade de "
        "processamento e inconsistência classificatória."
    )
    add_body(
        "Nesse cenário, a automação baseada em Inteligência Artificial, especificamente em Visão Computacional e Aprendizado "
        "Profundo (Deep Learning), surge como uma alternativa promissora para modernizar esteiras de separação e lixeiras "
        "inteligentes (YANG; THUNG, 2016). No entanto, o desenvolvimento de modelos neurais para classificação de resíduos "
        "enfrenta desafios práticos severos, incluindo a sobreposição visual e textural entre classes (por exemplo, garrafas "
        "plásticas transparentes e frascos de vidro), deformações geométricas de embalagens amassadas, o desbalanceamento inerente "
        "entre os tipos de descarte e a escolha de um compromisso viável entre resolução de imagem e viabilidade computacional em tempo real."
    )
    add_body(
        "O objetivo central deste trabalho é desenvolver um estudo sistemático no dataset TrashNet, contemplando seis experimentos "
        "controlados obrigatórios: (i) comparar diretamente redes clássicas totalmente conectadas (MLP) e Redes Neurais Convolucionais (CNN); "
        "(ii) investigar o efeito da ponderação de pesos nas perdas para atenuar o desbalanceamento da classe minoritária (trash); "
        "(iii) analisar o impacto da resolução espacial de entrada (128×96, 256×192 e 512×384) no desempenho da CNN frente ao custo de treinamento; "
        "e (iv) explorar, como extensão técnica, o impacto da transferência de aprendizado com uma arquitetura moderna state-of-the-art (YOLOv8n Classify)."
    )

    # --- SEÇÃO 2: MATERIAIS E MÉTODOS ---
    add_heading_1("2. MATERIAIS E MÉTODOS")
    add_heading_2("2.1 Conjunto de Dados: TrashNet")
    add_body(
        "O estudo utilizou a versão de referência padronizada do arquivo dataset-resized.zip do repositório TrashNet (THUNG; YANG, 2016). "
        "O dataset reúne exatamente 2.527 imagens coloridas em formato RGB com dimensões nominais de 512 × 384 pixels (largura × altura), "
        "distribuídas em seis classes funcionais: glass (501 imagens; 19,83%), paper (594 imagens; 23,51%), cardboard (403 imagens; 15,95%), "
        "plastic (482 imagens; 19,07%), metal (410 imagens; 16,22%) e trash (137 imagens; 5,42%). A classe trash configura-se como a classe "
        "minoritária explícita do estudo, possuindo menos de um quarto do volume da classe majoritária (paper), permitindo avaliar o comportamento "
        "dos algoritmos sob desequilíbrio amostral."
    )

    results_dir = PROJECT_ROOT / "results"
    add_figure(results_dir / "dataset_distribution.png", "Figura 1: Distribuição estratificada das classes de resíduos nas partições de Treino (70%), Validação (15%) e Teste (15%). Fonte: Autores (2026).")
    add_figure(results_dir / "class_samples.png", "Figura 2: Exemplos de imagens representativas das seis classes do dataset TrashNet. Fonte: Autores (2026).", width_in=4.8)

    add_heading_2("2.2 Particionamento Estratificado e Prevenção de Vazamento (Data Leakage)")
    add_body(
        "Para garantir a integridade dos dados e impedir vazamento de informação entre etapas, adotou-se uma divisão estratificada "
        "com semente fixa (random_seed = 42): 70% para treinamento (1.768 imagens), 15% para validação (379 imagens) e 15% para teste "
        "(380 imagens). A proporção das seis classes foi rigorosamente mantida nas três partições. Os identificadores e caminhos de cada "
        "arquivo foram serializados no arquivo imutável data/splits.json, assegurando que todos os seis experimentos e o modelo extra "
        "avaliassem rigorosamente as mesmas amostras de teste, sem qualquer reamostragem."
    )

    add_heading_2("2.3 Pré-processamento e Normalização")
    add_body(
        "As imagens foram carregadas no espaço de cores RGB e redimensionadas por interpolação bilinear a partir da versão de referência "
        "de 512×384 pixels para as resoluções investigadas: 128×96, 256×192 e 512×384 pixels (largura × altura). A normalização dos "
        "canais de cor consistiu na divisão escalar de cada pixel por 255,0, mapeando os tensores de entrada para o intervalo real [0,0, 1,0]."
    )
    add_figure(results_dir / "resolution_samples.png", "Figura 3: Comparativo de uma mesma imagem nas três resoluções avaliadas (128×96, 256×192 e 512×384 pixels). Fonte: Autores (2026).")

    add_heading_2("2.4 Arquiteturas Desenvolvidas: MLP e CNN")
    add_body(
        "A arquitetura MLP recebe as imagens em 128×96 pixels vetorizadas em um tensor unidimensional de 36.864 entradas "
        "(3 canais × 128 largura × 96 altura). Possui duas camadas ocultas totalmente conectadas (512 e 128 neurônios com ativação ReLU "
        "e Dropout de 0,2), totalizando 18.941.318 parâmetros treináveis (72,26 MB)."
    )
    add_body(
        "A arquitetura CNN é composta por três blocos convolucionais (filtros 3×3 com 32, 64 e 128 canais, seguidos de BatchNorm2d, "
        "ReLU e MaxPool2d 2×2), uma camada de Global Average Pooling (AdaptiveAvgPool2d((1, 1))) e uma cabeça classificadora densa "
        "(Dropout de 0,3 seguido de Linear 128→6). A inclusão do Global Average Pooling (GAP) garante que o vetor condensado antes "
        "da classificação tenha exatamente 128 atributos em qualquer resolução espacial, mantendo a contagem de parâmetros estritamente "
        "fixa em 94.470 parâmetros (0,37 MB) para 128×96, 256×192 e 512×384 pixels."
    )

    add_heading_2("2.5 Tratamento de Desbalanceamento e Protocolo de Treino")
    add_body(
        "Nos experimentos ponderados, a perda Cross-Entropy incorporou pesos inversamente proporcionais calculados exclusivamente "
        "no treino: w_c = N_treino / (C × N_c). O peso de trash foi de ~3,07 contra ~0,71 de paper. O treinamento utilizou o otimizador "
        "Adam (taxa de aprendizado 0,001, lote 32) com parada precoce (Early Stopping com paciência de 10 épocas e restauração dos melhores "
        "pesos pelo F1-Macro de validação). O hardware utilizado foi uma GPU NVIDIA GeForce RTX 3050 (8 GB VRAM)."
    )

    # --- SEÇÃO 3: RESULTADOS ---
    add_heading_1("3. RESULTADOS")
    add_body(
        "A Tabela 1 apresenta a síntese comparativa de desempenho e métricas de custo computacional de todos os seis experimentos "
        "obrigatórios e da extensão com YOLOv8n-cls avaliados no subconjunto de teste."
    )

    # Inserção da Tabela 1
    t1_data = [
        ["Exp", "Modelo", "Resolução", "Pesos", "Parâmetros", "Tam. (MB)", "Épocas", "Tempo Total", "Tempo/Época", "F1 Val.", "Acc Teste", "F1 Teste"],
        ["1", "MLP", "128×96", "Não", "18.941.318", "72,26", "17", "85,5 s", "4,98 s", "0,3733", "38,42%", "0,2938"],
        ["2", "MLP", "128×96", "Sim", "18.941.318", "72,26", "25", "125,1 s", "4,98 s", "0,1194", "28,42%", "0,1271"],
        ["3", "CNN", "128×96", "Não", "94.470", "0,37", "28", "376,2 s", "13,43 s", "0,6280", "67,89%", "0,6484"],
        ["4", "CNN (Melhor)", "128×96", "Sim", "94.470", "0,37", "35", "444,8 s", "12,71 s", "0,6646", "66,84%", "0,6303"],
        ["5", "CNN", "256×192", "Não", "94.470", "0,37", "21", "1.092,4 s", "52,02 s", "0,5802", "57,89%", "0,5185"],
        ["6", "CNN", "512×384", "Não", "94.470", "0,37", "28", "6.134,1 s", "219,08 s", "0,5939", "58,95%", "0,5402"],
        ["Bônus", "YOLOv8n-cls", "224×224", "Pré-trein.", "1.442.566", "3,00", "25", "377,5 s", "15,10 s", "0,9100", "92,89%", "0,9165"]
    ]

    p_t1_title = doc.add_paragraph()
    p_t1_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_t1 = p_t1_title.add_run("Tabela 1: Síntese comparativa de desempenho e custo computacional dos modelos no conjunto de teste.")
    r_t1.font.bold = True
    r_t1.font.size = Pt(9.5)

    table1 = doc.add_table(rows=len(t1_data), cols=len(t1_data[0]))
    table1.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(t1_data):
        for j, val in enumerate(row):
            cell = table1.cell(i, j)
            cell.text = val
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            run = p.runs[0]
            run.font.size = Pt(8.5)
            if i == 0:
                run.font.bold = True
                set_cell_background(cell, "EAEAEA")
            set_cell_border(cell, top={"sz": "4", "color": "BBBBBB"}, bottom={"sz": "4", "color": "BBBBBB"},
                            left={"sz": "4", "color": "BBBBBB"}, right={"sz": "4", "color": "BBBBBB"})

    add_body(
        "A Tabela 2 apresenta o detalhamento de precisão, recall, F1-Score e suporte para cada uma das seis classes "
        "no modelo obrigatório selecionado pelo maior F1-Macro de validação (Experimento 4 — CNN 128×96 com Pesos)."
    )

    # Inserção da Tabela 2
    t2_data = [
        ["Classe de Resíduo", "Precisão", "Recall (Sensibilidade)", "F1-Score", "Suporte (Exemplos de Teste)"],
        ["glass (vidro)", "0,683", "0,539", "0,603", "76"],
        ["paper (papel)", "0,735", "0,798", "0,765", "89"],
        ["cardboard (papelão)", "0,789", "0,750", "0,769", "60"],
        ["plastic (plástico)", "0,547", "0,644", "0,591", "73"],
        ["metal (metal)", "0,689", "0,677", "0,683", "62"],
        ["trash (outros resíduos)", "0,344", "0,550", "0,423", "20"],
        ["Média Macro (Macro Avg)", "0,631", "0,660", "0,630", "380"],
        ["Acurácia Global (Accuracy)", "—", "—", "0,668 (66,84%)", "380"]
    ]

    p_t2_title = doc.add_paragraph()
    p_t2_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_t2_title.paragraph_format.space_before = Pt(8)
    r_t2 = p_t2_title.add_run("Tabela 2: Relatório detalhado de métricas por classe no conjunto de teste — Experimento 4 (CNN 128×96 com Pesos).")
    r_t2.font.bold = True
    r_t2.font.size = Pt(9.5)

    table2 = doc.add_table(rows=len(t2_data), cols=len(t2_data[0]))
    table2.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(t2_data):
        for j, val in enumerate(row):
            cell = table2.cell(i, j)
            cell.text = val
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            run = p.runs[0]
            run.font.size = Pt(8.5)
            if i == 0 or i >= 7:
                run.font.bold = True
                set_cell_background(cell, "F0F0F0")
            set_cell_border(cell, top={"sz": "4", "color": "BBBBBB"}, bottom={"sz": "4", "color": "BBBBBB"},
                            left={"sz": "4", "color": "BBBBBB"}, right={"sz": "4", "color": "BBBBBB"})

    add_figure(results_dir / "curves_exp_4.png", "Figura 4: Curvas de perda e acurácia de treinamento e validação por época para a CNN 128×96 com Pesos (Exp 4). Fonte: Autores (2026).")
    add_figure(results_dir / "cm_exp_4.png", "Figura 5: Matriz de confusão no teste da CNN 128×96 com Pesos (contagens e porcentagens por linha). Fonte: Autores (2026).", width_in=4.8)
    add_figure(results_dir / "resolution_comparison.png", "Figura 6: Efeito da resolução espacial no desempenho e no tempo de treinamento da CNN (Exp 3, 5 e 6). Fonte: Autores (2026).")
    add_figure(results_dir / "best_model_errors.png", "Figura 7: Painel com amostras de teste classificadas incorretamente pelo modelo selecionado. Fonte: Autores (2026).")
    add_figure(results_dir / "cm_yolo.png", "Figura 8: Matriz de confusão no teste para a variante YOLOv8n-cls (Bônus). Fonte: Autores (2026).", width_in=4.8)

    # --- SEÇÃO 4: DISCUSSÃO ---
    add_heading_1("4. DISCUSSÃO")
    add_heading_2("4.1 Arquiteturas: MLP versus CNN")
    add_body(
        "A superioridade contundente da CNN sobre a MLP (67,89% contra 38,42%) decorre da preservação da topologia bidimensional "
        "da imagem. Ao vetorizar a entrada, a MLP perde as relações espaciais locais e necessita de quase 19 milhões de conexões ponderadas, "
        "o que ocasiona sobreajuste severo (overfitting) em poucas épocas, memorizando o treino sem generalizar. Em contrapartida, "
        "a CNN utiliza compartilhamento de pesos e Global Average Pooling (GAP), mantendo a invariância translacional com apenas "
        "94.470 parâmetros e apresentando convergência suave e estável."
    )

    add_heading_2("4.2 Desbalanceamento e Pesos de Classe")
    add_body(
        "A classe trash representa apenas 5,42% do dataset. Na CNN, a ponderação de perda elevou o recall da classe minoritária "
        "de 35% para 55%, fazendo com que o Experimento 4 alcançasse o maior F1-Macro de validação (0,6646). Em contrapartida, na MLP, "
        "a introdução de pesos desestabilizou o treinamento devido a gradientes desproporcionais, derrubando a acurácia para 28,42%."
    )

    add_heading_2("4.3 Efeito da Resolução: Imagens Maiores Compensam?")
    add_body(
        "A avaliação empírica nas resoluções 128×96, 256×192 e 512×384 pixels evidenciou que resoluções maiores não trouxeram "
        "nenhum ganho de desempenho preditivo (acurácia oscilando entre 57,89% e 58,95%), enquanto o tempo total de treinamento "
        "saltou de 6,2 minutos para mais de 102 minutos (aumento de mais de 16 vezes). Isso se explica pelo fato de que, em imagens "
        "muito grandes, um extrator de 3 blocos captura apenas texturas microscópicas locais sem integrar a geometria global do objeto, "
        "onerando desnecessariamente a computação. Conclui-se que a resolução de 128×96 é a escolha ideal para sistemas práticos de triagem."
    )

    add_heading_2("4.4 Comparação com o Estudo Seminal de Yang e Thung (2016)")
    add_body(
        "Yang e Thung (2016) reportaram 63% de acurácia com SVM em seis classes e 22% com uma CNN básica em cinco classes (após remover "
        "a classe trash devido ao desbalanceamento). Posteriormente, no README do repositório, mencionaram atingir cerca de 75% com "
        "experimentos adicionais. Nossa CNN atingiu 67,89% retendo integralmente todas as seis classes, comprovando que o descarte "
        "da classe minoritária é desnecessário quando se utilizam técnicas modernas de ponderação e Global Average Pooling."
    )

    add_heading_2("4.5 Extensão com Transferência de Aprendizado (YOLOv8n-cls)")
    add_body(
        "O ajuste fino do YOLOv8n-cls pré-treinado no ImageNet alcançou 92,89% de acurácia e 91,65% de F1-Macro em apenas 6 minutos "
        "de treinamento. O modelo alcançou sensibilidade superior a 88% em todas as classes principais e 90,0% na classe minoritária "
        "trash, demonstrando que a transferência de aprendizado é a solução definitiva para implementação industrial."
    )

    # --- SEÇÃO 5: CONCLUSÃO ---
    add_heading_1("5. CONCLUSÃO")
    add_body(
        "O estudo demonstrou experimentalmente que: (1) Redes Convolucionais com Global Average Pooling são mandatórias para classificação "
        "de resíduos, superando amplamente as MLPs em acurácia e eficiência de parâmetros; (2) O uso de pesos de classe inversos mitiga "
        "com sucesso o desbalanceamento da classe minoritária; (3) O aumento da resolução espacial de entrada não traz benefícios práticos "
        "e eleva o custo computacional em 16 vezes; e (4) A transferência de aprendizado com YOLOv8 viabiliza uma acurácia superior a 92%, "
        "estabelecendo uma solução pronta para operação em tempo real em centrais de triagem de reciclagem."
    )

    # --- SEÇÃO 6: REFERÊNCIAS ---
    add_heading_1("REFERÊNCIAS")
    refs = [
        "HE, K.; ZHANG, X.; REN, S.; SUN, J. Deep residual learning for image recognition. In: Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR), 2016. p. 770-778.",
        "JOCHER, G.; CHEN, A.; CORVO, A. Ultralytics YOLOv8. Versão 8.4.190, 2023. Disponível em: <https://github.com/ultralytics/ultralytics>. Acesso em: 29 out. 2026.",
        "LIN, M.; CHEN, Q.; YAN, S. Network in network. arXiv preprint arXiv:1312.4400, 2013.",
        "PASZKE, A. et al. PyTorch: An imperative style, high-performance deep learning library. In: Advances in Neural Information Processing Systems (NeurIPS 32), 2019. p. 8024-8035.",
        "PEDREGOSA, F. et al. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, v. 12, p. 2825-2830, 2011.",
        "THUNG, G.; YANG, M. TrashNet: Dataset of images of trash. Repositório GitHub, 2016. Disponível em: <https://github.com/garythung/trashnet>. Acesso em: 26 out. 2026.",
        "YANG, M.; THUNG, G. Classification of trash for recyclability status. CS229 Project Report, Stanford University, v. 2016, n. 1, p. 1-6, 2016."
    ]
    for r in refs:
        p_ref = doc.add_paragraph()
        p_ref.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p_ref.paragraph_format.space_after = Pt(4)
        run_ref = p_ref.add_run(r)
        run_ref.font.size = Pt(10)

    doc_path = PROJECT_ROOT / "docs" / "ARTIGO_TECNICO.docx"
    doc.save(str(doc_path))
    print(f"Sucesso! Documento Word gerado em: {doc_path} ({os.path.getsize(doc_path)/1024:.1f} KB)")

if __name__ == "__main__":
    create_word_document()
