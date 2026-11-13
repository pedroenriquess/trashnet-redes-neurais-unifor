import os
import sys
import subprocess
from pathlib import Path
import markdown

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def build_pdf():
    md_path = PROJECT_ROOT / "docs" / "ARTIGO_TECNICO.md"
    html_path = PROJECT_ROOT / "docs" / "ARTIGO_TECNICO.html"
    pdf_path = PROJECT_ROOT / "docs" / "ARTIGO_TECNICO.pdf"

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Converter Markdown para HTML
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "nl2br"]
    )

    # Injetar figuras diretamente nas seções de Resultados para atender à exigência do artigo ABNT
    results_dir = (PROJECT_ROOT / "results").resolve()
    
    # Preparar tags de imagem com caminhos absolutos file:///
    fig_dist = f'<div class="figure-box"><img src="{results_dir / "dataset_distribution.png"}" /><p class="fig-caption"><strong>Figura 1:</strong> Distribuição estratificada das classes de resíduos nas partições de Treino (70%), Validação (15%) e Teste (15%). Fonte: Autores (2026).</p></div>'
    fig_samples = f'<div class="figure-box"><img src="{results_dir / "class_samples.png"}" /><p class="fig-caption"><strong>Figura 2:</strong> Exemplos de imagens das seis classes do dataset TrashNet. Fonte: Autores (2026).</p></div>'
    fig_res_samples = f'<div class="figure-box"><img src="{results_dir / "resolution_samples.png"}" /><p class="fig-caption"><strong>Figura 3:</strong> Comparativo de uma mesma imagem nas três resoluções avaliadas (128×96, 256×192 e 512×384 pixels). Fonte: Autores (2026).</p></div>'
    fig_curves = f'<div class="figure-box"><img src="{results_dir / "curves_exp_4.png"}" /><p class="fig-caption"><strong>Figura 4:</strong> Curvas de perda (Loss) e acurácia por época no treino e validação — Modelo CNN 128×96 com Pesos (Exp 4). Fonte: Autores (2026).</p></div>'
    fig_cm = f'<div class="figure-box"><div class="side-by-side"><img src="{results_dir / "cm_exp_4.png"}" style="width:48%;" /><img src="{results_dir / "cm_yolo.png"}" style="width:48%;" /></div><p class="fig-caption"><strong>Figura 5:</strong> Matrizes de confusão no conjunto de teste: CNN 128×96 com Pesos (esquerda) e YOLOv8n-cls (direita). Fonte: Autores (2026).</p></div>'
    fig_res_comp = f'<div class="figure-box"><img src="{results_dir / "resolution_comparison.png"}" /><p class="fig-caption"><strong>Figura 6:</strong> Impacto da resolução espacial na acurácia/F1-macro e no tempo computacional da CNN. Fonte: Autores (2026).</p></div>'
    fig_errors = f'<div class="figure-box"><img src="{results_dir / "best_model_errors.png"}" /><p class="fig-caption"><strong>Figura 7:</strong> Painel de análise de erros do modelo CNN 128×96 no conjunto de teste. Fonte: Autores (2026).</p></div>'

    # Inserção das figuras no corpo HTML nos pontos correspondentes
    html_body = html_body.replace("### 3.3 Matrizes de Confusão e Curvas de Aprendizagem", fig_curves + fig_cm + "### 3.3 Matrizes de Confusão e Curvas de Aprendizagem")
    html_body = html_body.replace("### 3.4 Efeito da Resolução Espacial na CNN", fig_res_comp + "### 3.4 Efeito da Resolução Espacial na CNN")
    html_body = html_body.replace("### 3.5 Análise Qualitativa de Erros", fig_errors + "### 3.5 Análise Qualitativa de Erros")
    html_body = html_body.replace("### 2.2 Particionamento Estratificado", fig_dist + "### 2.2 Particionamento Estratificado")
    html_body = html_body.replace("### 2.3 Pré-processamento e Normalização", fig_samples + fig_res_samples + "### 2.3 Pré-processamento e Normalização")

    html_template = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<title>Artigo Técnico - UNIFOR</title>
<style>
    @page {{
        size: A4 portrait;
        margin: 25mm 20mm 20mm 25mm;
        @bottom-right {{
            content: counter(page);
        }}
    }}
    body {{
        font-family: "Arial", "Helvetica", sans-serif;
        font-size: 11pt;
        line-height: 1.45;
        color: #111;
        text-align: justify;
        margin: 0;
        padding: 0;
    }}
    h1 {{
        font-size: 16pt;
        text-align: center;
        text-transform: uppercase;
        margin-bottom: 18px;
        line-height: 1.3;
    }}
    h2 {{
        font-size: 13pt;
        margin-top: 22px;
        margin-bottom: 8px;
        border-bottom: 1px solid #ccc;
        padding-bottom: 3px;
        text-transform: uppercase;
    }}
    h3 {{
        font-size: 11.5pt;
        margin-top: 14px;
        margin-bottom: 6px;
    }}
    p {{
        margin-top: 0;
        margin-bottom: 8px;
        text-indent: 1.25cm;
    }}
    blockquote, .abstract-box {{
        background: #fdfdfd;
        border-left: 3px solid #0056b3;
        padding: 8px 14px;
        margin: 14px 0;
        font-size: 10pt;
        text-indent: 0;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 14px 0;
        font-size: 9.5pt;
    }}
    th, td {{
        border: 1px solid #bbb;
        padding: 5px 8px;
        text-align: center;
    }}
    th {{
        background-color: #f2f2f2;
        font-weight: bold;
    }}
    .figure-box {{
        text-align: center;
        margin: 14px 0;
        page-break-inside: avoid;
    }}
    .figure-box img {{
        max-width: 95%;
        height: auto;
        border: 1px solid #e0e0e0;
        border-radius: 3px;
    }}
    .side-by-side {{
        display: flex;
        justify-content: space-between;
    }}
    .fig-caption {{
        font-size: 9pt;
        color: #444;
        margin-top: 5px;
        text-indent: 0 !important;
        text-align: center;
    }}
    ul, ol {{
        margin-top: 4px;
        margin-bottom: 8px;
        padding-left: 28px;
    }}
    li {{
        margin-bottom: 4px;
    }}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_template)
    print(f"HTML gerado em: {html_path}")

    # Renderizar para PDF via Microsoft Edge Headless
    edge_exe = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    if not os.path.exists(edge_exe):
        edge_exe = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

    cmd = [
        edge_exe,
        "--headless=new",
        "--disable-gpu",
        f"--print-to-pdf={pdf_path}",
        "--no-pdf-header-footer",
        str(html_path)
    ]
    print("Exportando PDF com o navegador em modo headless...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if os.path.exists(pdf_path):
        size_kb = os.path.getsize(pdf_path) / 1024
        print(f"Sucesso! PDF do Artigo gerado em: {pdf_path} ({size_kb:.1f} KB)")
    else:
        print("Erro ao gerar PDF:", res.stderr)

if __name__ == "__main__":
    build_pdf()
