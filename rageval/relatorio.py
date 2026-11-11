"""Relatório HTML estático e autocontido (sem JavaScript, sem recursos externos)."""
from html import escape

from .avaliacao import Resumo
from .corpus import Trecho

CSS = """
:root{--fundo:#fbfaf7;--tinta:#1d2433;--suave:#5b6578;--linha:#dcd8cd;--acento:#1f4fb4;--ok:#1d7a46;--erro:#b3261e;--barra:#cfd8ee}
@media (prefers-color-scheme:dark){:root{--fundo:#12151c;--tinta:#e8e6df;--suave:#a3aab8;--linha:#2c3342;--acento:#8fb0ff;--ok:#5fcf8f;--erro:#ff8a80;--barra:#27345a}}
*{box-sizing:border-box}
body{margin:0;background:var(--fundo);color:var(--tinta);font:16px/1.55 Georgia,"Times New Roman",serif}
main{max-width:62rem;margin:0 auto;padding:2rem 1rem 4rem}
h1{font-size:1.9rem;margin:0 0 .25rem}h2{font-size:1.25rem;margin:2.5rem 0 .75rem;border-bottom:1px solid var(--linha);padding-bottom:.35rem}
p.sub{color:var(--suave);margin:0 0 1.5rem}
table{width:100%;border-collapse:collapse;font:14px/1.4 ui-monospace,Consolas,monospace}
th,td{padding:.45rem .5rem;border-bottom:1px solid var(--linha);text-align:left;vertical-align:top}
th{color:var(--suave);font-weight:600}
td.n{text-align:right;white-space:nowrap}
.barra{display:inline-block;height:.6rem;background:var(--barra);border-radius:2px;vertical-align:middle;margin-right:.4rem}
.barra i{display:block;height:100%;background:var(--acento);border-radius:2px}
.ok{color:var(--ok)}.erro{color:var(--erro)}
.rolagem{overflow-x:auto}
footer{margin-top:3rem;color:var(--suave);font-size:.9rem}
a{color:var(--acento)}
"""


def _barra(valor: float) -> str:
    return f'<span class="barra" style="width:5rem" aria-hidden="true"><i style="width:{valor * 100:.0f}%"></i></span>'


def gerar_html(resumos: list[Resumo], trechos: list[Trecho], k: int) -> str:
    colunas = list(resumos[0].metricas().keys())
    n = len(resumos[0].resultados)
    cab = "".join(f"<th>{escape(c)}</th>" for c in colunas)
    linhas = []
    for r in resumos:
        m = r.metricas()
        cel = "".join(f'<td class="n">{_barra(m[c])}{m[c]:.3f}</td>' for c in colunas)
        linhas.append(f"<tr><td>{escape(r.nome)}</td>{cel}</tr>")

    cab_q = "".join(f"<th>{escape(r.nome)}</th>" for r in resumos)
    matriz = []
    for i, base in enumerate(resumos[0].resultados):
        marcas = []
        for r in resumos:
            acertou = r.resultados[i].hit == 1.0
            marcas.append(f'<td class="{"ok" if acertou else "erro"}">{"acertou" if acertou else "errou"}</td>')
        matriz.append(f"<tr><td>{escape(base.pergunta.id)}</td><td>{escape(base.pergunta.pergunta)}</td>{''.join(marcas)}</tr>")

    return f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>rag-eval: relatório de recuperação</title>
<style>{CSS}</style>
</head>
<body>
<main>
<h1>rag-eval</h1>
<p class="sub">{n} perguntas de referência · {len(trechos)} trechos · k = {k}. Corpus e perguntas fictícios.</p>

<h2>Comparação dos recuperadores</h2>
<div class="rolagem"><table>
<thead><tr><th>recuperador</th>{cab}</tr></thead>
<tbody>{''.join(linhas)}</tbody>
</table></div>

<h2>Acerto por pergunta (algum trecho relevante entre os {k} primeiros)</h2>
<div class="rolagem"><table>
<thead><tr><th>id</th><th>pergunta</th>{cab_q}</tr></thead>
<tbody>{''.join(matriz)}</tbody>
</table></div>

<footer>Gerado por <code>python -m rageval relatorio</code>. Os recuperadores são implementações próprias, sem bibliotecas de busca.
A lista de sinônimos da expansão foi escrita depois de ver as falhas do conjunto de referência, então o ganho dela é otimista.</footer>
</main>
</body>
</html>
"""
