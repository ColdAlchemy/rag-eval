"""Linha de comando: avaliar, falhas, perguntar, validar e relatorio."""
import argparse
import json
import sys
from pathlib import Path

from .avaliacao import Resumo, avaliar, carregar_golden, validar_golden
from .corpus import Trecho, carregar_corpus
from .relatorio import gerar_html
from .resposta import ClienteAnthropic, ClienteFake, checar_citacoes, contem_fato, gerar_resposta
from .retrievers import BM25, Aleatorio, ComExpansao, Hibrido, Recuperador, TfIdf

RAIZ = Path(__file__).resolve().parent.parent


def construir(trechos: list[Trecho], sinonimos: dict[str, list[str]]) -> list[Recuperador]:
    bm25, tfidf = BM25(trechos), TfIdf(trechos)
    bm25_exp = ComExpansao(bm25, sinonimos)
    return [Aleatorio(trechos), tfidf, bm25, bm25_exp, Hibrido([bm25_exp, tfidf])]


def _carregar(args) -> tuple[list[Trecho], list, list[Recuperador]]:
    trechos = carregar_corpus(args.corpus)
    perguntas = carregar_golden(args.golden)
    sinonimos = json.loads(Path(args.sinonimos).read_text(encoding="utf-8"))
    return trechos, perguntas, construir(trechos, sinonimos)


def _tabela(resumos: list[Resumo]) -> str:
    colunas = list(resumos[0].metricas().keys())
    larg = max(len(r.nome) for r in resumos)
    linhas = [f"{'recuperador'.ljust(larg)}  " + "  ".join(c.rjust(9) for c in colunas)]
    for r in resumos:
        m = r.metricas()
        linhas.append(f"{r.nome.ljust(larg)}  " + "  ".join(f"{m[c]:9.3f}" for c in colunas))
    return "\n".join(linhas)


def _escolher(recuperadores: list[Recuperador], nome: str) -> Recuperador:
    for r in recuperadores:
        if r.nome == nome:
            return r
    raise SystemExit(f"Recuperador desconhecido: {nome}. Opções: " + ", ".join(r.nome for r in recuperadores))


def main(argv: list[str] | None = None) -> int:
    for fluxo in (sys.stdout, sys.stderr):  # o console do Windows costuma usar cp1252
        if hasattr(fluxo, "reconfigure"):
            fluxo.reconfigure(encoding="utf-8")
    pai = argparse.ArgumentParser(add_help=False)
    pai.add_argument("--corpus", default=str(RAIZ / "data" / "corpus"))
    pai.add_argument("--golden", default=str(RAIZ / "data" / "golden.json"))
    pai.add_argument("--sinonimos", default=str(RAIZ / "data" / "sinonimos.json"))
    pai.add_argument("-k", type=int, default=3, help="quantos trechos entram na conta (padrão 3)")

    ap = argparse.ArgumentParser(prog="rageval", description="Avalia a recuperação de um sistema RAG.")
    sub = ap.add_subparsers(dest="comando", required=True)
    sub.add_parser("validar", parents=[pai], help="confere o corpus e o conjunto de referência")
    sub.add_parser("avaliar", parents=[pai], help="compara os recuperadores")
    f = sub.add_parser("falhas", parents=[pai], help="lista as perguntas que um recuperador errou")
    f.add_argument("--recuperador", default="bm25")
    p = sub.add_parser("perguntar", parents=[pai], help="busca, responde com citação e confere")
    p.add_argument("pergunta")
    p.add_argument("--recuperador", default="bm25+expansao")
    p.add_argument("--real", action="store_true", help="usa a API da Anthropic (exige ANTHROPIC_API_KEY)")
    r = sub.add_parser("relatorio", parents=[pai], help="gera o relatório HTML")
    r.add_argument("--saida", default=str(RAIZ / "docs" / "index.html"))
    args = ap.parse_args(argv)

    trechos, perguntas, recuperadores = _carregar(args)

    if args.comando == "validar":
        erros = validar_golden(trechos, perguntas)
        for e in erros:
            print("ERRO", e)
        print(f"{len(trechos)} trechos, {len(perguntas)} perguntas, {len(erros)} problema(s).")
        return 1 if erros else 0

    erros = validar_golden(trechos, perguntas)
    if erros:
        print("Conjunto de referência inválido:\n" + "\n".join(erros), file=sys.stderr)
        return 1

    if args.comando == "avaliar":
        print(_tabela([avaliar(rec, perguntas, args.k) for rec in recuperadores]))
    elif args.comando == "falhas":
        resumo = avaliar(_escolher(recuperadores, args.recuperador), perguntas, args.k)
        for x in resumo.falhas():
            esperado = ", ".join(x.pergunta.relevantes)
            print(f"{x.pergunta.id}  {x.pergunta.pergunta}\n    esperado: {esperado}\n    veio: {', '.join(x.ranking[:args.k]) or '(nada)'}")
        print(f"{len(resumo.falhas())} falha(s) em {len(perguntas)} perguntas ({resumo.nome}, k={args.k}).")
    elif args.comando == "perguntar":
        rec = _escolher(recuperadores, args.recuperador)
        por_id = {t.id: t for t in trechos}
        ids = [i for i, _ in rec.buscar(args.pergunta)][: args.k]
        usados = [por_id[i] for i in ids]
        cliente = ClienteAnthropic() if args.real else ClienteFake()
        resposta = gerar_resposta(cliente, args.pergunta, usados)
        print("Trechos:", ", ".join(ids) or "(nenhum)")
        print("Resposta:", resposta)
        print("Citações:", checar_citacoes(resposta, ids)["status"])
        conhecida = next((q for q in perguntas if q.pergunta == args.pergunta), None)
        if conhecida:
            print("Contém o fato esperado:", contem_fato(resposta, conhecida.fato))
    elif args.comando == "relatorio":
        resumos = [avaliar(rec, perguntas, args.k) for rec in recuperadores]
        destino = Path(args.saida)
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(gerar_html(resumos, trechos, args.k), encoding="utf-8")
        print("Relatório gravado em", destino)
    return 0
