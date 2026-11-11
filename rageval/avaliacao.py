"""Roda um recuperador sobre o conjunto de perguntas de referência (golden set) e agrega as métricas."""
import json
from dataclasses import dataclass, field
from pathlib import Path

from .corpus import Trecho
from .metricas import hit_at_k, mrr, ndcg_at_k, recall_at_k
from .retrievers import Recuperador
from .texto import normalizar


@dataclass(frozen=True)
class Pergunta:
    id: str
    pergunta: str
    relevantes: tuple[str, ...]
    fato: str  # trecho de texto que a resposta correta precisa conter


def carregar_golden(caminho: str | Path) -> list[Pergunta]:
    dados = json.loads(Path(caminho).read_text(encoding="utf-8"))
    return [Pergunta(d["id"], d["pergunta"], tuple(d["relevantes"]), d["fato"]) for d in dados]


def validar_golden(trechos: list[Trecho], perguntas: list[Pergunta]) -> list[str]:
    """Confere o próprio conjunto de referência: ids existem, e o fato esperado aparece nos relevantes."""
    por_id = {t.id: t for t in trechos}
    erros: list[str] = []
    vistos: set[str] = set()
    for p in perguntas:
        if p.id in vistos:
            erros.append(f"{p.id}: id repetido")
        vistos.add(p.id)
        if not p.relevantes:
            erros.append(f"{p.id}: sem trechos relevantes")
        for r in p.relevantes:
            if r not in por_id:
                erros.append(f"{p.id}: trecho inexistente {r}")
        existentes = [por_id[r] for r in p.relevantes if r in por_id]
        if existentes and not any(normalizar(p.fato) in normalizar(t.texto) for t in existentes):
            erros.append(f"{p.id}: o fato '{p.fato}' não aparece nos trechos relevantes")
    return erros


@dataclass
class ResultadoPergunta:
    pergunta: Pergunta
    ranking: list[str]
    recall: float
    hit: float
    mrr: float
    ndcg: float


@dataclass
class Resumo:
    nome: str
    k: int
    resultados: list[ResultadoPergunta] = field(default_factory=list)

    def _media(self, atributo: str) -> float:
        return sum(getattr(r, atributo) for r in self.resultados) / len(self.resultados)

    def metricas(self) -> dict[str, float]:
        return {
            f"recall@{self.k}": self._media("recall"),
            f"hit@{self.k}": self._media("hit"),
            "mrr": self._media("mrr"),
            f"ndcg@{self.k}": self._media("ndcg"),
        }

    def falhas(self) -> list[ResultadoPergunta]:
        """Perguntas em que nenhum trecho relevante apareceu entre os k primeiros."""
        return [r for r in self.resultados if r.hit == 0]


def avaliar(recuperador: Recuperador, perguntas: list[Pergunta], k: int = 3) -> Resumo:
    resumo = Resumo(recuperador.nome, k)
    for p in perguntas:
        ranking = [id_ for id_, _ in recuperador.buscar(p.pergunta)]
        rel = set(p.relevantes)
        resumo.resultados.append(
            ResultadoPergunta(p, ranking, recall_at_k(ranking, rel, k), hit_at_k(ranking, rel, k), mrr(ranking, rel), ndcg_at_k(ranking, rel, k))
        )
    return resumo
