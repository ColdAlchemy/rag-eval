"""Recuperadores implementados do zero: BM25, TF-IDF (cosseno), híbrido (RRF), expansão e aleatório.

Todos devolvem o ranking como lista de (id_do_trecho, pontuação), do melhor para o pior.
Trechos sem nenhum termo em comum com a consulta (pontuação 0) não entram no ranking.
"""
import math
import random
from collections import Counter
from typing import Protocol

from .corpus import Trecho
from .texto import normalizar, tokens


class Recuperador(Protocol):
    nome: str

    def buscar(self, consulta: str) -> list[tuple[str, float]]: ...


def _ordenar(pontos: dict[str, float]) -> list[tuple[str, float]]:
    positivos = [(i, p) for i, p in pontos.items() if p > 0]
    return sorted(positivos, key=lambda x: (-x[1], x[0]))


class BM25:
    nome = "bm25"

    def __init__(self, trechos: list[Trecho], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = {t.id: Counter(tokens(t.indexado)) for t in trechos}
        self.tam = {i: sum(c.values()) for i, c in self.tf.items()}
        self.media = sum(self.tam.values()) / len(self.tam)
        df: Counter = Counter()
        for c in self.tf.values():
            df.update(c.keys())
        n = len(trechos)
        self.idf = {p: math.log(1 + (n - d + 0.5) / (d + 0.5)) for p, d in df.items()}

    def buscar(self, consulta: str) -> list[tuple[str, float]]:
        termos = set(tokens(consulta))
        pontos: dict[str, float] = {}
        for id_, tf in self.tf.items():
            s = 0.0
            for t in termos:
                f = tf.get(t, 0)
                if f:
                    norm = f + self.k1 * (1 - self.b + self.b * self.tam[id_] / self.media)
                    s += self.idf[t] * f * (self.k1 + 1) / norm
            pontos[id_] = s
        return _ordenar(pontos)


class TfIdf:
    nome = "tfidf"

    def __init__(self, trechos: list[Trecho]):
        contagens = {t.id: Counter(tokens(t.indexado)) for t in trechos}
        df: Counter = Counter()
        for c in contagens.values():
            df.update(c.keys())
        n = len(trechos)
        self.idf = {p: math.log((1 + n) / (1 + d)) + 1 for p, d in df.items()}
        self.vetores = {i: self._vetor(c) for i, c in contagens.items()}

    def _vetor(self, contagem: Counter) -> dict[str, float]:
        v = {p: (1 + math.log(f)) * self.idf[p] for p, f in contagem.items() if p in self.idf}
        norma = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {p: x / norma for p, x in v.items()}

    def buscar(self, consulta: str) -> list[tuple[str, float]]:
        q = self._vetor(Counter(tokens(consulta)))
        pontos = {i: sum(w * v.get(p, 0.0) for p, w in q.items()) for i, v in self.vetores.items()}
        return _ordenar(pontos)


class Hibrido:
    """Reciprocal Rank Fusion: soma 1/(k + posição) de cada recuperador."""

    def __init__(self, recuperadores: list[Recuperador], k: int = 60):
        self.recuperadores, self.k = recuperadores, k
        self.nome = "hibrido(" + "+".join(r.nome for r in recuperadores) + ")"

    def buscar(self, consulta: str) -> list[tuple[str, float]]:
        pontos: dict[str, float] = {}
        for r in self.recuperadores:
            for pos, (id_, _) in enumerate(r.buscar(consulta), start=1):
                pontos[id_] = pontos.get(id_, 0.0) + 1.0 / (self.k + pos)
        return _ordenar(pontos)


class ComExpansao:
    """Acrescenta sinônimos à consulta antes de buscar (ex.: inquilino -> locatario)."""

    def __init__(self, base: Recuperador, sinonimos: dict[str, list[str]]):
        self.base = base
        self.sinonimos = {normalizar(k): [normalizar(v) for v in vs] for k, vs in sinonimos.items()}
        self.nome = base.nome + "+expansao"

    def expandir(self, consulta: str) -> str:
        extras: list[str] = []
        for palavra in tokens(consulta, usar_radical=False):
            extras.extend(self.sinonimos.get(palavra, []))
        return consulta + " " + " ".join(extras) if extras else consulta

    def buscar(self, consulta: str) -> list[tuple[str, float]]:
        return self.base.buscar(self.expandir(consulta))


class Aleatorio:
    """Linha de base: ordem embaralhada com semente fixa. Qualquer método precisa superá-la."""

    nome = "aleatorio"

    def __init__(self, trechos: list[Trecho], semente: int = 7):
        self.ids = sorted(t.id for t in trechos)
        self.semente = semente

    def buscar(self, consulta: str) -> list[tuple[str, float]]:
        rng = random.Random(f"{self.semente}:{consulta}")
        ordem = self.ids[:]
        rng.shuffle(ordem)
        n = len(ordem)
        return [(i, float(n - pos)) for pos, i in enumerate(ordem)]
