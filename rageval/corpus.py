"""Carrega documentos .md e os divide em trechos (um parágrafo por trecho)."""
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Trecho:
    id: str       # "<documento>#<n>", com n começando em 1 depois do título
    doc: str
    titulo: str
    texto: str

    @property
    def indexado(self) -> str:
        """Texto usado na busca: o título do documento dá contexto ao parágrafo."""
        return f"{self.titulo}. {self.texto}"


def dividir(doc: str, conteudo: str) -> list[Trecho]:
    blocos = [b.strip() for b in re.split(r"\n\s*\n", conteudo.strip()) if b.strip()]
    titulo = doc
    if blocos and blocos[0].startswith("# "):
        titulo = blocos[0][2:].strip()
        blocos = blocos[1:]
    return [Trecho(f"{doc}#{n}", doc, titulo, " ".join(b.split())) for n, b in enumerate(blocos, start=1)]


def carregar_corpus(pasta: str | Path) -> list[Trecho]:
    trechos: list[Trecho] = []
    for arquivo in sorted(Path(pasta).glob("*.md")):
        trechos.extend(dividir(arquivo.stem, arquivo.read_text(encoding="utf-8")))
    if not trechos:
        raise FileNotFoundError(f"Nenhum .md encontrado em {pasta}")
    return trechos
