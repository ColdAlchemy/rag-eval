"""Normalização e tokenização de texto em português (sem dependências externas)."""
import re
import unicodedata

PARADAS = frozenset(
    "a o as os de da do das dos em no na nos nas um uma uns umas e ou que qual quais quanto quantos "
    "quantas quando como por para com se ao aos pelo pela pelos pelas mais mas seu sua seus suas meu "
    "minha meus minhas eu voce voces isso esse essa este esta ate sobre entre apos desde sem ja foi "
    "ser sao tem ha vez vezes".split()
)


def sem_acento(texto: str) -> str:
    decomposto = unicodedata.normalize("NFD", texto)
    return "".join(c for c in decomposto if unicodedata.category(c) != "Mn")


def normalizar(texto: str) -> str:
    return sem_acento(texto.lower())


def radical(palavra: str) -> str:
    """Radical bem leve: tira o 's' final de plurais (livros -> livro). Não é um stemmer completo."""
    if len(palavra) > 4 and palavra.endswith("s"):
        return palavra[:-1]
    return palavra


def tokens(texto: str, usar_radical: bool = True) -> list[str]:
    saida = []
    for palavra in re.findall(r"[a-z0-9]+", normalizar(texto)):
        if palavra in PARADAS:
            continue
        if len(palavra) < 2 and not palavra.isdigit():
            continue
        saida.append(radical(palavra) if usar_radical else palavra)
    return saida
