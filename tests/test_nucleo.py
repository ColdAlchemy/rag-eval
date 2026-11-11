import math
import unittest

from rageval.corpus import Trecho, dividir
from rageval.metricas import hit_at_k, mrr, ndcg_at_k, recall_at_k
from rageval.retrievers import BM25, Aleatorio, ComExpansao, Hibrido, TfIdf, trigramas
from rageval.texto import normalizar, radical, tokens

DOC = """# Regras da piscina

A piscina abre às 8h e fecha às 20h.

Crianças menores de 10 anos entram acompanhadas.

O uso de boias grandes é proibido.
"""


def mini_corpus() -> list[Trecho]:
    return dividir("piscina", DOC) + dividir("gato", "# Cuidados com o gato\n\nO gato come duas vezes por dia.\n\nA caixa de areia é limpa todo dia.")


class TextoTest(unittest.TestCase):
    def test_normaliza_acentos_e_caixa(self):
        self.assertEqual(normalizar("AÇÃO Rápida"), "acao rapida")

    def test_tokens_removem_palavras_vazias_e_mantem_numeros(self):
        self.assertEqual(tokens("Quantos livros até 5 dias?"), ["livro", "5", "dias"])

    def test_radical_tira_plural_simples(self):
        self.assertEqual(radical("livros"), "livro")
        self.assertEqual(radical("dias"), "dias")  # palavra curta: não mexe


class CorpusTest(unittest.TestCase):
    def test_divide_em_paragrafos_numerados_depois_do_titulo(self):
        trechos = dividir("piscina", DOC)
        self.assertEqual([t.id for t in trechos], ["piscina#1", "piscina#2", "piscina#3"])
        self.assertEqual(trechos[0].titulo, "Regras da piscina")
        self.assertTrue(trechos[1].texto.startswith("Crianças"))

    def test_sem_titulo_usa_o_nome_do_documento(self):
        self.assertEqual(dividir("x", "só um parágrafo")[0].titulo, "x")

    def test_texto_indexado_inclui_o_titulo(self):
        self.assertIn("Regras da piscina", dividir("piscina", DOC)[0].indexado)


class RecuperadoresTest(unittest.TestCase):
    def setUp(self):
        self.trechos = mini_corpus()

    def test_bm25_acha_o_trecho_certo(self):
        ranking = BM25(self.trechos).buscar("a que horas a piscina fecha")
        self.assertEqual(ranking[0][0], "piscina#1")

    def test_tfidf_acha_o_trecho_certo(self):
        self.assertEqual(TfIdf(self.trechos).buscar("boias grandes")[0][0], "piscina#3")

    def test_sem_termos_em_comum_nao_devolve_nada(self):
        self.assertEqual(BM25(self.trechos).buscar("astronomia quântica"), [])

    def test_ranking_e_ordenado_por_pontuacao_decrescente(self):
        pontos = [p for _, p in BM25(self.trechos).buscar("gato piscina dia")]
        self.assertEqual(pontos, sorted(pontos, reverse=True))

    def test_termo_raro_pesa_mais_que_termo_comum(self):
        bm = BM25(self.trechos)
        # "boia" está em 1 trecho; "dia" em 2
        self.assertGreater(bm.idf["boia"], bm.idf["dia"])
        self.assertGreater(bm.idf["dia"], 0)

    def test_hibrido_rrf_combina_os_dois(self):
        bm, tf = BM25(self.trechos), TfIdf(self.trechos)
        ranking = Hibrido([bm, tf]).buscar("boias grandes")
        self.assertEqual(ranking[0][0], "piscina#3")
        esperado = 1 / 61 + 1 / 61
        self.assertAlmostEqual(ranking[0][1], esperado)

    def test_expansao_acrescenta_sinonimos(self):
        exp = ComExpansao(BM25(self.trechos), {"felino": ["gato"]})
        self.assertEqual(BM25(self.trechos).buscar("felino"), [])  # sem a expansão, nenhuma palavra em comum
        self.assertEqual({i for i, _ in exp.buscar("felino")}, {"gato#1", "gato#2"})
        self.assertEqual(exp.nome, "bm25+expansao")

    def test_trigramas_toleram_flexao(self):
        comuns = set(trigramas("capital")) & set(trigramas("capitais"))
        self.assertTrue({"cap", "api", "pit"} <= comuns)
        self.assertEqual(trigramas("a"), [])  # palavra vazia de sentido (stopword) não gera trigramas

    def test_tfidf_de_trigramas_acha_variacao_da_palavra(self):
        trechos = dividir("c", "# Condomínio\n\nAs capitais pagam mais.\n\nO gato dorme.")
        rec = TfIdf(trechos, trigramas, "tri")
        self.assertEqual(rec.nome, "tri")
        self.assertEqual(rec.buscar("capital")[0][0], "c#1")

    def test_aleatorio_e_deterministico_e_completo(self):
        a = Aleatorio(self.trechos)
        self.assertEqual(a.buscar("qualquer"), a.buscar("qualquer"))
        self.assertEqual(len(a.buscar("qualquer")), len(self.trechos))


class MetricasTest(unittest.TestCase):
    def test_recall_e_hit(self):
        ranking, rel = ["a", "b", "c", "d"], {"b", "d"}
        self.assertEqual(recall_at_k(ranking, rel, 2), 0.5)
        self.assertEqual(recall_at_k(ranking, rel, 4), 1.0)
        self.assertEqual(hit_at_k(ranking, rel, 1), 0.0)
        self.assertEqual(hit_at_k(ranking, rel, 2), 1.0)

    def test_mrr(self):
        self.assertEqual(mrr(["a", "b", "c"], {"c"}), 1 / 3)
        self.assertEqual(mrr(["a"], {"z"}), 0.0)

    def test_ndcg_perfeito_e_com_erro(self):
        self.assertAlmostEqual(ndcg_at_k(["a", "b"], {"a"}, 2), 1.0)
        # único relevante na posição 2: (1/log2(3)) / (1/log2(2))
        self.assertAlmostEqual(ndcg_at_k(["x", "a"], {"a"}, 2), 1 / math.log2(3))
        self.assertEqual(ndcg_at_k(["x", "y"], {"a"}, 2), 0.0)

    def test_sem_relevantes_nao_quebra(self):
        self.assertEqual(recall_at_k(["a"], set(), 3), 0.0)
        self.assertEqual(ndcg_at_k(["a"], set(), 3), 0.0)


if __name__ == "__main__":
    unittest.main()
