import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from rageval.avaliacao import Pergunta, avaliar, carregar_golden, validar_golden
from rageval.cli import RAIZ, construir, main
from rageval.corpus import carregar_corpus, dividir
from rageval.relatorio import gerar_html
from rageval.resposta import (
    ClienteAnthropic,
    ClienteFake,
    checar_citacoes,
    contem_fato,
    gerar_resposta,
    juiz_llm,
    montar_prompt,
)
from rageval.retrievers import BM25, Aleatorio

DADOS = RAIZ / "data"


class DadosReaisTest(unittest.TestCase):
    """Protege o conjunto de referência: um erro nele distorceria todas as métricas."""

    @classmethod
    def setUpClass(cls):
        cls.trechos = carregar_corpus(DADOS / "corpus")
        cls.perguntas = carregar_golden(DADOS / "golden.json")
        cls.sinonimos = json.loads((DADOS / "sinonimos.json").read_text(encoding="utf-8"))

    def test_golden_e_consistente_com_o_corpus(self):
        self.assertEqual(validar_golden(self.trechos, self.perguntas), [])

    def test_ids_de_trechos_sao_unicos(self):
        ids = [t.id for t in self.trechos]
        self.assertEqual(len(ids), len(set(ids)))

    def test_bm25_supera_muito_o_aleatorio(self):
        bm = avaliar(BM25(self.trechos), self.perguntas, 3).metricas()
        rnd = avaliar(Aleatorio(self.trechos), self.perguntas, 3).metricas()
        self.assertGreater(bm["recall@3"], rnd["recall@3"] + 0.5)

    def test_expansao_nao_piora_o_bm25(self):
        recs = {r.nome: r for r in construir(self.trechos, self.sinonimos)}
        base = avaliar(recs["bm25"], self.perguntas, 3).metricas()["recall@3"]
        exp = avaliar(recs["bm25+expansao"], self.perguntas, 3).metricas()["recall@3"]
        self.assertGreaterEqual(exp, base)

    def test_ha_falhas_para_analisar(self):
        resumo = avaliar(BM25(self.trechos), self.perguntas, 3)
        self.assertGreater(len(resumo.falhas()), 0)


class ValidacaoDoGoldenTest(unittest.TestCase):
    def test_detecta_problemas(self):
        trechos = dividir("d", "# T\n\nO prazo é de 10 dias.")
        ruins = [
            Pergunta("a", "?", ("d#9",), "10 dias"),
            Pergunta("b", "?", ("d#1",), "99 dias"),
            Pergunta("b", "?", (), "x"),
        ]
        erros = " | ".join(validar_golden(trechos, ruins))
        self.assertIn("trecho inexistente d#9", erros)
        self.assertIn("não aparece", erros)
        self.assertIn("id repetido", erros)
        self.assertIn("sem trechos relevantes", erros)


class RespostasTest(unittest.TestCase):
    def setUp(self):
        self.trechos = dividir("g", "# Garantia\n\nA bateria tem garantia de 6 meses.\n\nNão cobre queda.")

    def test_prompt_rotula_cada_trecho(self):
        p = montar_prompt("quanto?", self.trechos)
        self.assertIn("[g#1] A bateria tem garantia de 6 meses.", p)
        self.assertIn("Pergunta: quanto?", p)

    def test_fake_responde_com_citacao_do_primeiro_trecho(self):
        r = gerar_resposta(ClienteFake(), "quanto?", self.trechos)
        self.assertTrue(r.endswith("[g#1]"))
        self.assertTrue(contem_fato(r, "6 meses"))

    def test_checagem_de_citacoes(self):
        self.assertEqual(checar_citacoes("6 meses [g#1]", ["g#1", "g#2"])["status"], "fundamentada")
        self.assertEqual(checar_citacoes("6 meses", ["g#1"])["status"], "sem_citacao")
        r = checar_citacoes("6 meses [x#7]", ["g#1"])
        self.assertEqual((r["status"], r["invalidas"]), ("citacao_invalida", ["x#7"]))

    def test_citacao_mista_com_uma_invalida_reprova(self):
        self.assertEqual(checar_citacoes("a [g#1] b [g#9]", ["g#1"])["status"], "citacao_invalida")

    def test_cliente_fake_sem_trechos(self):
        self.assertIn("Não encontrei", ClienteFake().responder("s", "Pergunta: x"))

    def test_juiz_le_o_veredito(self):
        class Sim:
            def responder(self, s, u): return "SIM, está sustentada."
        class Nao:
            def responder(self, s, u): return "Não."
        self.assertTrue(juiz_llm(Sim(), "q", "r", self.trechos))
        self.assertFalse(juiz_llm(Nao(), "q", "r", self.trechos))

    def test_cliente_anthropic_monta_a_requisicao_e_le_a_resposta(self):
        chamadas = []

        def post(url, cabecalhos, corpo):
            chamadas.append((url, cabecalhos, corpo))
            return {"content": [{"type": "text", "text": "6 meses [g#1]"}, {"type": "tool_use"}]}

        c = ClienteAnthropic(chave="k", modelo="m", post=post)
        self.assertEqual(c.responder("sis", "usu"), "6 meses [g#1]")
        url, cab, corpo = chamadas[0]
        self.assertEqual(url, "https://api.anthropic.com/v1/messages")
        self.assertEqual(cab["x-api-key"], "k")
        self.assertEqual(corpo["system"], "sis")
        self.assertEqual(corpo["messages"], [{"role": "user", "content": "usu"}])

    def test_cliente_anthropic_exige_chave(self):
        import os
        antigo = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            with self.assertRaises(ValueError):
                ClienteAnthropic(chave="")
        finally:
            if antigo is not None:
                os.environ["ANTHROPIC_API_KEY"] = antigo


class CliERelatorioTest(unittest.TestCase):
    def rodar(self, *args):
        saida = io.StringIO()
        with redirect_stdout(saida):
            codigo = main(list(args))
        return codigo, saida.getvalue()

    def test_validar(self):
        codigo, texto = self.rodar("validar")
        self.assertEqual(codigo, 0)
        self.assertIn("0 problema(s)", texto)

    def test_avaliar_mostra_todos_os_recuperadores(self):
        _, texto = self.rodar("avaliar", "-k", "3")
        for nome in ("aleatorio", "tfidf", "bm25", "bm25+expansao", "hibrido"):
            self.assertIn(nome, texto)
        self.assertIn("recall@3", texto)

    def test_falhas_lista_o_esperado(self):
        _, texto = self.rodar("falhas", "--recuperador", "bm25")
        self.assertIn("esperado:", texto)
        self.assertIn("falha(s)", texto)

    def test_perguntar_com_cliente_fake(self):
        _, texto = self.rodar("perguntar", "Qual a garantia da bateria do Orion 14?")
        self.assertIn("garantia-notebook#2", texto)
        self.assertIn("Citações: fundamentada", texto)
        self.assertIn("Contém o fato esperado: True", texto)

    def test_recuperador_desconhecido_encerra_com_mensagem(self):
        with self.assertRaises(SystemExit):
            self.rodar("falhas", "--recuperador", "nao-existe")

    def test_relatorio_html(self):
        with tempfile.TemporaryDirectory() as d:
            destino = Path(d) / "sub" / "index.html"
            self.rodar("relatorio", "--saida", str(destino))
            html = destino.read_text(encoding="utf-8")
        self.assertIn("<!doctype html>", html)
        self.assertIn("bm25+expansao", html)
        self.assertIn('lang="pt-BR"', html)

    def test_relatorio_escapa_html_das_perguntas(self):
        trechos = dividir("d", "# T\n\nO prazo é 10 dias.")
        p = Pergunta("q1", "<script>alert(1)</script> prazo", ("d#1",), "10 dias")
        html = gerar_html([avaliar(BM25(trechos), [p], 3)], trechos, 3)
        self.assertNotIn("<script>alert(1)</script>", html)
        self.assertIn("&lt;script&gt;", html)


if __name__ == "__main__":
    unittest.main()
