# rag-eval — como saber se a busca do seu RAG está boa

Ferramenta em Python (só biblioteca padrão) que **mede a recuperação de um sistema RAG**: dado um conjunto de perguntas com a resposta certa conhecida, ela compara vários recuperadores com Recall@k, MRR e nDCG, lista **onde cada um errou**, confere se as respostas **citam trechos que de fato foram recuperados** e gera um relatório HTML.

> Desenvolvido por mim com o auxílio do Claude Code. Eu defini o escopo, revisei e estudei o código; o guia `docs/ESTUDO.md` explica cada conceito. O corpus e as perguntas são fictícios.

## Para que serve (em termos simples)
Muitos sistemas de IA funcionam em dois passos: **primeiro procuram** os trechos de documentos que respondem à pergunta, **depois** um modelo escreve a resposta com eles. Se o primeiro passo falha, a resposta sai errada mesmo com um modelo excelente, e quase ninguém mede isso. Esta ferramenta mede o primeiro passo: faz um "simulado" de perguntas com gabarito, mostra a nota de cada jeito de buscar e **aponta exatamente quais perguntas ele errou**. Com isso dá para melhorar a busca com evidência, em vez de por impressão.

Nasceu da pergunta que ficou aberta depois do [DocsIA](https://github.com/ColdAlchemy/docs-ia-laravel): ele faz RAG e tem um avaliador de citações, mas **a busca em si é boa?** Aqui ela é tratada como algo que se mede.

**Artigo:** [Como saber se a busca do seu RAG está boa](docs/ARTIGO.md) · **Relatório online:** https://coldalchemy.github.io/rag-eval/

## Resultado (34 perguntas, 43 trechos, k = 3)
```
recuperador                     recall@3      hit@3        mrr     ndcg@3
aleatorio                          0.088      0.088      0.111      0.059
tfidf                              0.853      0.853      0.787      0.799
tfidf-trigramas                    0.853      0.853      0.827      0.820
bm25                               0.853      0.853      0.787      0.799
bm25+expansao                      0.882      0.882      0.816      0.828
hibrido(bm25+expansao+tfidf)       0.882      0.882      0.816      0.828
hibrido(bm25+tfidf-trigramas)      0.882      0.882      0.831      0.835
```
Como ler isso, sem exagerar:
- O **aleatório** é a linha de base: qualquer método tem de bater 0,09.
- **BM25 e TF-IDF empatam nas métricas, mas não são iguais**: em 17 das 34 perguntas o ranking completo é diferente. Neste corpus pequeno, as diferenças só não mexem na posição do trecho certo.
- A **expansão de consulta** (sinônimos) recupera 1 pergunta a mais, mas esse ganho é **otimista**: a lista foi escrita depois de eu ver falhas do conjunto. Por isso as perguntas q27 a q34 foram acrescentadas já parafraseadas e os sinônimos **não** foram ajustados a elas.
- O **híbrido BM25 + trigramas de caracteres** chega ao mesmo Recall (0,882) **sem nenhuma lista escrita à mão** e com o melhor MRR (0,831). É o resultado mais confiável da tabela: resolve "capital" × "capitais" porque compara pedaços de palavra.
- Sobram 4 falhas (`python -m rageval falhas --recuperador "hibrido(bm25+tfidf-trigramas)"`), todas por **significado**, não por grafia ("derrubar no chão" × "queda", "som alto" × "barulho"). É o limite de qualquer busca por palavras e o motivo para testar embeddings depois, medindo com esta mesma ferramenta.
- Corpus pequeno e fictício: os números servem para comparar métodos aqui, não para prometer desempenho em dados reais.

## Uso
```bash
python -m rageval validar              # confere o corpus e o conjunto de referência
python -m rageval avaliar -k 3         # tabela comparando os recuperadores
python -m rageval falhas --recuperador bm25
python -m rageval perguntar "Qual a garantia da bateria do Orion 14?"
python -m rageval relatorio            # grava docs/index.html
python -m unittest                     # 41 testes, sem rede
```
**Nenhuma dependência.** Testado no Python 3.14 (a sintaxe usada exige 3.10 ou mais novo).

## O que tem dentro
| Módulo | Papel |
|---|---|
| `texto.py` | normalização (acentos, caixa), palavras vazias em português, radical leve |
| `corpus.py` | divide documentos `.md` em trechos (um parágrafo, id `documento#n`) |
| `retrievers.py` | **BM25**, **TF-IDF com cosseno** (por palavras e por trigramas de caracteres), **híbrido por RRF**, **expansão de consulta** e **aleatório**, todos escritos do zero |
| `metricas.py` | Recall@k, Hit@k, MRR, nDCG@k |
| `avaliacao.py` | roda o conjunto de referência, agrega, lista falhas e **valida o próprio conjunto** (ids existem; o fato esperado aparece nos trechos relevantes) |
| `resposta.py` | prompt com trechos rotulados, `ClienteFake` (offline), `ClienteAnthropic`, checagem de citações e um juiz LLM |
| `relatorio.py` | HTML estático, sem JavaScript, com tema claro/escuro |

## Decisões de projeto
- **Validar o conjunto de referência é parte do teste.** Um erro no gabarito distorce todas as métricas, então há testes que conferem cada id e cada fato esperado.
- **Linha de base aleatória** e **falhas por pergunta**, não só a média: a média esconde onde o sistema erra.
- **Citação só vale se o trecho foi recuperado.** Uma resposta que cita `[x#7]` quando `x#7` nunca foi entregue ao modelo é marcada `citacao_invalida`.
- **Trechos tratados como dados.** O prompt do sistema manda ignorar ordens que apareçam dentro dos trechos.
- **Cliente de IA é uma interface**: o padrão é offline e determinístico; o real é opcional.

## Limitações
- Busca **lexical**: não entende sinônimos que não estejam na lista. Embeddings de um provedor seriam o próximo passo, e esta ferramenta serviria para medir o ganho.
- `ClienteAnthropic` está implementado e testado **só com HTTP simulado**; não foi exercitado contra a API real. O `ClienteFake` apenas devolve o primeiro trecho recebido, então a métrica "contém o fato" dele é um proxy da recuperação, não da qualidade de um modelo.
- Radical simplificado (tira o "s" final); não é um stemmer completo ("capital" × "capitais" não casam).
- Relevância binária, sem graus.
