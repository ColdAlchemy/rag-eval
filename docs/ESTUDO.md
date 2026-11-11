# Guia de estudo — rag-eval

Para entender cada parte e conseguir explicar em entrevista. Leia o código na ordem da seção 1 e faça os exercícios da seção 4.

## 1. Roteiro de leitura
1. `rageval/texto.py` e `corpus.py`: como o texto vira tokens e trechos.
2. `rageval/metricas.py` (30 linhas): as quatro métricas.
3. `rageval/retrievers.py`: BM25, TF-IDF, RRF, expansão.
4. `rageval/avaliacao.py`: o laço de avaliação e a validação do gabarito.
5. `rageval/resposta.py`: citações e juiz.
6. `tests/`: leia os testes como documentação do comportamento esperado.

## 2. Conceitos
**Recuperação × geração.** Num RAG, primeiro se *busca* trechos e depois um modelo *escreve* a resposta com eles. Se a busca não traz o trecho certo, nenhum modelo salva a resposta. Por isso se mede a busca separadamente.

**Conjunto de referência (golden set).** Perguntas com a resposta certa conhecida (aqui, qual trecho contém a resposta). Sem ele, "parece bom" é só opinião.

**Recall@k.** Dos trechos relevantes, que fração apareceu entre os k primeiros. Com 1 relevante, é 1 se apareceu e 0 se não (igual ao Hit@k).

**MRR.** 1 dividido pela posição do primeiro trecho certo (1º = 1, 2º = 0,5, 3º = 0,33). Premia colocar o certo no topo.

**nDCG@k.** Como o MRR, mas considera todos os relevantes e desconta por posição com 1/log2(posição+1), normalizado pelo melhor ranking possível.

**TF-IDF.** Peso de uma palavra = quantas vezes ela aparece no trecho (TF) × o quão rara ela é no corpus (IDF). Palavras comuns pesam pouco. A busca compara consulta e trecho pelo cosseno do ângulo entre os vetores.

**BM25.** Evolução do TF-IDF: a contribuição de repetir uma palavra *satura* (parâmetro k1) e trechos longos são penalizados (parâmetro b). É o padrão em motores como Elasticsearch/OpenSearch.

**RRF (Reciprocal Rank Fusion).** Junta rankings de métodos diferentes somando 1/(60 + posição). Não precisa comparar pontuações de escalas diferentes, só posições.

**Trigramas de caracteres.** Em vez de palavras inteiras, compara pedaços de 3 letras ("capital" e "capitais" compartilham "cap", "api", "pit"). Tolera plural, flexão e erro de digitação, sem lista de sinônimos.

**Expansão de consulta.** Acrescenta sinônimos antes de buscar, para cobrir "inquilino" × "locatário". Ajuda, mas exige manter a lista.

**Vazamento (overfitting ao conjunto de teste).** Se eu escrevo sinônimos olhando as perguntas que falharam, o ganho medido nelas é inflado. Por isso há perguntas novas (q27 a q34) que não foram usadas para ajustar nada.

**Fundamentação.** Resposta fundamentada cita trechos, e todo trecho citado foi de fato recuperado. É a mesma ideia do `AnswerEvaluator` do DocsIA.

## 3. Perguntas prováveis (treine em voz alta)
- *Por que medir a busca separada da resposta?* Porque são falhas diferentes: se o trecho certo não chega ao modelo, o erro é da busca; se chega e a resposta sai errada, é da geração.
- *Qual a diferença entre BM25 e TF-IDF?* Saturação de frequência e normalização por tamanho do trecho.
- *Por que o aleatório está na tabela?* É a linha de base; sem ela não dá para saber se o método ajuda.
- *Os números provam que o sistema é bom?* Não: corpus pequeno e fictício; servem para comparar métodos aqui. O que se leva para outros dados é o método de avaliar.
- *O que fazer com as falhas restantes?* Elas são de vocabulário; o próximo passo é testar embeddings e medir com esta mesma ferramenta.
- *Você testou a API real?* Não; o cliente está testado com HTTP simulado. Dizer isso é melhor que fingir.

## 4. Exercícios
1. Rode `python -m rageval falhas --recuperador bm25+expansao` e, para cada falha, escreva em uma frase por que a busca por palavras errou.
2. Mude `k1` do BM25 para 0,5 e 3,0 e veja se alguma métrica muda. Explique o resultado.
3. Tire o título do texto indexado (`Trecho.indexado`) e rode de novo. O que piora?
4. Escreva 5 perguntas novas com o corpus atual, sem olhar os sinônimos, e veja o que o sistema faz com elas.
5. Implemente um novo recuperador (por exemplo, TF-IDF de bigramas) e inclua-o em `cli.construir`.

## 5. O que dizer sobre a sua experiência (sem exagerar)
"Construí um avaliador de recuperação para RAG em Python, com BM25, TF-IDF, RRF e métricas Recall, MRR e nDCG implementados do zero, com conjunto de referência validado e 41 testes. Desenvolvi com o auxílio do Claude Code e estudei o código. Usei corpus fictício e não testei a API real de um modelo." Não diga que "otimizou um RAG em produção" nem que tem "experiência com embeddings": não é o caso.
