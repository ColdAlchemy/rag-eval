# Como saber se a busca do seu RAG está boa (e por que quase ninguém mede)

*Albert Ryan Brito dos Santos · novembro de 2026. Os números vêm do repositório [rag-eval](https://github.com/ColdAlchemy/rag-eval), que tem corpus e perguntas fictícios. Desenvolvi o projeto com o auxílio do Claude Code e estudei o código.*

## O problema
Um sistema de RAG (busca + geração) responde em dois passos. Primeiro **procura** trechos de documentos que parecem responder à pergunta; depois um modelo **escreve** a resposta usando esses trechos. A conversa costuma girar em torno do segundo passo: qual modelo, qual prompt. Mas se o primeiro passo não traz o trecho certo, nenhum modelo salva a resposta: ele vai escrever com convicção sobre o texto errado.

Quando construí o [DocsIA](https://github.com/ColdAlchemy/docs-ia-laravel), um arquivo de documentos com RAG e um agente que cita a fonte, percebi que eu tinha um avaliador das **citações**, mas nenhuma medida de se a **busca** estava boa. O `rag-eval` nasceu para isso.

## O método: um simulado com gabarito
A ideia é simples. Monte um conjunto de perguntas em que você **sabe qual trecho contém a resposta**. Rode a busca e veja se o trecho certo aparece entre os primeiros resultados. Fiz 34 perguntas sobre 8 documentos fictícios (contrato de aluguel, política de reembolso, garantia de um notebook e outros), com **43 trechos**, e escrevi algumas já parafraseadas de propósito: "cerveja" em vez de "bebidas alcoólicas", "derrubar no chão" em vez de "queda".

Três métricas bastam para começar, todas com um trecho relevante por pergunta:
- **Recall@3**: o trecho certo está entre os 3 primeiros? (sim ou não, depois a média)
- **MRR**: quão no topo ele está? Primeiro lugar vale 1, segundo 0,5, terceiro 0,33.
- **nDCG@3**: parecido com o MRR, mas considera vários trechos relevantes e desconta pela posição.

Duas regras que fizeram diferença: sempre incluir uma **linha de base aleatória** (sem ela, "0,85" não significa nada) e **validar o próprio gabarito** (cada id existe e o fato esperado aparece mesmo no trecho). Um erro no gabarito distorce tudo em silêncio.

## O que apareceu
Comparei recuperadores que escrevi do zero, sem bibliotecas:

| recuperador | recall@3 | mrr | ndcg@3 |
|---|---|---|---|
| aleatório | 0,088 | 0,111 | 0,059 |
| tf-idf | 0,853 | 0,787 | 0,799 |
| bm25 | 0,853 | 0,787 | 0,799 |
| tf-idf de trigramas | 0,853 | 0,827 | 0,820 |
| bm25 + sinônimos | 0,882 | 0,816 | 0,828 |
| bm25 + trigramas (híbrido) | 0,882 | 0,831 | 0,835 |

Quatro lições, nenhuma delas óbvia para mim antes de medir:

**1. Métricas iguais não significam métodos iguais.** BM25 e TF-IDF empataram em todas as colunas, mas em 17 das 34 perguntas o ranking completo era diferente. As diferenças só não mexiam na posição do trecho certo, neste corpus pequeno. Olhar só a média escondia isso.

**2. O ganho de uma lista de sinônimos é suspeito.** Escrevi uma lista depois de ver as falhas, e ela recuperou uma pergunta. Mas isso é **vazamento**: ajustei a solução às perguntas que eu já tinha visto. Por isso acrescentei 8 perguntas novas, parafraseadas, sem ajustar nada para elas.

**3. Comparar pedaços de palavra resolve sem lista nenhuma.** Os trigramas de caracteres ("capital" e "capitais" compartilham "cap", "api", "pit") chegaram ao mesmo recall da lista de sinônimos, **sem nenhuma lista escrita à mão**, e com o melhor MRR. É o resultado em que mais confio, justamente por não depender de eu ter visto as respostas.

**4. A lista de falhas vale mais que a média.** Sobraram 4 perguntas erradas por todos os métodos, e as quatro são do mesmo tipo: faltou **significado**, não grafia. "Som alto à noite" não tem nenhuma palavra em comum com "barulho". Nenhuma busca por palavras resolve isso. É o ponto em que embeddings passam a valer a pena, e agora eu teria como **medir** o ganho com a mesma ferramenta em vez de acreditar nele.

## Limites do que eu fiz
- O corpus é pequeno e fictício: os números servem para comparar métodos aqui, não para prometer desempenho em dados reais.
- Não testei embeddings nem um modelo de linguagem real. A ferramenta tem um cliente da API da Anthropic, mas ele foi testado só com HTTP simulado.
- O conjunto de perguntas foi escrito por mim, o que também tem viés.

## O que levo disso
Medir a busca separadamente da geração muda as decisões. Em vez de "trocar o modelo para ver se melhora", dá para perguntar **qual pergunta falhou e por quê**, e corrigir a parte certa. O custo é pequeno: algumas dezenas de perguntas com gabarito e meia hora de código.

Código, testes e o relatório completo: https://github.com/ColdAlchemy/rag-eval
