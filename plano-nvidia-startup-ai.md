# Decisões e especificação técnica — NVIDIA Startup AI Radar

## 1. Desisões e Planos

### Decisão 1 — Outbound como motor principal, inbound como porta de entrada complementar

O sistema é **outbound por padrão**: o gerente de Startups & VCs dispara uma consulta (setor, tema, sinal de mercado) e o pipeline vai atrás das startups. Mas o projeto também vai expor um **modo inbound leve**: um formulário simples onde a própria startup se cadastra e entra direto no banco estruturado, pulando a etapa de scraping (ela já fornece os dados) mas passando pelas mesmas etapas de classificação, RAG e recomendação. Os dois fluxos convergem no mesmo schema de perfil (seção 2) e no mesmo pipeline a partir do Startup Classifier Agent. Isso significa que tecnicamente você só precisa construir **uma entrada alternativa para o estado do grafo**, não um sistema paralelo.

### Passo 2 — O que estamos procurando: schema do perfil + sinais de AI-native vs. wrapper

Esse é o contrato de dados que todo agente do pipeline lê e escreve. Definir isso agora evita refatorar o grafo inteiro depois.

**Schema do perfil da startup** (o que cada registro no banco estruturado deve conter):

```
StartupProfile {
  id, nome, site, ano_fundacao, setor, subsetor,
  estagio_funding, valor_captado_total, investidores[], headcount_estimado,
  origem: "outbound" | "inbound",

  produto_descricao, publico_alvo,

  sinais_ai_native: [ {sinal, evidencia_trecho, fonte_url, data_coleta} ],
  sinais_wrapper_risco: [ {sinal, evidencia_trecho, fonte_url, data_coleta} ],
  stack_tecnica_detectada: [tecnologia],
  stack_concorrente_detectada: [ "Bedrock" | "Vertex AI" | "Azure OpenAI" | ... ],

  score_maturidade_ia: number,
  classificacao: "AI-native" | "AI-enabled" | "non-AI" | "indeterminado",

  evidencias: [ {fonte_url, trecho_resumido, data_coleta} ],
  casos_similares: [ {case_id, tipo: "sucesso"|"fracasso", similaridade} ],

  recomendacoes_nvidia: [
    {tecnologia, justificativa_tecnica, justificativa_negocio,
     prioridade, complexidade, proxima_acao, evidencias}
  ],

  estimativa_economica: {cenario_assumido, economia_estimada, fonte_benchmark, nivel_confianca},

  ultima_atualizacao
}
```

**Sinais de AI-native** (o que o Extractor/Classifier Agent deve procurar no texto coletado — cada sinal encontrado é uma linha em `sinais_ai_native`, sempre com a evidência e a fonte, nunca uma classificação sem rastro):

- Menção a modelo proprietário, fine-tuning próprio, ou dataset proprietário ("treinamos nosso próprio modelo", "dataset exclusivo")
- Menção a infraestrutura própria: GPU, self-hosted, on-premise, otimização de custo/latência de inferência
- Vagas abertas para ML Engineer, MLOps, Platform/Infra Engineer, Data Engineer (não só growth/vendas)
- Atuação em setor regulado com compliance citado (LGPD, BACEN, ANVISA, HIPAA) — setor regulado tende a exigir profundidade técnica real
- Integração profunda com sistemas legados do cliente (ERP, CRM, EHR, core bancário) em vez de só uma interface de chat
- Parcerias com universidades/centros de pesquisa (sinal de P&D real, como no caso da Noleak — ver Passo 3)
- Linguagem de "automação de processo/trabalho" em vez de "converse com seus documentos"
- Já cita uso de GPU/infraestrutura NVIDIA ou de outro provedor de cômputo dedicado

**Sinais de risco "wrapper"** (não são acusação — são sinais de atenção que pedem investigação mais profunda antes de classificar):

- Diferencial central comunicado é "powered by GPT-4/ChatGPT" sem nenhuma camada técnica própria mencionada
- Nenhuma vaga técnica aberta, só vendas/growth/marketing
- Categoria de produto facilmente replicável por um update nativo do provedor de modelo (ex.: "chat com PDF", "gerador de post")
- Múltiplos pivots de produto em pouco tempo (sinal via notícias/redes sociais)
- Dependência total de uma única API de terceiro, sem menção de fallback ou estratégia multi-modelo

Esses dois conjuntos de sinais são literalmente o prompt do Startup Classifier Agent — ele não "decide com base em vibe", ele soma evidências de cada lista e gera o `score_maturidade_ia` com a lista de evidências anexada.

### Passo 3 — Golden set: perguntas norteadoras + startups reais que respondem cada uma

Adaptando a lógica de revisão de literatura (pergunta → estudo que responde) para esse case: aqui a "pergunta" é uma característica que queremos que o classificador saiba reconhecer, e a "resposta" é um caso real e documentado que ilustra essa característica — positiva ou negativamente. Isso vira seu golden set de avaliação (você anota manualmente como cada caso *deveria* ser classificado e usa isso para validar o Classifier e o Recommendation Agent).

**P1 — Como é uma startup que nasceu com IA no núcleo do negócio, não como feature?**
A CloudWalk, dona do InfinitePay, usa IA e blockchain proprietário para processamento de pagamentos, atende micro e pequenos empreendedores em mais de 5.000 municípios e já operou a internacionalização do negócio com a marca Jim.com nos Estados Unidos. A QI Tech se tornou o único novo unicórnio latino-americano de 2024 usando IA para análise de crédito e automação de operações financeiras. Resposta esperada do classificador: AI-native, alta prioridade.

**P2 — Como é uma startup vertical, com dado proprietário, em setor regulado (perfil que tende a sobreviver)?**
A Laura aplica IA cognitiva ao monitoramento hospitalar e à detecção precoce de riscos clínicos como sepse, coletando dados em tempo real de prontuários eletrônicos e emitindo alertas preditivos. A OncoAI oferece uma plataforma de IA para prever a recorrência do câncer e otimizar decisões de tratamento. Resposta esperada: AI-native, setor saúde, alta prioridade para NeMo Guardrails (compliance) + Clara.

**P3 — Como é uma startup que já usa stack técnica próxima da NVIDIA (alto fit de ICP)?**
A Noleak usa NVIDIA Metropolis, GPUs, TensorRT e Triton Inference Server para detectar comportamentos suspeitos por câmeras de segurança, com tecnologia proprietária desenvolvida em parceria com universidades brasileiras e canadenses. A Mr. Turing combina visão computacional e processamento de linguagem natural para interpretar documentos de forma "quase humana", rodando em instâncias GPU NVIDIA T4 tanto em desenvolvimento quanto em produção na AWS. Resposta esperada: já é cliente Inception, prioridade é aprofundar (upsell de Triton/TensorRT-LLM), não convencer do zero.

**P4 — Como é uma startup de infraestrutura B2B ("pick and shovel"), candidata a virar unicórnio AI-native?**
A Stark Bank oferece infraestrutura bancária (PIX, boletos, transferências) para outras empresas, com automação por IA desde o início, e é apontada como potencial candidata ao primeiro unicórnio de IA nativo do Brasil. Resposta esperada: AI-native, prioridade alta mesmo não sendo consumer-facing.

**P5 — Como é o padrão de fracasso "wrapper fino" mesmo levantando capital?**
A Wuri começou como app de geração de novelas visuais com IA generativa, pivotou para soluções de IA empresarial e "wrappers de IA", passou pela Y Combinator, mas nunca travou em um mercado específico — conforme plataformas maiores lançavam recursos próprios, sua oferta passou a parecer infraestrutura comoditizada com uma camada de interface fina. Resposta esperada do classificador: wrapper, risco alto, não prioritário (ou prioritário só se houver sinal claro de pivot para profundidade técnica).

**P6 — Como é o padrão de "ser engolido" pelo próprio provedor de modelo?**
A CodeWhisper fechou em julho de 2025 depois que a OpenAI lançou um modo de debug avançado que fazia tudo que o produto fazia, incluído no plano ChatGPT Plus de US$20/mês. Resposta esperada: sinal de alerta para qualquer startup cujo diferencial inteiro seja replicável por uma atualização nativa de modelo.

**P7 — Como é o padrão de fracasso por motivos não-técnicos, mesmo com tecnologia validada (relevante para o ICP de saúde)?**
A Cydoc operou por sete anos como startup de IA em saúde, alcançou clientes pagantes, patentes e impacto clínico demonstrado, mas seu fundador concluiu que implantar IA em saúde é só 20% do desafio — os outros 80%, como integração de workflow, infraestrutura de vendas e modelo de negócio sustentável, são onde a maioria das health AI companies falha. Resposta esperada: mesmo com sinais_ai_native fortes, o briefing deveria sinalizar risco de adoção lenta em vendas/integração — é exatamente o tipo de nuance qualitativa que só aparece se você registrar casos de fracasso, não só de sucesso (ver seção 4.1).

> Use essas 7 entradas como seu primeiro golden set real. Para cada uma, anote manualmente: classificação esperada, tecnologias NVIDIA que fariam sentido (ou não) e por quê. Quando o Classifier/Recommendation Agent rodar sobre esses mesmos casos, compare a saída com sua anotação — é sua métrica de qualidade sem precisar de um dataset acadêmico.

---

## 2. Tecnologias que serão utilizadas — pra quê, por quê, como, onde

| Tecnologia | Pra quê (função no pipeline) | Por quê essa escolha | Onde entra |
|---|---|---|---|
| LangGraph | Orquestrar os 8 agentes com estado compartilhado e transições condicionais | Controle explícito sobre o fluxo, checkpoint nativo, padrão supervisor bem documentado | Orquestração central, schema do StartupProfile como estado |
| Playwright | Scraping de páginas dinâmicas (carreiras, sites institucionais com JS) | Lida com conteúdo renderizado em JS, mais robusto que requests simples | Scraper Agent |
| trafilatura | Extrair o texto principal de páginas/blogs/notícias, removendo menu/rodapé | Leve, rápido, não precisa de chave de API | Scraper Agent (fontes simples) |
| Firecrawl | Extração limpa pronta para RAG, com fallback de busca quando o crawl não traz dado suficiente | Reduz parsing manual; é a peça central do projeto ai-company-researcher, usável como referência direta | Scraper Agent (fallback / fontes complexas) |
| PostgreSQL | Armazenar o StartupProfile estruturado | Suporta queries relacionais (filtrar por setor, score, classificação) | Banco estruturado de startups + banco histórico (4.1) |
| Qdrant | Banco vetorial da base de conhecimento NVIDIA e do banco histórico de casos | Roda local em Docker, suporta busca híbrida nativamente | NVIDIA RAG Agent |
| BM25 | Busca lexical complementar à busca vetorial | Captura nomes exatos de tecnologia que embeddings às vezes diluem | NVIDIA RAG Agent (busca híbrida) |
| Reranker (Cohere Rerank ou BGE-Reranker local) | Reordenar os trechos recuperados pela relevância real à pergunta | Busca híbrida traz uma aproximação; reranking melhora a precisão dos top-k | NVIDIA RAG Agent -> Reranker |
| NVIDIA NIM | Servir modelos otimizados via API padronizada | É o que você vai recomendar às startups — vale "dogfoodar" no próprio pipeline | Recommendation/Briefing Agent |
| NeMo Guardrails | Validar que a saída não alucina e segue formato esperado | Mesma tecnologia recomendada às startups, reforça credibilidade técnica | Evidence Validator / saída do Briefing Agent |
| LangSmith ou Langfuse | Observabilidade: rastrear execução do grafo, custo, latência, erro por agente | Essencial para debugar um grafo com 8 nós | Todo o pipeline |
| Streamlit | Dashboard de consulta, visualização e exportação do briefing | Mais rápido para um MVP orientado a dados | Interface web |

### Ferramentas para acelerar (forks/templates, não comece do zero)

- mayooear/ai-company-researcher (GitHub) — LangGraph + Firecrawl, já implementa rotas condicionais e fallback de busca. Use como esqueleto do Search Planner + Scraper + Extractor.
- Artigo de Sid Bharath, "Building a Deep Research Agent with LangGraph And Exa" — tem um nó de reranking com Cohere e um nó "writer" que gera relatório executivo com citações. Praticamente o seu RAG Agent + Reranker + Briefing Agent pronto para adaptar.
- NVIDIA-AI-Blueprints/rag (GitHub oficial) — pipeline RAG de referência com NeMo Guardrails opcional antes do retrieval. Use como checklist do que sua pipeline RAG precisa ter.
- langchain-ai/open_deep_research — bom modelo de padrão supervisor com sub-agentes paralelos, caso queira paralelizar a coleta de múltiplas startups de uma vez.
- NVIDIA GenAI-Perf — ferramenta oficial de benchmarking de custo/latência de LLM; é a peça que você vai usar para o diferencial de viés econômico (seção 4.2), em vez de inventar números.

---

## 3. Base de conhecimento RAG — estrutura de conteúdo

Para programar rápido, ingestão não pode ser "jogar PDFs no Qdrant". Defina o template de cada entrada antes de escrever o ingestor:

```
KnowledgeEntry {
  id,
  tecnologia,
  categoria,
  problema_que_resolve,
  descricao_tecnica,
  descricao_negocio,
  complexidade_implementacao,
  sinais_de_gatilho: [],
  casos_de_uso_tipicos: [],
  fonte_url,
  data_ultima_verificacao
}
```

O campo `sinais_de_gatilho` é o que conecta o RAG ao motor de recomendação: em vez do Recommendation Agent "adivinhar" qual tecnologia citar, ele consulta o RAG filtrando por entradas cujos `sinais_de_gatilho` batem com os `sinais_ai_native`/`stack_concorrente_detectada` daquele perfil específico.

### Categorias de documentos a ingerir (ordem de prioridade)

1. Documentações oficiais NVIDIA já listadas no enunciado do case (NIM, NeMo, Triton, TensorRT-LLM, RAPIDS, Riva, Clara, Morpheus, AI Enterprise, Inception) — comece só por NIM, NeMo Guardrails, Triton e Inception, que cobrem a maioria dos casos do golden set (Passo 3); expanda depois.
2. Material conceitual "AI-native vs. wrapper" — Sequoia, Emergence (já no enunciado), mais artigos sobre o "AI Wrapper Problem" e critérios de investidores reais — isso vira a base textual que justifica o score_maturidade_ia.
3. Casos reais de startups (Passo 3 + seção 4.1) — cada entrada do golden set/banco histórico também entra no RAG como documento, para o agente conseguir citar "casos parecidos" no briefing.
4. Metodologia de benchmark/custo da NVIDIA (série "LLM Inference Benchmarking" do NVIDIA Technical Blog, incluindo "How Much Does Your LLM Inference Cost?") — base do diferencial de viés econômico (seção 4.2).
5. Benefícios concretos do programa Inception (créditos de cloud, preços preferenciais, VC Alliance) — para o Briefing Agent escrever a "próxima ação" com precisão, não genericamente.

---

## 4. Diferenciais definidos para o projeto

### 4.1 Banco histórico de sucessos e fracassos (case bank)

Por quê: classificar uma startup como valiosa ou de risco fica mais robusto se o agente puder comparar com casos reais parecidos — os que deram certo e os que não deram. Guardar só sucesso cria viés de sobrevivência; o relatório final fica mais honesto (qualitativo + quantitativo) incluindo os dois lados.

Schema de cada case:

```
HistoricalCase {
  empresa, tipo: "sucesso" | "fracasso" | "pivot" | "alerta",
  setor, ano,
  data_moat: bool, infra_propria: bool, dependencia_api_externa: bool, setor_regulado: bool,
  resumo_o_que_aconteceu,
  licao_estruturada,
  fonte_url
}
```

Seed inicial (pode entrar direto no banco para já ter massa crítica no dia 1):

| Empresa | Tipo | Setor | Características-chave | Lição |
|---|---|---|---|---|
| CloudWalk / InfinitePay | Sucesso | Fintech | IA + blockchain proprietário, foco em PMEs | Modelo proprietário + base de clientes nichada sustenta crescimento |
| QI Tech | Sucesso | Fintech | Unicórnio 2024, IA para crédito | IA como núcleo de decisão de negócio, não interface |
| Stark Bank | Sucesso | Fintech/Infra | Infra B2B com IA desde o início | Modelo "pick and shovel" tem moat de integração, não de UI |
| Laura | Sucesso | Healthtech | Dado proprietário de prontuário, detecção de sepse | Setor regulado + dado real-time é barreira de entrada alta |
| Noleak | Sucesso | Segurança | Stack NVIDIA nativa, P&D com universidades | Parceria acadêmica + infra própria é ICP ideal para Inception |
| Wuri | Fracasso | IA generativa genérica | Múltiplos pivots, "wrapper fino" | Pivot constante sem trava de mercado é sinal de alerta, não de agilidade |
| CodeWhisper | Fracasso | Dev tools | Diferencial replicado nativamente pela OpenAI | Se o provedor do modelo pode copiar sua feature, não é produto |
| Cydoc | Fracasso (após 7 anos) | Healthtech | Tecnologia validada, falhou em vendas/integração/modelo de negócio | Validação técnica é só uma fração do desafio em setor regulado |
| Jasper AI | Fracasso parcial (queda de valuation) | Conteúdo/Marketing | Cresceu rápido, caiu com o avanço nativo do ChatGPT | Crescimento sem moat é vulnerável à curva de melhoria dos labs |
| Builder.ai | Fracasso (fraude reportada) | Dev tools/no-code | Caso amplamente noticiado de "IA" operada majoritariamente por humanos | "IA" sem evidência técnica verificável é risco reputacional para quem recomenda |

Esse banco alimenta diretamente o campo casos_similares do StartupProfile (seção 1) — o Recommendation Agent busca por similaridade (setor + sinais em comum) antes de gerar a recomendação final, e o Briefing Agent pode citar "esse perfil se parece com X, que teve resultado Y porque Z".

### 4.2 Viés econômico com dados reais (não inventados)

Regra inegociável: nenhum número de economia/eficiência entra no briefing sem uma de duas origens — (a) um benchmark público e citável (ex.: a série "LLM Inference Benchmarking" do NVIDIA Technical Blog, que ensina a calcular TCO usando a ferramenta oficial GenAI-Perf, com métricas como TTFT, throughput e custo por token), ou (b) uma estimativa explicitamente rotulada como tal, com a fórmula e as premissas visíveis no relatório.

Como estruturar isso tecnicamente:

```
EstimativaEconomica {
  cenario_assumido,
  metodologia,
  economia_estimada_percentual,
  nivel_confianca: "medido" | "estimado_por_metodologia_publica" | "projecao_com_premissas",
  fonte_url
}
```

Nunca apresente nivel_confianca "projecao_com_premissas" como se fosse "medido" no texto do briefing — isso é o que separa um relatório crível de um chutômetro disfarçado de dado.

### 4.3 Loop de auto-avaliação (LLM-as-judge)

Um agente adicional roda depois do Recommendation Agent e antes do Briefing Agent: compara a recomendação gerada contra o golden set (seção 1, Passo 3) e o banco histórico (4.1) por similaridade, e sinaliza como "baixa confiança" qualquer recomendação que diverja do padrão esperado para aquele perfil — essas vão para revisão humana antes de sair no relatório.

### 4.4 Radar competitivo

O Extractor Agent já captura stack_concorrente_detectada (seção 1) ao identificar menções a Bedrock, Vertex AI, Azure OpenAI etc. em vagas, documentação técnica ou estudos de caso publicados pela própria startup. O Briefing Agent usa esse campo para decidir o enquadramento da "próxima ação": substituição (já usa concorrente, abordagem é migração) vs. complemento (ainda não tem stack definida, abordagem é adoção nativa).

### 4.5 Interface conversacional sobre a base já mapeada

Não exige infraestrutura nova: é uma camada de consulta em linguagem natural sobre o mesmo Postgres (filtros estruturados) + Qdrant (busca semântica) que já vai estar rodando. Perguntas como "quais startups de saúde usam LLM mas não têm guardrails" viram um filtro estruturado (setor=saúde + ausência de NeMo Guardrails em sinais_ai_native) combinado com busca semântica nas evidências.

### 4.6 Multilíngue

O Briefing Agent gera a versão canônica em pt-BR; uma segunda chamada (mesmo prompt, instrução de tradução técnica) gera a versão em inglês sob demanda — não precisa duplicar o pipeline inteiro, só o último nó.

---

## 5. Estratégias e estruturas sugeridas

- Schema-first: trave o StartupProfile e o KnowledgeEntry (seções 1 e 3) antes de escrever qualquer agente — é a causa mais comum de retrabalho em projetos LangGraph com muitos nós.
- Vertical slice antes de escala: rode 1 startup real (ex.: uma do golden set) por todo o pipeline ponta a ponta antes de tentar processar 20 de uma vez. Mais fácil de debugar.
- RAG orientado por avaliação: use o golden set (Passo 3) para medir se o reranking está realmente melhorando a recuperação, não só "parece melhor".
- Cache por hash de conteúdo: evite raspar/reprocessar a mesma startup repetidamente — chave de cache pode ser hash da URL + data de coleta.
- Checkpoint humano antes do envio: o briefing final passa por revisão humana opcional (principalmente os sinalizados como baixa confiança pelo LLM-as-judge, seção 4.3) antes de ir para o gerente da NVIDIA.
- Ingestão incremental da base RAG: comece pelas tecnologias que cobrem a maioria dos casos do golden set (NIM, NeMo Guardrails, Triton, Inception) e só expanda para Riva/Clara/Isaac/Morpheus quando aparecer um perfil real que precise delas.
- Scraping com fallback em camadas: trafilatura/Playwright primeiro (grátis), Firecrawl como fallback pago só quando o crawl simples falhar — controla custo sem perder cobertura.
