# Plano de execução — NVIDIA Startup AI Radar

---

## 1. Como pensar o problema antes de codar

Antes de abrir o editor, vale fechar três decisões de design — elas vão guiar todo o resto:

1. **O sistema é "outbound" (a NVIDIA procura startups) ou também "inbound" (startups se cadastram)?** O case pede outbound, mas um formulário simples de auto-cadastro é um diferencial barato de implementar (ver seção 6).
2. **Qual é a unidade de verdade do "perfil da startup"?** Defina o schema (nome, setor, produto, stack de IA detectada, founders, funding, fontes, score de maturidade) antes de escrever o primeiro scraper — isso evita retrabalho no Extractor Agent.
3. **Como você vai avaliar se a recomendação é boa?** Defina de 5 a 10 startups reais como "golden set" manualmente anotado (qual tecnologia NVIDIA faria sentido para cada uma). Isso vira seu conjunto de avaliação do RAG e do motor de recomendação.

---

## 2. Trilhas de conhecimento necessárias

### 2.1 Sistemas multiagente com LangGraph

**O que dominar:**
- Diferença entre "chain" (LangChain) e "graph com estado" (LangGraph): nós, edges condicionais, checkpoints, retry, human-in-the-loop.
- Padrão **Supervisor**: um agente orquestrador delega para agentes especializados e decide quando parar — é o padrão mais usado em produção e se encaixa bem na sua pipeline (Search Planner como supervisor inicial, por exemplo).
- Design de **schema de estado**: comece definindo um `TypedDict`/`Pydantic model` único para o estado compartilhado entre os 8 agentes do case (startup_query, raw_pages, structured_profile, classification, evidence, retrieved_chunks, recommendation, briefing). Evita o erro mais comum de projetos LangGraph: estado em "dict solto" que vira impossível de refatorar depois de 10 nós.
- **Persistência/checkpointing**: comece com checkpoint em memória para prototipar rápido, mas planeje migrar para um backend em banco (Postgres/SQLite) antes da entrega final — é o que permite retomar uma execução que falhou no meio do scraping.

**Materiais:**
- Documentação oficial: https://langchain-ai.github.io/langgraph/ (comece pelos tutoriais de "multi-agent supervisor" e "persistence")
- Anthropic — *Building Effective Agents*: https://www.anthropic.com/research/building-effective-agents (ótimo para decidir quando usar agentes vs. workflows determinísticos — relevante porque nem todo nó do seu pipeline precisa ser um "agente" com LLM; alguns podem ser funções determinísticas)
- Curso curto (gratuito) DeepLearning.AI: "AI Agents in LangGraph" — busque por esse nome na plataforma deeplearning.ai
- LangSmith Studio (visual debugger dos grafos) — útil para depurar por que um agente está em loop

### 2.2 Web scraping e coleta de dados públicos

**O que dominar:**
- Diferença entre scraping de páginas estáticas (BeautifulSoup/trafilatura) e dinâmicas (Playwright).
- Extração de "conteúdo principal" de uma página (remover menu, rodapé, anúncios) — é justamente o que `trafilatura` e `Firecrawl` resolvem.
- Boas práticas: respeitar `robots.txt`, rate limiting, identificação no user-agent, e — fundamental para esse case — **registrar a fonte e a data de coleta de cada dado**, já que o Evidence Validator Agent depende disso.

**Materiais:**
- Playwright Python docs: https://playwright.dev/python/
- trafilatura docs: https://trafilatura.readthedocs.io/
- Firecrawl docs: https://docs.firecrawl.dev/
- Scrapy docs (só se for crawlear em maior escala, ex.: vários diretórios de startups): https://docs.scrapy.org/

### 2.3 RAG com busca híbrida e reranking

**O que dominar:**
- Pipeline completo: ingestão → chunking semântico → embeddings → vector DB → busca híbrida (vetorial + BM25) → reranking → geração com citação → avaliação.
- **Chunking por unidade semântica** (página/seção coerente) tende a funcionar melhor que chunking por tamanho fixo de caracteres, especialmente para documentação técnica como a da NVIDIA.
- **Reranking**: depois de buscar (ex.: top 20-50 candidatos via busca híbrida), um cross-encoder reordena por relevância real à pergunta. Isso normalmente traz ganhos visíveis de qualidade (a ordem da busca vetorial é só uma aproximação).
- **Avaliação de RAG**: defina métricas como precisão@5 ou NDCG@5 usando seu golden set, não confie só em "parece bom".

**Tecnologias para a etapa de reranking** (o case sugere Cohere Rerank, mas vale conhecer alternativas, já que esse mercado muda rápido):
- Cohere Rerank — opção paga, multilíngue, boa para PT-BR, fácil de integrar via API.
- Modelos open-source via `sentence-transformers` (ex.: BGE-Reranker, mxbai-rerank) — rodam localmente, sem custo de API, boa opção se o orçamento for zero.
- `FlashRank` — biblioteca leve em Python para reranking local, baixa fricção para prototipar rápido antes de decidir se vale pagar por uma API.

**Materiais:**
- Qdrant docs (filtros, busca híbrida): https://qdrant.tech/documentation/
- Cohere Rerank docs: https://docs.cohere.com/docs/reranking
- RAGAS (framework de avaliação de RAG): https://docs.ragas.io/
- Pinecone Learning Center (conceitos gerais de RAG, mesmo se você não usar o Pinecone como banco): https://www.pinecone.io/learn/

### 2.4 Stack NVIDIA

Como o objetivo final é recomendar tecnologias NVIDIA, vale entender cada uma o suficiente para escrever boas justificativas técnicas — você não precisa rodar todas elas, mas precisa saber **quando cada uma resolve qual dor**.

| Dor da startup | Tecnologia NVIDIA | Por quê |
|---|---|---|
| Depende 100% de API externa (OpenAI/Anthropic) para LLM | NIM + Triton | Deploy de modelo otimizado, controle de custo/latência |
| Quer trocar de modelo sem reescrever a aplicação | NIM (API padronizada estilo OpenAI) | Portabilidade entre modelos |
| Precisa de guardrails / governança de agentes | NeMo Guardrails | Controle de comportamento e compliance |
| Inferência lenta ou cara em produção | TensorRT-LLM + Triton | Otimização e batching de inferência |
| Processa grandes volumes de dados tabulares | RAPIDS (cuDF, cuML) | Pipelines de dados acelerados por GPU |
| Voz, call center, transcrição | NVIDIA Riva | ASR/TTS pronto para produção |
| Saúde / life sciences | NVIDIA Clara | Modelos e ferramentas de domínio médico |
| Robótica / simulação | Isaac + Omniverse | Simulação física e digital twins |
| Cybersecurity com IA | Morpheus | Pipelines de detecção acelerados |
| Quer entrar no ecossistema de startups | NVIDIA Inception | Créditos de cloud (parceria com AWS Activate, até US$ 100k em créditos), preços preferenciais, treinamentos gratuitos, acesso à comunidade de VCs — gratuito, sem equity, para empresas com até 10 anos |

**Materiais (use as documentações oficiais já listadas no case como base; complemente com):**
- NVIDIA Deep Learning Institute (DLI) — cursos curtos e gratuitos sobre NIM e IA generativa: https://www.nvidia.com/en-us/training/
- Canal "NVIDIA Developer" no YouTube — bom para ver demos rápidas de NIM, Triton e RAPIDS
- NVIDIA AI Blueprints (pipelines de referência prontos, úteis como inspiração de arquitetura): https://build.nvidia.com/blueprints

### 2.5 Frontend / dashboard

O case deixa livre, então escolha pelo que te dá velocidade:
- **Streamlit** ou **Gradio**: mais rápido para um MVP funcional focado em dados (tabelas, filtros, exportação de briefing em PDF).
- **Next.js + shadcn/ui**: se quiser um produto com cara mais "comercial" para apresentar no final.
- Evite gastar tempo demais aqui — o enunciado é explícito que o foco de avaliação é a arquitetura de IA, não o frontend.

---

## 3. Stack tecnológica recomendada (resumo prático)

| Camada | Escolha sugerida | Alternativa |
|---|---|---|
| Orquestração de agentes | LangGraph | CrewAI, AutoGen |
| Scraping dinâmico | Playwright | Selenium |
| Extração de texto limpo | trafilatura / Firecrawl | newspaper3k |
| Banco estruturado | PostgreSQL | SQLite (prototipagem) |
| Banco vetorial | Qdrant (Docker local, fácil de rodar) | ChromaDB, pgvector |
| Busca lexical | BM25 (`rank_bm25` em Python, ou nativo no Qdrant/Elasticsearch) | Elasticsearch |
| Reranking | Cohere Rerank (se houver orçamento) | BGE-Reranker local via `sentence-transformers` |
| LLM de orquestração/agentes | Modelo via API (Claude/GPT) durante o desenvolvimento | NVIDIA NIM endpoint, para "dogfooding" do produto final |
| Observabilidade | LangSmith ou Langfuse (open-source, self-host) | — |
| Frontend | Streamlit | Next.js |

---

## 4. Plano de execução em 4 semanas

### Semana 1 — Fundação + coleta
- Estudar LangGraph (supervisor pattern) e desenhar o schema de estado compartilhado.
- Implementar Search Planner Agent + Scraper Agent (Playwright + trafilatura) para 1 fonte só (ex.: Distrito ou um diretório de notícias).
- Subir Qdrant local via Docker e testar ingestão de 2-3 documentos da base NVIDIA.
- **Entregável da semana**: pipeline de scraping rodando ponta a ponta para uma única consulta, com dados salvos em JSON/Postgres.

### Semana 2 — Estruturação + classificação
- Extractor Agent: transformar HTML/texto bruto em schema estruturado (usar LLM com saída em JSON validado via Pydantic).
- Startup Classifier Agent + Evidence Validator Agent.
- Expandir scraping para 3-5 fontes diferentes do case.
- **Entregável da semana**: banco com pelo menos 15-20 startups reais classificadas (AI-native / AI-enabled / non-AI), com fontes registradas.

### Semana 3 — RAG + motor de recomendação
- Ingestão completa da base de conhecimento NVIDIA (documentações oficiais + materiais do case).
- Implementar busca híbrida + reranking + geração com citação.
- Recommendation Agent: cruzar perfil da startup com a tabela de gaps (seção 2.4) consultando o RAG.
- Criar o golden set de avaliação (5-10 casos anotados manualmente) e medir precisão das recomendações.
- **Entregável da semana**: RAG funcional com reranking + motor de recomendação gerando output estruturado (tecnologia, justificativa técnica, justificativa de negócio, prioridade, complexidade, próxima ação, evidências).

### Semana 4 — Briefing, interface e polimento
- Briefing Agent: gerar relatório executivo a partir da recomendação.
- Interface web (dashboard) conectando tudo.
- Implementar pelo menos 1-2 diferenciais da seção 6.
- Testes de ponta a ponta, ajuste de prompts, documentação do repositório (README com arquitetura, decisões técnicas e como rodar).
- **Entregável da semana**: sistema completo, documentado, com pelo menos um vídeo curto ou GIF demonstrando o fluxo.

> Lembre-se: o enunciado pede commits constantes ao longo do mês, não uma entrega única no fim — distribua esse cronograma em commits incrementais reais.

---

## 5. Cuidados e riscos a não ignorar

- **Legal/ético no scraping**: respeite `robots.txt` e termos de uso. Para fontes que bloqueiam scraping agressivo (ex.: LinkedIn), prefira buscar informação pública indexada (Google/Bing) em vez de raspar diretamente o site.
- **Alucinação no Extractor/Recommendation Agent**: sempre exija que o LLM cite a fonte/trecho usado. O Evidence Validator Agent deve rejeitar afirmações sem fonte rastreável — isso é central para a credibilidade do briefing.
- **Viés de amostra**: startups com mais presença digital (mais conteúdo público) tendem a ser super-representadas. Vale mencionar essa limitação no relatório final.
- **Custo de API**: chamadas de LLM em cada nó do grafo somam rápido. Use cache de resultados de scraping/extração para não reprocessar a mesma startup, e considere usar um modelo mais barato para extração estruturada (tarefa mais mecânica) e um mais forte só para classificação/recomendação (tarefa mais analítica).

---

## 6. Ideias para ir além (diferencial competitivo)

Algumas dessas são rápidas de implementar e têm alto impacto na apresentação; outras são mais ambiciosas. Priorize 1-2 que conversem melhor com o que você já está construindo.

**Rápidas de implementar:**
- **"AI maturity score" explicável**: em vez de só classificar AI-native/AI-enabled/non-AI, gere uma nota de 0-100 com a justificativa de cada componente (ex.: "+20 pontos: usa modelo proprietário fine-tunado", "-10 pontos: depende só de wrapper de API"). Fica mais acionável para o time comercial.
- **Sinal de "intenção de contratação"**: cruzar vagas abertas da startup (página de carreiras) buscando termos como "ML engineer", "GPU", "inference" como proxy de maturidade técnica crescente — informação que normalmente não está no site institucional.
- **Exportação do briefing em PDF/slide**: usar a skill de geração de documentos para o Briefing Agent já entregar um PDF pronto para o gerente de Inception enviar por e-mail.
- **Dogfooding NVIDIA**: usar o próprio NVIDIA Riva (TTS) para gerar um áudio do briefing executivo — é um detalhe que mostra domínio real da stack que você está recomendando, não só citando da documentação.

**Médio esforço:**
- **Benchmark de custo/latência simulado**: dado o uso estimado de tokens/requisições da startup, simular quanto ela economizaria migrando de API externa para NIM self-hosted. Números concretos vendem melhor que recomendação genérica.
- **Funil inbound**: landing page simples onde a própria startup preenche um formulário e recebe na hora um diagnóstico inicial — inverte o fluxo (de outbound para também inbound) e é um ângulo que praticamente nenhum outro grupo vai propor.
- **Loop de auto-avaliação (LLM-as-judge)**: um agente extra que audita a saída do Recommendation Agent contra o golden set antes de liberar o briefing, sinalizando recomendações de baixa confiança para revisão humana.
- **Observabilidade do pipeline**: dashboard de execuções (LangSmith/Langfuse) mostrando quantas startups foram processadas, taxa de erro por agente, custo por execução — sinaliza maturidade de engenharia, não só de produto.

**Mais ambiciosas:**
- **Radar competitivo**: detectar e sinalizar quando a startup já usa stack de concorrentes (Bedrock, Vertex AI, Azure OpenAI) — útil para o time de vendas calibrar a abordagem ("substituição" vs. "complemento").
- **Interface conversacional**: permitir que o gerente da NVIDIA "converse" com a base de startups já mapeadas (“quais startups de saúde usam LLM mas não têm guardrails?”) em vez de só consultar uma por vez.
- **Multilíngue**: versão em inglês do briefing, para o caso de o achado interessar a outras unidades da NVIDIA fora do Brasil.

---

## 7. Checklist final de habilidades a dominar

- [ ] Modelar estado compartilhado e grafo com transições condicionais no LangGraph
- [ ] Implementar pelo menos um agente supervisor com delegação a agentes especializados
- [ ] Fazer scraping respeitando robots.txt e registrando proveniência de cada dado
- [ ] Construir pipeline RAG completo: chunking → embeddings → busca híbrida → reranking → citação
- [ ] Avaliar qualidade de retrieval com métrica objetiva (NDCG@k ou precisão@k) sobre um golden set
- [ ] Validar saída estruturada de LLM com Pydantic/JSON schema
- [ ] Justificar tecnicamente, para cada tecnologia NVIDIA do case, quando ela é a recomendação certa
- [ ] Documentar decisões de arquitetura no README do repositório

---

*Este plano é um roteiro, não uma camisa de força — ajuste o cronograma conforme o ritmo real do seu grupo, mas mantenha a sequência lógica: coleta → estruturação → conhecimento (RAG) → recomendação → interface.*
