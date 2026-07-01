# Guia Completo Do Case NVIDIA Startup AI Radar

Este documento e o guia unico do projeto. Ele explica por que a plataforma
existe, como ela funciona, como usar cada parte da interface e como apresentar o
case de forma objetiva.

## 1. Motivacao

O desafio do case e criar uma plataforma para a NVIDIA identificar startups que
tenham uso real de IA e que possam se beneficiar do ecossistema NVIDIA. A ideia
nao e apenas listar empresas. A plataforma precisa descobrir candidatas, filtrar
ruido, entender a maturidade tecnica, comparar sinais, consultar uma base de
conhecimento NVIDIA e transformar isso em uma recomendacao acionavel.

Na pratica, a pergunta central e:

> Esta startup tem uma dor real de IA que a NVIDIA pode acelerar, proteger,
> reduzir custo ou tornar mais robusta?

Por isso o projeto combina discovery outbound, agentes, RAG, avaliacao,
interface web e briefing.

## 2. O Produto

O NVIDIA Startup AI Radar funciona como uma base operacional para times de
parcerias, vendas tecnicas, Inception e desenvolvimento de ecossistema. O fluxo
esperado e:

1. Encontrar candidatas em fontes publicas.
2. Remover paginas que nao sao startups.
3. Coletar evidencias publicas da empresa.
4. Montar um `StartupProfile` estruturado.
5. Classificar maturidade de IA.
6. Consultar o RAG NVIDIA.
7. Recomendar tecnologias NVIDIA com fonte.
8. Mostrar oportunidades por prioridade.
9. Exportar briefing para acao comercial.

O diferencial e que a recomendacao nao fica solta. Ela nasce de sinais
extraidos, evidencias, maturidade, stack concorrente e conhecimento NVIDIA
recuperado pelo RAG.

## 3. Como O Pipeline Funciona

O pipeline principal roda em agentes:

- `Search Planner`: transforma tema, query ou texto em plano de busca.
- `Scraper`: coleta paginas publicas com cache, robots.txt, requests,
  Playwright e Firecrawl opcional.
- `Extractor`: cria o `StartupProfile`, sinais fortes, riscos e stack
  concorrente.
- `Classifier`: classifica AI-native, AI-enabled, non-AI ou indeterminado.
- `Evidence Validator`: verifica se os sinais tem trecho e fonte.
- `NVIDIA RAG`: busca tecnologias, casos e argumentos em `data/raw_sources`.
- `Recommendation`: recomenda tecnologias NVIDIA com base nos chunks retornados.
- `Economic Estimator`: descreve onde medir custo, latencia e throughput.
- `Judge`: checa qualidade da recomendacao contra golden set e historico.
- `Briefing`: gera texto executivo e proxima acao.

Se `LLM_PROVIDER=groq` e `GROQ_API_KEY` estiverem configurados, os agentes de
raciocinio tentam usar LLM real. Se nao houver chave, cota ou rede, o sistema
usa fallback local e registra isso no trace.

## 4. Dados E Persistencia

O banco padrao e SQLite, isolado por uma interface de repositorio. Isso deixa o
projeto pronto para um futuro backend Postgres sem reescrever a logica de
negocio.

Arquivos importantes:

- `src/nvidia_startup_ai_radar/agents.py`: agentes do pipeline.
- `src/nvidia_startup_ai_radar/pipeline.py`: execucao principal.
- `src/nvidia_startup_ai_radar/storage.py`: repositorio SQLite.
- `src/nvidia_startup_ai_radar/rag.py`: busca, embeddings e reranking.
- `src/nvidia_startup_ai_radar/web_api.py`: API FastAPI.
- `frontend/src/App.tsx`: interface web.
- `data/raw_sources/`: corpus versionavel do RAG.
- `config/source_registry.json`: fontes governadas para discovery.
- `config/activation_playbooks.json`: roteiros de ativacao NVIDIA.
- `tests/fixtures/golden_set.json`: avaliacao esperada do case.

Arquivos gerados, como `.sqlite`, logs, cache, exports e builds, nao devem
entrar no commit.

## 5. Como Rodar

Instalacao:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
python -m playwright install chromium
```

Frontend:

```powershell
cd frontend
npm install
npm run build
cd ..
```

Modo local sem LLM:

```powershell
$env:LLM_PROVIDER="none"
$env:RADAR_ENABLE_WEB_FETCH="false"
nvidia-radar --query "Noleak usa IA para visao computacional em seguranca" --save-profile --json
```

Modo com Groq:

```powershell
$env:LLM_PROVIDER="groq"
$env:GROQ_API_KEY="sua-chave"
$env:GROQ_MODEL="llama-3.3-70b-versatile"
$env:GROQ_BASE_URL="https://api.groq.com/openai/v1"
nvidia-radar --query "healthtech com IA para predicao clinica e LGPD" --save-profile --json
```

Subir a interface:

```powershell
nvidia-radar-web
```

Abrir:

```text
http://127.0.0.1:8000
```

Se der erro de porta:

```powershell
$env:RADAR_WEB_PORT="8010"
nvidia-radar-web
```

## 6. Como Usar Cada Aba

### Radar

E a tela principal da carteira. Ela mostra quantas startups foram mapeadas,
graficos de maturidade, tecnologias recomendadas, crescimento da base e
distribuicao de score. Ao selecionar uma startup, o painel lateral mostra:

- classificacao;
- score;
- setor;
- origem;
- sinais fortes;
- sinais de risco;
- tecnologias recomendadas;
- briefing;
- atividade da execucao.

Use esta aba para responder: "quais startups ja estao no radar e qual e a
prioridade de cada uma?".

### Oportunidades

E a tela que conecta a startup ao valor para NVIDIA. Ela deve mostrar, por
startup:

- maturidade;
- dor principal;
- tecnologia NVIDIA recomendada;
- prioridade;
- stack concorrente detectada;
- proxima acao comercial;
- status de revisao.

Use esta aba para responder: "o que a NVIDIA deveria oferecer para cada
startup?".

### Nova Analise

Permite analisar uma startup individual. Cole um resumo, URL ou texto publico e
clique em analisar. A execucao passa pelo pipeline completo e salva o resultado
no SQLite.

Use esta aba quando voce ja tem uma empresa especifica em mente.

### Encontrar

Roda discovery outbound. A plataforma busca candidatas em fontes governadas,
filtra conteudo que nao e startup e mostra uma lista de empresas encontradas.
Depois voce pode adicionar uma ou todas ao Radar.

Use esta aba quando voce quer partir de um tema, setor ou campanha.

### Filtrar Base

Consulta a base salva em linguagem natural. Exemplos:

```text
quais healthtechs usam LLM mas nao tem guardrails
quais startups tem stack concorrente da AWS
quais fintechs parecem AI-native
```

O resultado deve ser uma lista filtrada de startups, nao uma resposta solta.

### Conhecimento

Busca diretamente no RAG NVIDIA com citacao de fonte. E util para provar que a
recomendacao vem de uma base tecnica e nao de texto inventado.

Acoes principais:

- buscar uma tecnologia ou dor;
- ver chunks recuperados;
- ingerir fontes;
- recriar indice.

### Atividade

Mostra observabilidade: agente, latencia, sucesso, erro e modo de execucao. Ela
serve para defender maturidade de engenharia e mostrar quando o sistema usou LLM
ou fallback.

### Setup

Mostra o estado do ambiente, fontes governadas e roteiros de ativacao. Tambem
serve como tela de explicacao rapida para jurados entenderem que a plataforma
tem governanca de fontes, base de conhecimento e fallback local.

## 7. Como Apresentar Em 4 Minutos

Comece no Radar:

"Esta e a base operacional criada para a NVIDIA encontrar startups com potencial
real de IA e transformar isso em recomendacoes tecnicas acionaveis. O objetivo
nao e apenas ranquear empresas, mas entender quais dores cada startup tem e qual
parte do ecossistema NVIDIA faz sentido para ela."

Mostre os graficos:

"A primeira tela mostra a carteira mapeada. Os graficos usam dados reais do
banco: classificacao por maturidade, tecnologias mais recomendadas, crescimento
da base e distribuicao de score. Isso mostra se a carteira esta madura ou cheia
de casos fracos."

Abra Oportunidades:

"Aqui esta o coracao comercial do produto. Para cada startup, a plataforma
mostra a dor detectada, a tecnologia NVIDIA recomendada, a prioridade e a
proxima acao. Se houver AWS Bedrock, Vertex AI ou Azure OpenAI, a abordagem muda
para migracao ou substituicao. Se nao houver stack concorrente, a abordagem e
adocao nativa ou complemento."

Abra Encontrar:

"A descoberta outbound busca candidatas em fontes governadas e filtra paginas
que nao sao startups, como Wikipedia, tutoriais e documentacao. Depois de
encontrar candidatas, eu posso adicionar ao Radar e o pipeline processa cada uma
com os mesmos criterios."

Abra Conhecimento:

"A recomendacao usa RAG com fontes NVIDIA. Aqui eu consigo buscar, por exemplo,
Triton, NIM ou NeMo Guardrails e ver os trechos usados como base. Isso e
importante porque a recomendacao precisa ser explicavel."

Abra Atividade:

"Cada execucao gera trace por agente. A tela mostra latencia, sucesso e se o
agente usou LLM ou fallback. Como estamos preparados para Groq, quando a chave e
cota estao disponiveis os agentes usam LLM; quando nao estao, o sistema continua
rodando localmente."

Feche:

"O resultado final e uma plataforma que descobre, qualifica e transforma
startups em oportunidades NVIDIA, com evidencias, recomendacao tecnica, RAG,
briefing e governanca operacional."

## 8. Diferenciais

- Discovery outbound integrado ao pipeline, nao apenas cadastro manual.
- Filtro contra conteudo que nao e startup.
- Schema `StartupProfile` consistente para toda execucao.
- RAG com corpus NVIDIA versionavel em `data/raw_sources`.
- Recomendacao vinculada a evidencias e chunks recuperados.
- Deteccao de stack concorrente para orientar abordagem comercial.
- Consulta em linguagem natural sobre a base salva.
- Revisao humana para casos com lacunas.
- Observabilidade por agente.
- Preparado para Groq, mas funcional offline.

## 9. Limitacoes Conhecidas

- A qualidade da descoberta depende das fontes publicas acessiveis.
- Algumas paginas bloqueiam scraping ou exigem JavaScript pesado.
- Sem Groq ou outro LLM, o fallback e robusto, mas menos rico.
- Os scores nao substituem due diligence humana.
- O SQLite serve bem para demo e avaliacao; escala de producao pediria Postgres,
  filas e jobs assicronos.
- A avaliacao de custo/latencia e metodologica; numeros reais exigem benchmark
  com dados da startup.

## 10. Golden Set

O golden set usado para avaliacao fica em `tests/fixtures/golden_set.json`. Ele
inclui casos positivos e casos de risco para testar se o pipeline classifica e
recomenda de forma coerente:

- positivos: CloudWalk/InfinitePay, QI Tech, Stark Bank, Laura, OncoAI, Noleak
  e Mr. Turing;
- risco ou fracasso: Wuri, CodeWhisper e Cydoc.

Use:

```powershell
nvidia-radar --eval-golden-set
```

O objetivo nao e esconder erro. O relatorio deve indicar quais casos ainda
precisam de ajuste de prompt, fonte ou regra de classificacao.

## 11. Checklist Antes Da Demo

```powershell
pytest -q
python -m compileall src tests
nvidia-radar --rag-rebuild
nvidia-radar --list-runs --limit 3
nvidia-radar-web
```

Se precisar popular rapidamente:

```powershell
nvidia-radar --query "Boosted.ai usa AWS Bedrock para analise de portfolio financeiro" --save-profile --json
nvidia-radar --query "Noleak usa IA para visao computacional em seguranca e precisa reduzir latencia de inferencia" --save-profile --json
```
