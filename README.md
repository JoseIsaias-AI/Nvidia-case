# NVIDIA Startup AI Radar

Implementacao em LangGraph do pipeline multiagente descrito nos documentos em
`docs/`. O objetivo e apoiar um gerente de Startups & VCs da NVIDIA a descobrir,
classificar e priorizar startups AI-native, com evidencias rastreaveis e
recomendacoes de tecnologias NVIDIA.

## Agentes implementados

1. `search_planner`: transforma uma consulta outbound ou perfil inbound em buscas
   e URLs-alvo.
2. `scraper`: coleta texto publico ou cria uma pagina local quando o modo offline
   esta ativo.
3. `extractor`: converte texto bruto em `StartupProfile`.
4. `startup_classifier`: calcula score de maturidade e classifica como
   `AI-native`, `AI-enabled`, `non-AI` ou `indeterminado`.
5. `evidence_validator`: remove sinais sem fonte/trecho e marca revisao humana
   quando necessario.
6. `nvidia_rag`: recupera entradas da base NVIDIA inicial.
7. `recommendation`: gera recomendacoes NVIDIA estruturadas.
8. `economic_estimator`: cria estimativa economica sem inventar numeros.
9. `llm_as_judge`: compara o perfil com o golden set/case bank.
10. `briefing`: escreve o briefing executivo em pt-BR.
11. `technical_translation`: gera versao em ingles quando solicitado.

Os prompts ficam em `src/nvidia_startup_ai_radar/prompts.py` e os contratos de
estado em `src/nvidia_startup_ai_radar/schemas.py`.

## Como rodar

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

Execute um fluxo offline:

```powershell
nvidia-radar --query "Noleak usa NVIDIA GPUs, TensorRT e Triton Inference Server para visao computacional em cameras de seguranca, com P&D em universidades."
```

Salve o `StartupProfile` estruturado e o briefing em SQLite:

```powershell
nvidia-radar --query "Noleak usa NVIDIA GPUs, TensorRT e Triton Inference Server para visao computacional em cameras de seguranca, com P&D em universidades." --save-profile
```

Liste execucoes persistidas:

```powershell
nvidia-radar --list-runs
```

Abra o dashboard local:

```powershell
nvidia-radar-dashboard
```

Ou diretamente com Streamlit:

```powershell
$env:PYTHONPATH="src"
streamlit run src\nvidia_startup_ai_radar\dashboard.py
```

Ou via modulo:

```powershell
$env:PYTHONPATH="src"
python -m nvidia_startup_ai_radar.cli --query "healthtech com IA para prontuario, LGPD e predicao clinica" --output-language both
```

Rodar testes:

```powershell
pytest -q
```

## Configuracao

Copie `.env.example` para `.env` ou exporte variaveis no shell.

- Sem `NVIDIA_API_KEY` ou `OPENAI_API_KEY`, o pipeline usa fallback
  deterministico local.
- Com `NVIDIA_API_KEY`, tenta usar `ChatNVIDIA`.
- Com `OPENAI_API_KEY`, usa `ChatOpenAI` como fallback.
- `RADAR_ENABLE_WEB_FETCH=true` habilita coleta HTTP com `requests` +
  `trafilatura`. O padrao e `false` para evitar scraping acidental.
- `--save-profile` persiste o resultado em `data/radar_profiles.sqlite` por
  padrao. Use `--profile-db caminho\arquivo.sqlite` para escolher outro local.

## Estrutura

```text
src/nvidia_startup_ai_radar/
  agents.py          # nos do grafo
  graph.py           # montagem LangGraph
  prompts.py         # prompts de todos os agentes
  schemas.py         # StartupProfile, evidencias, recomendacoes e estado
  knowledge_base.py  # seed de tecnologias NVIDIA e case bank
  storage.py         # persistencia SQLite local de perfis e briefings
  pipeline.py        # runner compartilhado por CLI e dashboard
  dashboard.py       # interface Streamlit para rodar e explorar analises
  scraping.py        # fetch publico MVP
  llm.py             # adaptador opcional de LLM
  cli.py             # entrada de linha de comando
```

## Proximos passos naturais

- Migrar a persistencia SQLite de `StartupProfile` para Postgres.
- Migrar `KNOWLEDGE_ENTRIES` e `HISTORICAL_CASES` para Qdrant + BM25.
- Adicionar Playwright/Firecrawl como fallback do `scraper`.
- Adicionar exportacao de briefing em PDF/slide a partir do dashboard.
