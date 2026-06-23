"""Seed knowledge base and retrieval helpers.

This module is intentionally in-memory for the MVP. The docs in this repo
recommend moving these entries to Qdrant/Postgres once the vertical slice is
validated.
"""

from __future__ import annotations

import re

from nvidia_startup_ai_radar.schemas import HistoricalCase, KnowledgeEntry


KNOWLEDGE_ENTRIES: list[KnowledgeEntry] = [
    KnowledgeEntry(
        id="nvidia-nim",
        tecnologia="NVIDIA NIM",
        categoria="LLM serving",
        problema_que_resolve="Deploy padronizado de modelos com controle de custo, latencia e portabilidade.",
        descricao_tecnica="Microservicos de inferencia otimizados, com API compativel com padroes populares.",
        descricao_negocio="Reduz dependencia de APIs externas e acelera a entrada em producao.",
        complexidade_implementacao="media",
        sinais_de_gatilho=[
            "api externa",
            "openai",
            "anthropic",
            "llm",
            "modelo proprietario",
            "self-hosted",
            "latencia",
            "custo por token",
        ],
        casos_de_uso_tipicos=["agentes", "chat corporativo", "copilots", "inferencia LLM"],
        fonte_url="https://build.nvidia.com/nim",
    ),
    KnowledgeEntry(
        id="nvidia-nemo-guardrails",
        tecnologia="NeMo Guardrails",
        categoria="AI governance",
        problema_que_resolve="Governanca, seguranca e controle de comportamento de agentes e LLMs.",
        descricao_tecnica="Camada de rails conversacionais, regras e validacoes para aplicacoes generativas.",
        descricao_negocio="Ajuda a vender IA em ambientes regulados com mais confianca e rastreabilidade.",
        complexidade_implementacao="media",
        sinais_de_gatilho=[
            "compliance",
            "lgpd",
            "bacen",
            "anvisa",
            "hipaa",
            "saude",
            "fintech",
            "guardrail",
            "governanca",
        ],
        casos_de_uso_tipicos=["healthtech", "fintech", "atendimento regulado", "agentes seguros"],
        fonte_url="https://github.com/NVIDIA/NeMo-Guardrails",
    ),
    KnowledgeEntry(
        id="nvidia-triton",
        tecnologia="Triton Inference Server",
        categoria="Inference serving",
        problema_que_resolve="Servir modelos em producao com batching, observabilidade e multiplos backends.",
        descricao_tecnica="Servidor de inferencia para modelos de deep learning em CPU/GPU com alto throughput.",
        descricao_negocio="Melhora uso de infraestrutura e confiabilidade operacional.",
        complexidade_implementacao="media",
        sinais_de_gatilho=[
            "gpu",
            "inferencia",
            "throughput",
            "mlops",
            "computer vision",
            "visao computacional",
            "tensorrt",
            "nvidia",
        ],
        casos_de_uso_tipicos=["visao computacional", "detecao em video", "LLM serving", "MLOps"],
        fonte_url="https://developer.nvidia.com/triton-inference-server",
    ),
    KnowledgeEntry(
        id="nvidia-tensorrt-llm",
        tecnologia="TensorRT-LLM",
        categoria="LLM optimization",
        problema_que_resolve="Otimizar inferencia de LLM para menor latencia e maior throughput.",
        descricao_tecnica="Biblioteca de otimizacao e runtime para modelos generativos em GPU NVIDIA.",
        descricao_negocio="Pode melhorar economia unitaria quando a startup tem volume relevante de inferencia.",
        complexidade_implementacao="alta",
        sinais_de_gatilho=[
            "llm",
            "latencia",
            "throughput",
            "gpu",
            "modelo proprietario",
            "fine-tuning",
            "custo",
        ],
        casos_de_uso_tipicos=["LLM em escala", "batching", "produtos AI-native"],
        fonte_url="https://github.com/NVIDIA/TensorRT-LLM",
    ),
    KnowledgeEntry(
        id="nvidia-rapids",
        tecnologia="RAPIDS",
        categoria="Data processing",
        problema_que_resolve="Acelerar pipelines tabulares e ML classico com GPU.",
        descricao_tecnica="Suite cuDF, cuML e bibliotecas para dataframes e machine learning acelerados.",
        descricao_negocio="Reduz tempo de processamento de dados e experimentacao em times de dados.",
        complexidade_implementacao="media",
        sinais_de_gatilho=["dados tabulares", "data engineer", "credito", "fraude", "analytics", "etl"],
        casos_de_uso_tipicos=["fintech", "risco de credito", "fraude", "analytics"],
        fonte_url="https://rapids.ai/",
    ),
    KnowledgeEntry(
        id="nvidia-riva",
        tecnologia="NVIDIA Riva",
        categoria="Speech AI",
        problema_que_resolve="ASR/TTS e voz em tempo real para produtos conversacionais.",
        descricao_tecnica="SDK para reconhecimento e sintese de fala acelerados por GPU.",
        descricao_negocio="Habilita automacao de atendimento, call center e interfaces de voz.",
        complexidade_implementacao="media",
        sinais_de_gatilho=["voz", "call center", "transcricao", "audio", "tts", "asr"],
        casos_de_uso_tipicos=["contact center", "assistentes de voz", "transcricao"],
        fonte_url="https://developer.nvidia.com/riva",
    ),
    KnowledgeEntry(
        id="nvidia-clara",
        tecnologia="NVIDIA Clara",
        categoria="Healthcare AI",
        problema_que_resolve="IA medica, imagem, life sciences e workflows clinicos.",
        descricao_tecnica="Plataforma e modelos para workloads de saude e ciencias da vida.",
        descricao_negocio="Aumenta fit tecnico em healthtechs reguladas com dados clinicos.",
        complexidade_implementacao="alta",
        sinais_de_gatilho=["saude", "healthtech", "clinico", "prontuario", "anvisa", "medico", "cancer"],
        casos_de_uso_tipicos=["imagem medica", "monitoramento hospitalar", "oncologia"],
        fonte_url="https://developer.nvidia.com/clara",
    ),
    KnowledgeEntry(
        id="nvidia-morpheus",
        tecnologia="NVIDIA Morpheus",
        categoria="Cybersecurity AI",
        problema_que_resolve="Pipelines acelerados de deteccao de ameacas e anomalias.",
        descricao_tecnica="Framework para ciberseguranca com IA e processamento acelerado.",
        descricao_negocio="Ajuda startups de seguranca a escalar deteccao com menor latencia.",
        complexidade_implementacao="alta",
        sinais_de_gatilho=["cybersecurity", "seguranca", "anomalia", "ameaca", "fraude"],
        casos_de_uso_tipicos=["SIEM", "deteccao de anomalias", "fraude"],
        fonte_url="https://developer.nvidia.com/morpheus-cybersecurity",
    ),
    KnowledgeEntry(
        id="nvidia-ai-enterprise",
        tecnologia="NVIDIA AI Enterprise",
        categoria="Enterprise platform",
        problema_que_resolve="Stack empresarial suportada para IA em producao.",
        descricao_tecnica="Suite de software empresarial NVIDIA com suporte e integracoes.",
        descricao_negocio="Reduz risco de implantacao para startups que vendem a enterprise.",
        complexidade_implementacao="media",
        sinais_de_gatilho=["enterprise", "on-premise", "compliance", "governanca", "producao"],
        casos_de_uso_tipicos=["enterprise AI", "setores regulados", "private cloud"],
        fonte_url="https://www.nvidia.com/en-us/data-center/products/ai-enterprise/",
    ),
    KnowledgeEntry(
        id="nvidia-inception",
        tecnologia="NVIDIA Inception",
        categoria="Startup program",
        problema_que_resolve="Acesso a ecossistema, credito, treinamento, go-to-market e VC Alliance.",
        descricao_tecnica="Programa gratuito para startups com beneficios tecnicos e comerciais.",
        descricao_negocio="Proxima acao natural para startups com fit NVIDIA ainda sem relacionamento.",
        complexidade_implementacao="baixa",
        sinais_de_gatilho=["startup", "funding", "vc", "gpu", "ai-native", "go-to-market"],
        casos_de_uso_tipicos=["introducao comercial", "ativacao de parceria", "creditos cloud"],
        fonte_url="https://www.nvidia.com/en-us/startups/",
    ),
]


HISTORICAL_CASES: list[HistoricalCase] = [
    HistoricalCase(
        empresa="CloudWalk / InfinitePay",
        tipo="sucesso",
        setor="Fintech",
        data_moat=True,
        infra_propria=True,
        setor_regulado=True,
        resumo_o_que_aconteceu="IA e blockchain proprietario para pagamentos de PMEs.",
        licao_estruturada="Modelo proprietario e base nichada sustentam crescimento.",
    ),
    HistoricalCase(
        empresa="QI Tech",
        tipo="sucesso",
        setor="Fintech",
        data_moat=True,
        setor_regulado=True,
        resumo_o_que_aconteceu="IA para credito e automacao financeira.",
        licao_estruturada="IA no nucleo da decisao de negocio tem maior defensibilidade.",
    ),
    HistoricalCase(
        empresa="Stark Bank",
        tipo="sucesso",
        setor="Fintech/Infra",
        data_moat=True,
        infra_propria=True,
        setor_regulado=True,
        resumo_o_que_aconteceu="Infra bancaria B2B com automacao por IA.",
        licao_estruturada="Moat de integracao pode ser mais forte que interface.",
    ),
    HistoricalCase(
        empresa="Laura",
        tipo="sucesso",
        setor="Healthtech",
        data_moat=True,
        setor_regulado=True,
        resumo_o_que_aconteceu="Monitoramento hospitalar e deteccao precoce de riscos clinicos.",
        licao_estruturada="Dado clinico em tempo real cria barreira alta.",
    ),
    HistoricalCase(
        empresa="Noleak",
        tipo="sucesso",
        setor="Seguranca",
        data_moat=True,
        infra_propria=True,
        resumo_o_que_aconteceu="Usa Metropolis, GPUs, TensorRT e Triton para video analytics.",
        licao_estruturada="Stack NVIDIA nativa e P&D academico indicam ICP ideal.",
    ),
    HistoricalCase(
        empresa="Wuri",
        tipo="fracasso",
        setor="IA generativa",
        dependencia_api_externa=True,
        resumo_o_que_aconteceu="Pivots sucessivos e wrappers de IA empresarial.",
        licao_estruturada="Pivot constante sem trava de mercado e sinal de alerta.",
    ),
    HistoricalCase(
        empresa="CodeWhisper",
        tipo="fracasso",
        setor="Dev tools",
        dependencia_api_externa=True,
        resumo_o_que_aconteceu="Feature principal foi replicada por provedor de modelo.",
        licao_estruturada="Se o provedor copia a feature, nao ha produto defensavel.",
    ),
    HistoricalCase(
        empresa="Cydoc",
        tipo="fracasso",
        setor="Healthtech",
        data_moat=True,
        setor_regulado=True,
        resumo_o_que_aconteceu="Tecnologia validada, mas falhou em vendas, integracao e modelo de negocio.",
        licao_estruturada="Em saude, tecnologia e apenas parte do desafio.",
    ),
    HistoricalCase(
        empresa="Jasper AI",
        tipo="alerta",
        setor="Conteudo/Marketing",
        dependencia_api_externa=True,
        resumo_o_que_aconteceu="Queda de valuation com avanco nativo do ChatGPT.",
        licao_estruturada="Crescimento sem moat e vulneravel a melhoria dos labs.",
    ),
    HistoricalCase(
        empresa="Builder.ai",
        tipo="alerta",
        setor="Dev tools/no-code",
        resumo_o_que_aconteceu="Caso reportado de IA operada majoritariamente por humanos.",
        licao_estruturada="IA sem evidencia tecnica verificavel e risco reputacional.",
    ),
]


def tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z0-9\-_]+", text.lower()) if len(token) > 2}


def retrieve_knowledge(query: str, limit: int = 6) -> list[KnowledgeEntry]:
    query_tokens = tokenize(query)
    scored: list[tuple[float, KnowledgeEntry]] = []
    for entry in KNOWLEDGE_ENTRIES:
        haystack = " ".join(
            [
                entry.tecnologia,
                entry.categoria,
                entry.problema_que_resolve,
                entry.descricao_tecnica,
                entry.descricao_negocio,
                " ".join(entry.sinais_de_gatilho),
                " ".join(entry.casos_de_uso_tipicos),
            ]
        )
        entry_tokens = tokenize(haystack)
        overlap = len(query_tokens & entry_tokens)
        trigger_hits = sum(1 for trigger in entry.sinais_de_gatilho if trigger.lower() in query.lower())
        score = overlap + (2.5 * trigger_hits)
        if score > 0:
            scored.append((score, entry))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [entry for _, entry in scored[:limit]]


def similar_cases(profile_text: str, limit: int = 4) -> list[HistoricalCase]:
    tokens = tokenize(profile_text)
    scored: list[tuple[float, HistoricalCase]] = []
    for case in HISTORICAL_CASES:
        haystack = " ".join(
            [
                case.empresa,
                case.setor,
                case.resumo_o_que_aconteceu,
                case.licao_estruturada,
                "data_moat" if case.data_moat else "",
                "infra_propria" if case.infra_propria else "",
                "api_externa" if case.dependencia_api_externa else "",
                "setor_regulado" if case.setor_regulado else "",
            ]
        )
        score = len(tokens & tokenize(haystack))
        if score > 0:
            scored.append((score, case))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [case for _, case in scored[:limit]]
