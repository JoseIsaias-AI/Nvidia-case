const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

export type StartupRun = {
  run_id: number;
  created_at?: string;
  nome?: string;
  site?: string | null;
  setor?: string | null;
  subsetor?: string | null;
  classificacao?: string;
  score_maturidade_ia?: number;
  score_wrapper_risco?: number;
  origem?: string;
  needs_human_review?: boolean;
  human_review_required?: boolean;
  review_status?: "pendente" | "aprovado" | "rejeitado";
  review_nota?: string | null;
  judge_status?: string | null;
  judge_motivos?: string[];
  profile?: StartupProfile;
  briefing_pt?: string;
  briefing_en?: string;
};

export type NvidiaNeed = {
  need: string;
  technologies?: string[];
  evidence_terms?: string[];
  confidence?: number;
};

export type OpportunityItem = {
  run_id: number;
  name?: string;
  sector?: string;
  classification?: string;
  maturity_score?: number;
  opportunity_score?: number;
  action_type?: string;
  decision_bucket?: string;
  review_status?: string;
  needs?: NvidiaNeed[];
  recommendations?: Recommendation[];
  recommended_stack?: string[];
  primary_recommendation?: Recommendation | null;
  next_action?: string;
  competitor_stack?: string[];
  risk_flags?: string[];
  evidence_summary?: string;
  source_urls?: string[];
  case_matches?: { case_id?: string; tipo?: string; licao?: string }[];
  evidence_count?: number;
};

export type OpportunityMatrix = {
  summary: {
    startup_count: number;
    ready_count: number;
    competitor_count: number;
    technology_count: number;
    top_technology?: string | null;
    priority_count?: number;
    migration_count?: number;
    validation_count?: number;
  };
  decision_counts?: { bucket: string; count: number }[];
  technology_counts: { technology: string; count: number }[];
  need_counts: { need: string; count: number }[];
  items: OpportunityItem[];
};

export type StartupProfile = {
  nome?: string;
  site?: string | null;
  setor?: string | null;
  subsetor?: string | null;
  produto_descricao?: string | null;
  publico_alvo?: string | null;
  classificacao?: string;
  explicacao_classificacao?: string | null;
  score_maturidade_ia?: number;
  score_wrapper_risco?: number;
  sinais_ai_native?: SignalEvidence[];
  sinais_wrapper_risco?: SignalEvidence[];
  stack_tecnica_detectada?: string[];
  stack_concorrente_detectada?: string[];
  stack_concorrente_evidencias?: SignalEvidence[];
  evidencias?: Evidence[];
  recomendacoes_nvidia?: Recommendation[];
  estimativa_economica?: Record<string, unknown>;
  casos_similares?: Record<string, unknown>[];
};

export type SignalEvidence = {
  sinal?: string;
  evidencia_trecho?: string;
  fonte_url?: string;
  data_coleta?: string;
};

export type Evidence = {
  trecho_resumido?: string;
  fonte_url?: string;
  data_coleta?: string;
};

export type Recommendation = {
  tecnologia?: string;
  prioridade?: "alta" | "media" | "baixa";
  complexidade?: string;
  justificativa_tecnica?: string;
  justificativa_negocio?: string;
  proxima_acao?: string;
  evidencias?: Evidence[];
};

export type Overview = {
  record_count: number;
  minimum_required: number;
  empty_message: string;
  classification_by_sector: {
    ready: boolean;
    classification_counts: { name: string; count: number; color: string }[];
    sector_segments: {
      sector: string;
      total: number;
      segments: { classification: string; count: number; share: number; color: string }[];
    }[];
  };
  technology_ranking: {
    ready: boolean;
    items: { technology: string; count: number; predominant_priority: string; color: string }[];
  };
  mapping_evolution: {
    ready: boolean;
    points: { period: string; count: number; total: number; grouping: string }[];
  };
  maturity_distribution: {
    ready: boolean;
    bins: { range: string; count: number; color: string }[];
  };
};

export type ProfileRadar = {
  run_id: number;
  startup_name?: string;
  reference_count: number;
  reference_available: boolean;
  empty_reference_message: string;
  axes: { axis: string; startup: number; reference: number | null }[];
};

export type Quality = {
  claim_count: number;
  supported_evidence_count: number;
  unsupported_claim_count: number;
  unsupported_claim_rate: number;
  critical_unsupported_claim_count: number;
  evidence_coverage: number;
  export_ready: boolean;
  status: string;
};

export type Readiness = {
  ready: boolean;
  checks: { id: string; label: string; status: string; detail: string }[];
  settings: Record<string, string>;
};

export type TimelineItem = {
  step?: string;
  started_at?: string;
  ended_at?: string;
  latency_ms?: number;
  avg_latency_ms?: number;
  success_rate?: number;
  count?: number;
  success?: boolean;
  mode?: string;
  units?: number | null;
  estimated_cost_usd?: number | null;
  error?: string | null;
  [key: string]: unknown;
};

export type DiscoveryItem = {
  nome?: string;
  name?: string;
  url?: string;
  source_domain?: string;
  source_type?: string;
  title?: string;
  snippet?: string;
  evidence_excerpt?: string;
  quality_tier?: string;
  score?: number;
  recommended_action?: string;
  analysis_query?: string;
  nvidia_signals?: string[];
  ai_framework_signals?: string[];
  competitor_stack_signals?: string[];
  maturity_signals?: string[];
  wrapper_risk_signals?: string[];
  motivo?: string;
  snippets?: string[];
};

export type DiscoveryAnalyzeResponse = {
  requested_count?: number;
  saved_count: number;
  failed_count: number;
  saved: { run_id: number; candidate?: string; profile_name?: string }[];
  failed: { candidate?: string; error?: string }[];
};

export type KnowledgeHit = {
  tecnologia?: string;
  title?: string;
  source_id?: string;
  fonte_url?: string;
  score?: number;
  text?: string;
  conteudo?: string;
  [key: string]: unknown;
};

export type BaseAnswer = {
  filters?: Record<string, unknown>;
  items?: StartupRun[];
  results?: StartupRun[];
};

export async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Erro HTTP ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export function downloadUrl(path: string): string {
  return `${API_BASE_URL}${path}`;
}
