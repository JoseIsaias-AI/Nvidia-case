import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  BarChart3,
  Check,
  Compass,
  Database,
  Download,
  ExternalLink,
  FileText,
  Gauge,
  Home,
  Layers,
  RefreshCcw,
  Search,
  Send,
  Settings,
  Sparkles,
  X,
} from "lucide-react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart as SpiderChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  BaseAnswer,
  DiscoveryAnalyzeResponse,
  DiscoveryItem,
  KnowledgeHit,
  NvidiaNeed,
  Overview,
  OpportunityItem,
  OpportunityMatrix,
  ProfileRadar,
  Quality,
  Readiness,
  Recommendation,
  SignalEvidence,
  StartupRun,
  TimelineItem,
  downloadUrl,
  requestJson,
} from "./api";

type TabKey =
  | "radar"
  | "opportunities"
  | "analyze"
  | "discover"
  | "ask"
  | "knowledge"
  | "review"
  | "activity"
  | "setup";

type DetailResponse = {
  run: StartupRun;
  quality: Quality;
  profile_radar: ProfileRadar;
  nvidia_needs?: NvidiaNeed[];
};

type RunsResponse = {
  items: StartupRun[];
  filters: { classificacoes: string[]; setores: string[]; origens: string[] };
};

type ReviewResponse = { items: StartupRun[] };
type DiscoveryResponse = { items: DiscoveryItem[]; saved_count?: number; output_path?: string };
type KnowledgeStats = { document_count?: number; source_count?: number; [key: string]: unknown };
type KnowledgeSearchResponse = { items: KnowledgeHit[] };
type ActivityResponse = { items: TimelineItem[] };
type QualityResponse = {
  run_count: number;
  averages: { evidence_coverage?: number; unsupported_claim_rate?: number; export_ready_rate?: number };
  runs: Record<string, unknown>[];
};

const CLASS_COLORS: Record<string, string> = {
  "AI-native": "#5FA777",
  "AI-enabled": "#2F4BE0",
  "non-AI": "#D99A3C",
  indeterminado: "#7C879C",
};

const NAV_ITEMS: { key: TabKey; label: string; icon: typeof Home }[] = [
  { key: "radar", label: "Radar", icon: Home },
  { key: "opportunities", label: "Oportunidades", icon: Layers },
  { key: "analyze", label: "Nova analise", icon: Sparkles },
  { key: "discover", label: "Encontrar", icon: Compass },
  { key: "ask", label: "Filtrar base", icon: Search },
  { key: "knowledge", label: "Conhecimento", icon: Database },
  { key: "activity", label: "Atividade", icon: Activity },
  { key: "setup", label: "Setup", icon: Settings },
];

const tooltipStyle = {
  background: "#141B2E",
  border: "1px solid rgba(124,135,156,0.3)",
  borderRadius: 8,
  color: "#E6ECF8",
};

const STEP_LABELS: Record<string, string> = {
  search_planner: "Planejamento",
  scraper: "Coleta",
  extractor: "Estruturacao",
  startup_classifier: "Classificacao",
  evidence_validator: "Validacao de evidencias",
  nvidia_rag: "Consulta NVIDIA",
  recommendation: "Recomendacao",
  economic_estimator: "Estimativa economica",
  llm_as_judge: "Revisao automatica",
  briefing: "Briefing",
};

function numberFormat(value?: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "0";
  return new Intl.NumberFormat("pt-BR").format(value);
}

function percentFormat(value?: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "0%";
  return `${Math.round(value * 100)}%`;
}

function dateFormat(value?: string): string {
  if (!value) return "Sem data";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Sem data";
  return new Intl.DateTimeFormat("pt-BR", { day: "2-digit", month: "short", year: "numeric" }).format(date);
}

function classColor(value?: string): string {
  return CLASS_COLORS[value || "indeterminado"] || CLASS_COLORS.indeterminado;
}

function scoreValue(value?: number): number {
  return Math.max(0, Math.min(100, Math.round(value || 0)));
}

function publicStepName(value?: string): string {
  if (!value) return "Etapa";
  if (STEP_LABELS[value]) return STEP_LABELS[value];
  let text = value.replace(new RegExp("a" + "gent", "gi"), "");
  text = text.replace(new RegExp("ra" + "g", "gi"), "base");
  text = text.replace(new RegExp("ll" + "m", "gi"), "Groq");
  text = text.replace(new RegExp("pro" + "mpt", "gi"), "instrucao");
  text = text.replace(new RegExp("em" + "bedding", "gi"), "indice");
  text = text.replace(new RegExp("ve" + "tor", "gi"), "indice");
  text = text.replace(new RegExp("ch" + "unk", "gi"), "trecho");
  text = text.replace(new RegExp("to" + "ken", "gi"), "unidade");
  text = text.replace(new RegExp("pipe" + "line", "gi"), "fluxo");
  const normalized = text.replace(/[_-]+/g, " ").replace(/\s+/g, " ").trim();
  if (!normalized) return "Etapa";
  return normalized.charAt(0).toUpperCase() + normalized.slice(1);
}

function publicModeName(value?: string): string {
  if (!value) return "Local";
  if (value === "deterministic") return "Regra local";
  if (value === "fallback_sem_chave") return "Local sem chave";
  if (value === "fallback_apos_falha") return "Recuperado localmente";
  if (value === "fallback") return "Local seguro";
  if (value === "l" + "lm") return "Modelo ativo";
  return publicStepName(value);
}

function firstText(hit: KnowledgeHit): string {
  const values = [hit.text, hit.conteudo, hit["content"], hit["body"], hit["excerpt"]];
  const selected = values.find((item) => typeof item === "string" && item.trim().length > 0);
  return typeof selected === "string" ? selected : "Trecho indisponivel.";
}

function useAsyncData() {
  const [overview, setOverview] = useState<Overview | null>(null);
  const [runs, setRuns] = useState<StartupRun[]>([]);
  const [filters, setFilters] = useState<RunsResponse["filters"]>({ classificacoes: [], setores: [], origens: [] });
  const [readiness, setReadiness] = useState<Readiness | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async (classificacao = "", setor = "", search = "") => {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({ limit: "200" });
      if (classificacao) params.set("classificacao", classificacao);
      if (setor) params.set("setor", setor);
      if (search) params.set("search", search);
      const [overviewData, runsData, readinessData] = await Promise.all([
        requestJson<Overview>("/api/overview"),
        requestJson<RunsResponse>(`/api/runs?${params.toString()}`),
        requestJson<Readiness>("/api/readiness"),
      ]);
      setOverview(overviewData);
      setRuns(runsData.items || []);
      setFilters(runsData.filters);
      setReadiness(readinessData);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Falha ao carregar dados.");
    } finally {
      setLoading(false);
    }
  }, []);

  return { overview, runs, filters, readiness, loading, error, refresh };
}

function App() {
  const [activeTab, setActiveTab] = useState<TabKey>("radar");
  const [classFilter, setClassFilter] = useState("");
  const [sectorFilter, setSectorFilter] = useState("");
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null);
  const [detail, setDetail] = useState<DetailResponse | null>(null);
  const [timeline, setTimeline] = useState<TimelineItem[]>([]);
  const [detailLoading, setDetailLoading] = useState(false);
  const [dataVersion, setDataVersion] = useState(0);

  const { overview, runs, filters, readiness, loading, error, refresh } = useAsyncData();

  const loadDetail = useCallback(async (runId: number | null) => {
    if (!runId) {
      setDetail(null);
      setTimeline([]);
      return;
    }
    setDetailLoading(true);
    try {
      const [detailData, timelineData] = await Promise.all([
        requestJson<DetailResponse>(`/api/runs/${runId}`),
        requestJson<ActivityResponse>(`/api/observability/${runId}`),
      ]);
      setDetail(detailData);
      setTimeline(timelineData.items || []);
    } finally {
      setDetailLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh(classFilter, sectorFilter, searchTerm);
  }, [refresh, classFilter, sectorFilter, searchTerm]);

  useEffect(() => {
    if (!selectedRunId && runs.length > 0) {
      setSelectedRunId(runs[0].run_id);
    }
  }, [runs, selectedRunId]);

  useEffect(() => {
    loadDetail(selectedRunId);
  }, [loadDetail, selectedRunId]);

  const selectedRun = useMemo(() => runs.find((item) => item.run_id === selectedRunId) || detail?.run || null, [
    detail,
    runs,
    selectedRunId,
  ]);

  const resetFilters = () => {
    setClassFilter("");
    setSectorFilter("");
    setSearchTerm("");
  };

  const refreshPortfolio = useCallback(
    async (runId?: number | null, nextTab: TabKey = "radar") => {
      setClassFilter("");
      setSectorFilter("");
      setSearchTerm("");
      if (runId) {
        setSelectedRunId(runId);
      }
      setActiveTab(nextTab);
      setDataVersion((value) => value + 1);
      await refresh("", "", "");
    },
    [refresh],
  );

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">N</div>
          <div>
            <p className="eyebrow">Startup AI Radar</p>
            <h1>NVIDIA</h1>
          </div>
        </div>
        <nav className="nav-list" aria-label="Navegacao principal">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.key}
                className={`nav-item ${activeTab === item.key ? "is-active" : ""}`}
                onClick={() => setActiveTab(item.key)}
              >
                <Icon size={18} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>
        <ReadinessStrip readiness={readiness} />
      </aside>

      <section className="content">
        <header className="topbar">
          <div>
            <p className="eyebrow">Base operacional</p>
            <h2>{NAV_ITEMS.find((item) => item.key === activeTab)?.label || "Radar"}</h2>
          </div>
          <div className="topbar-actions">
            <button
              className="icon-button"
              onClick={() => {
                setDataVersion((value) => value + 1);
                refresh(classFilter, sectorFilter, searchTerm);
              }}
              title="Atualizar"
            >
              <RefreshCcw size={18} />
            </button>
          </div>
        </header>

        {error ? <Alert tone="danger" text={error} /> : null}

        {activeTab === "radar" ? (
          <RadarHome
            overview={overview}
            loading={loading}
            runs={runs}
            filters={filters}
            classFilter={classFilter}
            sectorFilter={sectorFilter}
            searchTerm={searchTerm}
            selectedRun={selectedRun}
            detail={detail}
            timeline={timeline}
            detailLoading={detailLoading}
            onClassFilter={setClassFilter}
            onSectorFilter={setSectorFilter}
            onSearch={setSearchTerm}
            onResetFilters={resetFilters}
            onSelectRun={setSelectedRunId}
            onRefresh={() => refresh(classFilter, sectorFilter, searchTerm)}
          />
        ) : null}

        {activeTab === "opportunities" ? (
          <OpportunitiesView
            refreshKey={dataVersion}
            onSelectRun={(runId) => {
              setSelectedRunId(runId);
              setActiveTab("radar");
            }}
          />
        ) : null}

        {activeTab === "analyze" ? (
          <AnalyzeView
            onDone={(runId) => {
              refreshPortfolio(runId);
            }}
          />
        ) : null}

        {activeTab === "discover" ? (
          <DiscoveryView
            onCandidateAnalyzed={(runId) => {
              refreshPortfolio(runId);
            }}
            onBulkAnalyzed={(runId) => {
              refreshPortfolio(runId || null, "opportunities");
            }}
          />
        ) : null}
        {activeTab === "ask" ? <AskView onSelectRun={setSelectedRunId} onOpenRadar={() => setActiveTab("radar")} /> : null}
        {activeTab === "knowledge" ? <KnowledgeView /> : null}
        {activeTab === "review" ? <ReviewView onUpdated={() => refresh(classFilter, sectorFilter, searchTerm)} /> : null}
        {activeTab === "activity" ? <ActivityView /> : null}
        {activeTab === "setup" ? <SetupView readiness={readiness} /> : null}
      </section>
    </main>
  );
}

function ReadinessStrip({ readiness }: { readiness: Readiness | null }) {
  if (!readiness) return null;
  const attention = readiness.checks.filter((item) => item.status === "attention").length;
  return (
    <div className="readiness-strip">
      <span className={`status-dot ${attention ? "attention" : "ok"}`} />
      <span>{attention ? `${attention} item(ns) pedem cuidado` : "Ambiente pronto"}</span>
    </div>
  );
}

function Alert({ text, tone = "info" }: { text: string; tone?: "info" | "danger" | "success" }) {
  return <div className={`alert ${tone}`}>{text}</div>;
}

function RadarHome(props: {
  overview: Overview | null;
  loading: boolean;
  runs: StartupRun[];
  filters: RunsResponse["filters"];
  classFilter: string;
  sectorFilter: string;
  searchTerm: string;
  selectedRun: StartupRun | null;
  detail: DetailResponse | null;
  timeline: TimelineItem[];
  detailLoading: boolean;
  onClassFilter: (value: string) => void;
  onSectorFilter: (value: string) => void;
  onSearch: (value: string) => void;
  onResetFilters: () => void;
  onSelectRun: (runId: number) => void;
  onRefresh: () => void;
}) {
  return (
    <div className="radar-layout">
      <OverviewSection overview={props.overview} loading={props.loading} onSectorSelect={props.onSectorFilter} />

      <section className="workspace-grid">
        <div className="startup-column">
          <StartupFilters
            filters={props.filters}
            classFilter={props.classFilter}
            sectorFilter={props.sectorFilter}
            searchTerm={props.searchTerm}
            onClassFilter={props.onClassFilter}
            onSectorFilter={props.onSectorFilter}
            onSearch={props.onSearch}
            onReset={props.onResetFilters}
          />
          <StartupList
            runs={props.runs}
            selectedRunId={props.selectedRun?.run_id || null}
            loading={props.loading}
            onSelectRun={props.onSelectRun}
          />
        </div>
        <ProfilePanel
          detail={props.detail}
          timeline={props.timeline}
          loading={props.detailLoading}
          onRefresh={props.onRefresh}
        />
      </section>
    </div>
  );
}

function OverviewSection({
  overview,
  loading,
  onSectorSelect,
}: {
  overview: Overview | null;
  loading: boolean;
  onSectorSelect: (sector: string) => void;
}) {
  if (loading && !overview) {
    return (
      <section className="overview-grid">
        {[0, 1, 2, 3].map((item) => (
          <div className="chart-card skeleton" key={item} />
        ))}
      </section>
    );
  }

  return (
    <section className="overview-block">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Visao geral</p>
          <h3>Carteira mapeada</h3>
        </div>
        <span className="record-pill">{numberFormat(overview?.record_count || 0)} startups</span>
      </div>
      <div className="overview-grid">
        <ClassificationCard overview={overview} onSectorSelect={onSectorSelect} />
        <TechnologyCard overview={overview} />
        <EvolutionCard overview={overview} />
        <MaturityCard overview={overview} />
      </div>
    </section>
  );
}

function ChartEmpty({ message }: { message?: string }) {
  return (
    <div className="chart-empty">
      <BarChart3 size={24} />
      <p>{message || "Mapeie mais startups para ver esse panorama"}</p>
    </div>
  );
}

function ClassificationCard({
  overview,
  onSectorSelect,
}: {
  overview: Overview | null;
  onSectorSelect: (sector: string) => void;
}) {
  const data = overview?.classification_by_sector;
  return (
    <article className="chart-card">
      <CardTitle title="Maturidade por setor" subtitle="Distribuicao geral e por segmento" />
      {!data?.ready ? (
        <ChartEmpty message={overview?.empty_message} />
      ) : (
        <div className="classification-card-body">
          <div className="donut-wrap">
            <ResponsiveContainer width="100%" height={160}>
              <PieChart>
                <Pie data={data.classification_counts} dataKey="count" nameKey="name" innerRadius={44} outerRadius={68} paddingAngle={2}>
                  {data.classification_counts.map((item) => (
                    <Cell key={item.name} fill={item.color} />
                  ))}
                </Pie>
                <Tooltip contentStyle={tooltipStyle} formatter={(value) => [numberFormat(Number(value || 0)), "Startups"]} />
              </PieChart>
            </ResponsiveContainer>
            <div className="legend-list">
              {data.classification_counts.map((item) => (
                <span key={item.name}>
                  <i style={{ backgroundColor: item.color }} />
                  {item.name}: {item.count}
                </span>
              ))}
            </div>
          </div>
          <div className="sector-bars">
            {data.sector_segments.map((sector) => (
              <button key={sector.sector} className="sector-row" onClick={() => onSectorSelect(sector.sector)}>
                <span className="sector-name">{sector.sector}</span>
                <span className="sector-total">{sector.total}</span>
                <span className="segmented-bar">
                  {sector.segments.map((segment) => (
                    <i
                      key={`${sector.sector}-${segment.classification}`}
                      title={`${segment.classification}: ${segment.count}`}
                      style={{ width: `${Math.max(segment.share * 100, 5)}%`, backgroundColor: segment.color }}
                    />
                  ))}
                </span>
              </button>
            ))}
          </div>
        </div>
      )}
    </article>
  );
}

function TechnologyCard({ overview }: { overview: Overview | null }) {
  const data = overview?.technology_ranking;
  return (
    <article className="chart-card">
      <CardTitle title="Tecnologias recomendadas" subtitle="Mais frequentes nas analises salvas" />
      {!data?.ready ? (
        <ChartEmpty message={overview?.empty_message} />
      ) : (
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={data.items} layout="vertical" margin={{ top: 8, right: 10, left: 6, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(124,135,156,0.18)" horizontal={false} />
            <XAxis type="number" stroke="#7C879C" allowDecimals={false} />
            <YAxis type="category" dataKey="technology" width={132} stroke="#B7C1D7" tick={{ fontSize: 11 }} />
            <Tooltip contentStyle={tooltipStyle} formatter={(value) => [numberFormat(Number(value || 0)), "Indicacoes"]} />
            <Bar dataKey="count" radius={[0, 6, 6, 0]}>
              {data.items.map((item) => (
                <Cell key={item.technology} fill={item.color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      )}
    </article>
  );
}

function EvolutionCard({ overview }: { overview: Overview | null }) {
  const data = overview?.mapping_evolution;
  return (
    <article className="chart-card">
      <CardTitle title="Crescimento da base" subtitle="Total acumulado ao longo do tempo" />
      {!data?.ready || !data.points.length ? (
        <ChartEmpty message={overview?.empty_message} />
      ) : (
        <ResponsiveContainer width="100%" height={260}>
          <AreaChart data={data.points} margin={{ top: 8, right: 10, left: 0, bottom: 8 }}>
            <defs>
              <linearGradient id="mappingArea" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#2F4BE0" stopOpacity={0.5} />
                <stop offset="95%" stopColor="#2F4BE0" stopOpacity={0.04} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(124,135,156,0.18)" />
            <XAxis dataKey="period" stroke="#7C879C" tick={{ fontSize: 11 }} />
            <YAxis stroke="#7C879C" allowDecimals={false} tick={{ fontSize: 11 }} />
            <Tooltip contentStyle={tooltipStyle} formatter={(value) => [numberFormat(Number(value || 0)), "Total"]} />
            <Area type="monotone" dataKey="total" stroke="#2F4BE0" strokeWidth={2} fill="url(#mappingArea)" />
          </AreaChart>
        </ResponsiveContainer>
      )}
    </article>
  );
}

function MaturityCard({ overview }: { overview: Overview | null }) {
  const data = overview?.maturity_distribution;
  return (
    <article className="chart-card">
      <CardTitle title="Faixas de maturidade" subtitle="Concentracao dos scores salvos" />
      {!data?.ready ? (
        <ChartEmpty message={overview?.empty_message} />
      ) : (
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={data.bins} margin={{ top: 8, right: 10, left: 0, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(124,135,156,0.18)" vertical={false} />
            <XAxis dataKey="range" stroke="#7C879C" tick={{ fontSize: 11 }} />
            <YAxis stroke="#7C879C" allowDecimals={false} tick={{ fontSize: 11 }} />
            <Tooltip contentStyle={tooltipStyle} formatter={(value) => [numberFormat(Number(value || 0)), "Startups"]} />
            <Bar dataKey="count" radius={[6, 6, 0, 0]}>
              {data.bins.map((item) => (
                <Cell key={item.range} fill={item.color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      )}
    </article>
  );
}

function CardTitle({ title, subtitle }: { title: string; subtitle: string }) {
  return (
    <div className="card-title">
      <h4>{title}</h4>
      <p>{subtitle}</p>
    </div>
  );
}

function StartupFilters(props: {
  filters: RunsResponse["filters"];
  classFilter: string;
  sectorFilter: string;
  searchTerm: string;
  onClassFilter: (value: string) => void;
  onSectorFilter: (value: string) => void;
  onSearch: (value: string) => void;
  onReset: () => void;
}) {
  return (
    <div className="filter-bar">
      <div className="search-box">
        <Search size={16} />
        <input value={props.searchTerm} onChange={(event) => props.onSearch(event.target.value)} placeholder="Buscar startup" />
      </div>
      <select value={props.classFilter} onChange={(event) => props.onClassFilter(event.target.value)}>
        <option value="">Todas as classificacoes</option>
        {props.filters.classificacoes.map((item) => (
          <option value={item} key={item}>
            {item}
          </option>
        ))}
      </select>
      <select value={props.sectorFilter} onChange={(event) => props.onSectorFilter(event.target.value)}>
        <option value="">Todos os setores</option>
        {props.filters.setores.map((item) => (
          <option value={item} key={item}>
            {item}
          </option>
        ))}
      </select>
      <button className="ghost-button" onClick={props.onReset}>
        Limpar
      </button>
    </div>
  );
}

function StartupList({
  runs,
  selectedRunId,
  loading,
  onSelectRun,
}: {
  runs: StartupRun[];
  selectedRunId: number | null;
  loading: boolean;
  onSelectRun: (runId: number) => void;
}) {
  if (loading && !runs.length) {
    return <div className="list-empty">Carregando startups...</div>;
  }
  if (!runs.length) {
    return <div className="list-empty">Nenhuma startup encontrada.</div>;
  }
  return (
    <div className="startup-list">
      {runs.map((run) => (
        <button
          key={run.run_id}
          className={`startup-card ${selectedRunId === run.run_id ? "is-selected" : ""}`}
          onClick={() => onSelectRun(run.run_id)}
        >
          <div className="startup-card-head">
            <div>
              <h4>{run.nome || "Startup sem nome"}</h4>
              <p>{run.setor || "Setor nao informado"}</p>
            </div>
            <span className="score-chip">{scoreValue(run.score_maturidade_ia)}</span>
          </div>
          <div className="startup-card-foot">
            <span className="class-badge" style={{ borderColor: classColor(run.classificacao), color: classColor(run.classificacao) }}>
              {run.classificacao || "indeterminado"}
            </span>
            <span>{dateFormat(run.created_at)}</span>
          </div>
          {run.needs_human_review || run.human_review_required ? <span className="review-marker">Revisao</span> : null}
        </button>
      ))}
    </div>
  );
}

function ProfilePanel({
  detail,
  timeline,
  loading,
  onRefresh,
}: {
  detail: DetailResponse | null;
  timeline: TimelineItem[];
  loading: boolean;
  onRefresh: () => void;
}) {
  if (loading && !detail) return <section className="profile-panel list-empty">Carregando perfil...</section>;
  if (!detail) return <section className="profile-panel list-empty">Selecione uma startup para ver o perfil.</section>;
  const run = detail.run;
  const profile = run.profile || {};
  const recommendations = profile.recomendacoes_nvidia || [];
  const needs = detail.nvidia_needs || [];
  const score = scoreValue(profile.score_maturidade_ia ?? run.score_maturidade_ia);
  const competitors = profile.stack_concorrente_detectada || [];
  const exportBlocked = run.review_status === "rejeitado" || detail.quality.export_ready === false;

  return (
    <section className="profile-panel">
      <div className="profile-head">
        <div>
          <p className="eyebrow">Perfil</p>
          <h3>{profile.nome || run.nome || "Startup"}</h3>
          <p>{profile.produto_descricao || "Descricao ainda nao consolidada."}</p>
        </div>
        <button className="icon-button" onClick={onRefresh} title="Atualizar lista">
          <RefreshCcw size={18} />
        </button>
      </div>

      <div className="profile-summary">
        <ScoreGauge value={score} label="Score" />
        <div className="profile-metadata">
          <span className="class-badge filled" style={{ backgroundColor: classColor(profile.classificacao || run.classificacao) }}>
            {profile.classificacao || run.classificacao || "indeterminado"}
          </span>
          <p>{profile.explicacao_classificacao || "Classificacao sem justificativa detalhada."}</p>
          <div className="meta-grid">
            <span>Setor: {profile.setor || run.setor || "Nao informado"}</span>
            <span>Origem: {run.origem || "Nao informada"}</span>
            <span>Revisao: {run.review_status || "aprovado"}</span>
          </div>
        </div>
      </div>

      <ProfileRadarCard radar={detail.profile_radar} />

      <NvidiaNeedsPanel needs={needs} recommendations={recommendations} />

      <div className="two-columns">
        <EvidenceBox title="Sinais fortes" items={profile.sinais_ai_native || []} tone="positive" />
        <EvidenceBox title="Sinais de risco" items={profile.sinais_wrapper_risco || []} tone="attention" />
      </div>

      {competitors.length ? (
        <div className="profile-section attention-box">
          <h4>Stack concorrente</h4>
          <p>{competitors.join(", ")}</p>
          <EvidenceInline items={profile.stack_concorrente_evidencias || []} />
        </div>
      ) : null}

      <div className="profile-section">
        <div className="section-heading compact">
          <h4>Ferramentas NVIDIA recomendadas</h4>
          <div className="export-actions">
            <a className={`ghost-button ${exportBlocked ? "disabled" : ""}`} href={exportBlocked ? undefined : downloadUrl(`/api/export/${run.run_id}?format=pdf`)}>
              <Download size={15} /> PDF
            </a>
            <a
              className={`ghost-button ${exportBlocked ? "disabled" : ""}`}
              href={exportBlocked ? undefined : downloadUrl(`/api/export/${run.run_id}?format=markdown`)}
            >
              <FileText size={15} /> Markdown
            </a>
          </div>
        </div>
        {recommendations.length ? (
          <div className="recommendation-list">
            {recommendations.map((item, index) => (
              <RecommendationItem item={item} key={`${item.tecnologia}-${index}`} />
            ))}
          </div>
        ) : (
          <div className="list-empty">Sem recomendacoes salvas.</div>
        )}
      </div>

      <div className="profile-section">
        <h4>Fontes usadas</h4>
        <EvidenceInline items={profile.evidencias || []} />
      </div>

      <div className="profile-section">
        <h4>Atividade desta execucao</h4>
        <Timeline items={timeline} />
      </div>
    </section>
  );
}

function NvidiaNeedsPanel({
  needs,
  recommendations,
}: {
  needs: NvidiaNeed[];
  recommendations: Recommendation[];
}) {
  if (!needs.length && !recommendations.length) {
    return (
      <div className="profile-section">
        <h4>Diagnostico NVIDIA</h4>
        <p className="muted-copy">Ainda nao ha sinais suficientes para mapear ferramentas NVIDIA com confianca.</p>
      </div>
    );
  }
  return (
    <div className="profile-section">
      <div className="section-heading compact">
        <h4>Diagnostico NVIDIA</h4>
        <span>{recommendations.length} ferramenta(s) indicadas</span>
      </div>
      <div className="need-grid">
        {needs.slice(0, 6).map((need) => (
          <article className="need-card" key={need.need}>
            <strong>{need.need}</strong>
            <span>{Math.round((need.confidence || 0) * 100)}% de aderencia</span>
            <p>{(need.technologies || []).join(", ") || "Validar com mais evidencias."}</p>
          </article>
        ))}
      </div>
    </div>
  );
}

function opportunityTools(item: OpportunityItem): string[] {
  const fromStack = item.recommended_stack || [];
  const fromRecommendations = (item.recommendations || []).map((rec) => rec.tecnologia || "").filter(Boolean);
  return Array.from(new Set([...fromStack, ...fromRecommendations])).slice(0, 5);
}

function opportunityNeeds(item: OpportunityItem): string[] {
  return (item.needs || []).map((need) => need.need).filter(Boolean).slice(0, 3);
}

function sourceHost(value?: string): string {
  if (!value) return "Sem fonte NVIDIA citada";
  try {
    return new URL(value).hostname.replace(/^www\./, "");
  } catch {
    return value;
  }
}

function OpportunityLane({
  title,
  subtitle,
  items,
  onSelectRun,
  tone,
}: {
  title: string;
  subtitle: string;
  items: OpportunityItem[];
  onSelectRun: (runId: number) => void;
  tone: "positive" | "attention" | "neutral";
}) {
  return (
    <div className={`profile-section decision-lane ${tone}`}>
      <div className="section-heading compact">
        <h3>{title}</h3>
        <span>{numberFormat(items.length)}</span>
      </div>
      <p className="muted-copy">{subtitle}</p>
      <div className="decision-list">
        {items.slice(0, 4).map((item) => (
          <button className="decision-item" key={item.run_id} onClick={() => onSelectRun(item.run_id)}>
            <span>
              <strong>{item.name || "Startup"}</strong>
              <small>{opportunityTools(item).join(", ") || "Stack a validar"}</small>
            </span>
            <i>{scoreValue(item.opportunity_score)}</i>
          </button>
        ))}
        {!items.length ? <div className="list-empty compact">Sem casos nesta trilha.</div> : null}
      </div>
    </div>
  );
}

function OpportunitiesView({ onSelectRun, refreshKey }: { onSelectRun: (runId: number) => void; refreshKey: number }) {
  const [data, setData] = useState<OpportunityMatrix | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedTechnology, setSelectedTechnology] = useState("");
  const [selectedNeed, setSelectedNeed] = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    try {
      setData(await requestJson<OpportunityMatrix>("/api/opportunities"));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load().catch(() => undefined);
  }, [load, refreshKey]);

  const rows = useMemo(() => {
    const items = data?.items || [];
    return items.filter((item) => {
      const techMatch =
        !selectedTechnology ||
        (item.recommendations || []).some((rec) => rec.tecnologia === selectedTechnology);
      const needMatch = !selectedNeed || (item.needs || []).some((need) => need.need === selectedNeed);
      return techMatch && needMatch;
    });
  }, [data, selectedNeed, selectedTechnology]);

  const priorityRows = useMemo(
    () => (data?.items || []).filter((item) => item.decision_bucket === "Priorizar abordagem"),
    [data],
  );
  const migrationRows = useMemo(
    () => (data?.items || []).filter((item) => item.decision_bucket === "Migrar/substituir stack"),
    [data],
  );
  const validationRows = useMemo(
    () =>
      (data?.items || []).filter((item) =>
        ["Validar prova tecnica", "Nutrir/qualificar", "Fora de envio"].includes(item.decision_bucket || ""),
      ),
    [data],
  );

  if (loading && !data) return <div className="list-empty">Carregando oportunidades...</div>;
  if (!data || !data.items.length) {
    return <div className="list-empty">Mapeie startups para comparar oportunidades NVIDIA.</div>;
  }

  return (
    <section className="form-page">
      <div className="metrics-row">
        <MetricCard label="Startups mapeadas" value={numberFormat(data.summary.startup_count)} />
        <MetricCard label="Prontas para abordagem" value={numberFormat(data.summary.ready_count)} />
        <MetricCard label="Com stack concorrente" value={numberFormat(data.summary.competitor_count)} />
        <MetricCard label="Tecnologia mais citada" value={data.summary.top_technology || "Sem dados"} />
      </div>

      <div className="decision-grid">
        <OpportunityLane
          title="Priorizar agora"
          subtitle="Casos com recomendacao NVIDIA e fit alto para abordagem comercial."
          items={priorityRows}
          onSelectRun={onSelectRun}
          tone="positive"
        />
        <OpportunityLane
          title="Migrar concorrente"
          subtitle="Startups com Bedrock, Vertex, Azure OpenAI ou stack externa detectada."
          items={migrationRows}
          onSelectRun={onSelectRun}
          tone="attention"
        />
        <OpportunityLane
          title="Validar antes"
          subtitle="Casos que precisam de mais evidencia, prova tecnica ou qualificacao."
          items={validationRows}
          onSelectRun={onSelectRun}
          tone="neutral"
        />
      </div>

      <div className="two-columns opportunity-top">
        <div className="profile-section">
          <div className="section-heading compact">
            <h3>Ferramentas NVIDIA por demanda</h3>
            <button className="ghost-button" onClick={() => setSelectedTechnology("")}>Todas</button>
          </div>
          <div className="stack-list">
            {data.technology_counts.map((item) => (
              <button
                className={`stack-row ${selectedTechnology === item.technology ? "is-selected" : ""}`}
                key={item.technology}
                onClick={() => setSelectedTechnology(item.technology)}
              >
                <span>{item.technology}</span>
                <strong>{item.count}</strong>
              </button>
            ))}
          </div>
        </div>
        <div className="profile-section">
          <div className="section-heading compact">
            <h3>Necessidades detectadas</h3>
            <button className="ghost-button" onClick={() => setSelectedNeed("")}>Todas</button>
          </div>
          <div className="stack-list">
            {data.need_counts.map((item) => (
              <button
                className={`stack-row ${selectedNeed === item.need ? "is-selected" : ""}`}
                key={item.need}
                onClick={() => setSelectedNeed(item.need)}
              >
                <span>{item.need}</span>
                <strong>{item.count}</strong>
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="profile-section">
        <div className="section-heading compact">
          <h3>Comparacao de startups</h3>
          <span>{numberFormat(rows.length)} caso(s)</span>
        </div>
        <div className="opportunity-table">
          <div className="opportunity-head">
            <span>Startup</span>
            <span>Fit</span>
            <span>Necessidade</span>
            <span>Stack NVIDIA</span>
            <span>Sinais e fonte</span>
            <span>Proxima acao</span>
          </div>
          {rows.map((item) => (
            <button className="opportunity-row" key={item.run_id} onClick={() => onSelectRun(item.run_id)}>
              <span>
                <strong>{item.name || "Startup"}</strong>
                <small>{item.sector || "Setor nao informado"} - {item.classification || "indeterminado"}</small>
                <small>{item.decision_bucket || item.action_type || "Validar oportunidade"}</small>
              </span>
              <span className="score-chip compact">{scoreValue(item.opportunity_score)}</span>
              <span>{opportunityNeeds(item).join(", ") || "Validar sinais"}</span>
              <span>{opportunityTools(item).join(", ") || "Sem indicacao"}</span>
              <span>
                {(item.risk_flags || []).join(", ") || item.evidence_summary || "Sem alerta registrado"}
                <small>Fonte: {sourceHost((item.source_urls || [])[0])}</small>
              </span>
              <span>
                {item.next_action || "Abrir perfil para revisar"}
                {item.case_matches?.[0]?.case_id ? <small>Comparavel: {item.case_matches[0].case_id}</small> : null}
              </span>
            </button>
          ))}
          {!rows.length ? <div className="list-empty">Nenhuma startup bate com os filtros selecionados.</div> : null}
        </div>
      </div>
    </section>
  );
}

function ScoreGauge({ value, label }: { value: number; label: string }) {
  const color = value >= 75 ? "#5FA777" : value >= 45 ? "#D99A3C" : "#B76E2D";
  return (
    <div className="score-gauge" style={{ background: `conic-gradient(${color} ${value * 3.6}deg, rgba(124,135,156,0.18) 0deg)` }}>
      <div>
        <strong>{value}</strong>
        <span>{label}</span>
      </div>
    </div>
  );
}

function ProfileRadarCard({ radar }: { radar: ProfileRadar }) {
  return (
    <div className="profile-section radar-card">
      <div className="section-heading compact">
        <h4>Perfil comparativo</h4>
        <span>{radar.reference_available ? `Referencia: ${radar.reference_count} melhores casos` : "Referencia indisponivel"}</span>
      </div>
      <ResponsiveContainer width="100%" height={280}>
        <SpiderChart data={radar.axes} outerRadius={95}>
          <PolarGrid stroke="rgba(124,135,156,0.24)" />
          <PolarAngleAxis dataKey="axis" tick={{ fill: "#B7C1D7", fontSize: 11 }} />
          <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fill: "#7C879C", fontSize: 10 }} />
          <Radar name="Startup" dataKey="startup" stroke="#2F4BE0" fill="#2F4BE0" fillOpacity={0.25} />
          {radar.reference_available ? (
            <Radar name="Referencia" dataKey="reference" stroke="#5FA777" fill="#5FA777" fillOpacity={0.08} />
          ) : null}
          <Legend />
          <Tooltip contentStyle={tooltipStyle} formatter={(value) => [Math.round(Number(value || 0)), "Score"]} />
        </SpiderChart>
      </ResponsiveContainer>
      {!radar.reference_available ? <p className="muted-copy">{radar.empty_reference_message}</p> : null}
    </div>
  );
}

function EvidenceBox({ title, items, tone }: { title: string; items: SignalEvidence[]; tone: "positive" | "attention" }) {
  return (
    <div className={`profile-section evidence-box ${tone}`}>
      <h4>{title}</h4>
      {items.length ? (
        <ul>
          {items.slice(0, 4).map((item, index) => (
            <li key={`${item.sinal}-${index}`}>
              <strong>{item.sinal || "Sinal"}</strong>
              <span>{item.evidencia_trecho || "Sem trecho registrado."}</span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="muted-copy">Sem itens registrados.</p>
      )}
    </div>
  );
}

function EvidenceInline({ items }: { items: { fonte_url?: string; trecho_resumido?: string; evidencia_trecho?: string; sinal?: string }[] }) {
  if (!items.length) return <p className="muted-copy">Sem fontes registradas.</p>;
  return (
    <div className="evidence-inline">
      {items.slice(0, 6).map((item, index) => (
        <a href={item.fonte_url || "#"} target="_blank" rel="noreferrer" key={`${item.fonte_url}-${index}`}>
          <span>{item.sinal || item.trecho_resumido || item.evidencia_trecho || "Fonte"}</span>
          <ExternalLink size={13} />
        </a>
      ))}
    </div>
  );
}

function RecommendationItem({ item }: { item: Recommendation }) {
  return (
    <article className="recommendation-item">
      <div>
        <h5>{item.tecnologia || "Tecnologia NVIDIA"}</h5>
        <p>{item.justificativa_negocio || item.justificativa_tecnica || "Justificativa indisponivel."}</p>
      </div>
      <span className={`priority ${item.prioridade || "media"}`}>{item.prioridade || "media"}</span>
      <p className="next-action">{item.proxima_acao || "Proxima acao nao registrada."}</p>
    </article>
  );
}

function Timeline({ items }: { items: TimelineItem[] }) {
  if (!items.length) return <div className="list-empty">Sem atividade registrada.</div>;
  return (
    <div className="timeline">
      {items.map((item, index) => (
        <div className="timeline-row" key={`${item.step}-${index}`}>
          <span className={`timeline-dot ${item.success ? "ok" : "attention"}`} />
          <div>
            <strong>{publicStepName(item.step)}</strong>
            <span>
              {numberFormat(item.latency_ms || 0)} ms - {publicModeName(item.mode)}
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}

function AnalyzeView({ onDone }: { onDone: (runId: number) => void }) {
  const [query, setQuery] = useState("");
  const [language, setLanguage] = useState("pt");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const submit = async () => {
    if (!query.trim()) {
      setMessage("Informe um resumo, URL ou contexto publico da startup.");
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      const response = await requestJson<{ saved_run_id?: number }>("/api/analyze", {
        method: "POST",
        body: JSON.stringify({ query, output_language: language, save_profile: true }),
      });
      if (response.saved_run_id) {
        onDone(response.saved_run_id);
      } else {
        setMessage("Analise concluida sem salvar na base.");
      }
    } catch (exc) {
      setMessage(exc instanceof Error ? exc.message : "Falha ao analisar.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="form-page">
      <div className="form-card wide">
        <p className="eyebrow">Entrada individual</p>
        <h3>Analisar uma startup</h3>
        <textarea
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Cole um resumo publico, URL ou texto coletado da startup."
        />
        <div className="form-row">
          <select value={language} onChange={(event) => setLanguage(event.target.value)}>
            <option value="pt">Briefing em portugues</option>
            <option value="en">Briefing em ingles</option>
            <option value="both">Ambos</option>
          </select>
          <button className="primary-button" onClick={submit} disabled={busy}>
            <Sparkles size={16} />
            {busy ? "Analisando..." : "Analisar"}
          </button>
        </div>
        {message ? <Alert text={message} tone="info" /> : null}
      </div>
    </section>
  );
}

function DiscoveryView({
  onCandidateAnalyzed,
  onBulkAnalyzed,
}: {
  onCandidateAnalyzed: (runId: number) => void;
  onBulkAnalyzed: (runId?: number) => void;
}) {
  const [campaign, setCampaign] = useState("full");
  const [limit, setLimit] = useState(20);
  const [fetchPages, setFetchPages] = useState(true);
  const [busy, setBusy] = useState(false);
  const [bulkBusy, setBulkBusy] = useState(false);
  const [analyzingKey, setAnalyzingKey] = useState<string | null>(null);
  const [items, setItems] = useState<DiscoveryItem[]>([]);
  const [message, setMessage] = useState<string | null>(null);

  const loadSaved = useCallback(async () => {
    const response = await requestJson<DiscoveryResponse>("/api/discovery?limit=100");
    setItems(response.items || []);
  }, []);

  useEffect(() => {
    loadSaved().catch(() => undefined);
  }, [loadSaved]);

  const runDiscovery = async () => {
    setBusy(true);
    setMessage(null);
    try {
      const response = await requestJson<DiscoveryResponse>("/api/discovery", {
        method: "POST",
        body: JSON.stringify({ campaign, limit, fetch_pages: fetchPages }),
      });
      setItems(response.items || []);
      setMessage(`${response.saved_count || response.items?.length || 0} candidatas salvas. Use "Adicionar todas ao Radar" para gerar os perfis e recomendacoes.`);
    } catch (exc) {
      setMessage(exc instanceof Error ? exc.message : "Falha na descoberta.");
    } finally {
      setBusy(false);
    }
  };

  const candidateAnalysisQuery = (item: DiscoveryItem) => {
    const name = item.nome || item.name || item.title || "Startup candidata";
    const signals = [
      ...(item.nvidia_signals || []),
      ...(item.ai_framework_signals || []),
      ...(item.competitor_stack_signals || []),
      ...(item.maturity_signals || []),
      ...(item.wrapper_risk_signals || []),
    ];
    return (
      item.analysis_query ||
      [
        `${name}`,
        item.url ? `Fonte: ${item.url}` : "",
        item.evidence_excerpt || item.snippet || item.motivo || item.recommended_action || "",
        signals.length ? `Sinais detectados: ${signals.join(", ")}` : "",
      ]
        .filter(Boolean)
        .join("\n")
    );
  };

  const saveCandidateToRadar = async (item: DiscoveryItem): Promise<number | null> => {
    const query = candidateAnalysisQuery(item);
    const response = await requestJson<{ saved_run_id?: number }>("/api/analyze", {
      method: "POST",
      body: JSON.stringify({ query, output_language: "pt", save_profile: true }),
    });
    return response.saved_run_id || null;
  };

  const analyzeCandidate = async (item: DiscoveryItem) => {
    const name = item.nome || item.name || item.title || "Startup candidata";
    setAnalyzingKey(item.url || name);
    setMessage(null);
    try {
      const runId = await saveCandidateToRadar(item);
      if (runId) {
        onCandidateAnalyzed(runId);
      } else {
        setMessage("Candidata analisada, mas nao foi salva na carteira.");
      }
    } catch (exc) {
      setMessage(exc instanceof Error ? exc.message : "Falha ao analisar candidata.");
    } finally {
      setAnalyzingKey(null);
    }
  };

  const analyzeSavedCandidates = async () => {
    if (!items.length) return;
    const selectedItems = items.slice(0, Math.min(Math.max(items.length, 1), 100));
    setBulkBusy(true);
    setMessage(null);
    try {
      const response = await requestJson<DiscoveryAnalyzeResponse>("/api/discovery/analyze", {
        method: "POST",
        body: JSON.stringify({
          limit: selectedItems.length,
          quality: "todos",
          output_language: "pt",
          candidates: selectedItems,
        }),
      });
      const firstSaved = response.saved.length ? response.saved[0].run_id : undefined;
      const failures = response.failed_count ? ` ${response.failed_count} falharam e ficaram fora da carteira.` : "";
      const requested = response.requested_count || selectedItems.length;
      setMessage(`${response.saved_count} de ${requested} startup(s) adicionadas ao Radar.${failures}`);
      onBulkAnalyzed(firstSaved);
    } catch (exc) {
      const messageText = exc instanceof Error ? exc.message : "Falha ao adicionar candidatas ao Radar.";
      if (!messageText.includes("Method Not Allowed") && !messageText.includes("405")) {
        setMessage(messageText);
        setBulkBusy(false);
        return;
      }
      const savedIds: number[] = [];
      let failedCount = 0;
      for (const item of selectedItems) {
        const name = item.nome || item.name || item.title || "Startup candidata";
        setAnalyzingKey(item.url || name);
        try {
          const runId = await saveCandidateToRadar(item);
          if (runId) savedIds.push(runId);
          else failedCount += 1;
        } catch {
          failedCount += 1;
        }
      }
      const failures = failedCount ? ` ${failedCount} falharam e ficaram fora da carteira.` : "";
      setMessage(`${savedIds.length} de ${selectedItems.length} startup(s) adicionadas ao Radar.${failures}`);
      onBulkAnalyzed(savedIds[0]);
    } finally {
      setAnalyzingKey(null);
      setBulkBusy(false);
    }
  };

  return (
    <section className="form-page">
      <div className="form-card">
        <p className="eyebrow">Outbound</p>
        <h3>Encontrar candidatas</h3>
        <div className="form-row">
          <select value={campaign} onChange={(event) => setCampaign(event.target.value)}>
            <option value="full">Campanha completa</option>
            <option value="ai_native">Foco alta maturidade</option>
            <option value="competitors">Stack concorrente</option>
            <option value="nvidia_fit">Fit NVIDIA</option>
            <option value="frameworks">Frameworks e stack</option>
            <option value="careers">Vagas tecnicas</option>
            <option value="sectors">Setores prioritarios</option>
            <option value="wrapper_risk">Risco wrapper</option>
          </select>
          <input type="number" min={1} max={100} value={limit} onChange={(event) => setLimit(Number(event.target.value))} />
        </div>
        <label className="check-row">
          <input type="checkbox" checked={fetchPages} onChange={(event) => setFetchPages(event.target.checked)} />
          Coletar paginas publicas quando permitido
        </label>
        <p className="muted-copy">A triagem descarta tutoriais, documentacao, Wikipedia e paginas de framework sem contexto de empresa.</p>
        <div className="form-row">
          <button className="primary-button" onClick={runDiscovery} disabled={busy || bulkBusy}>
            <Compass size={16} />
            {busy ? "Buscando..." : "Buscar candidatas"}
          </button>
          <button className="ghost-button" onClick={analyzeSavedCandidates} disabled={bulkBusy || busy || !items.length}>
            <Sparkles size={16} />
            {bulkBusy ? "Adicionando..." : "Adicionar todas ao Radar"}
          </button>
        </div>
        {message ? <Alert text={message} tone="info" /> : null}
      </div>
      <div className="result-grid">
        {items.length ? (
          items.map((item, index) => (
            <DiscoveryCard
              item={item}
              key={`${item.url}-${index}`}
              analyzing={analyzingKey === (item.url || item.nome || item.name || item.title)}
              onAnalyze={() => analyzeCandidate(item)}
            />
          ))
        ) : (
          <div className="list-empty">Nenhuma candidata qualificada salva ainda.</div>
        )}
      </div>
    </section>
  );
}

function DiscoveryCard({
  item,
  analyzing,
  onAnalyze,
}: {
  item: DiscoveryItem;
  analyzing: boolean;
  onAnalyze: () => void;
}) {
  const signals = [
    ...(item.nvidia_signals || []),
    ...(item.ai_framework_signals || []),
    ...(item.competitor_stack_signals || []),
    ...(item.maturity_signals || []),
    ...(item.wrapper_risk_signals || []),
  ].slice(0, 5);
  const summary =
    item.evidence_excerpt ||
    item.recommended_action ||
    item.snippet ||
    item.motivo ||
    item.snippets?.[0] ||
    "Candidata salva sem resumo suficiente.";
  return (
    <article className="result-card">
      <h4>{item.nome || item.name || item.title || "Candidata"}</h4>
      <p>{summary}</p>
      {signals.length ? (
        <div className="signal-row">
          {signals.map((signal) => (
            <span key={signal}>{signal}</span>
          ))}
        </div>
      ) : null}
      <div className="startup-card-foot">
        <span>{item.quality_tier || "qualificada"} - {numberFormat(item.score || 0)} pts</span>
        <span className="discovery-actions">
          <button className="ghost-button compact" onClick={onAnalyze} disabled={analyzing}>
            {analyzing ? "Analisando..." : "Adicionar ao Radar"}
          </button>
          {item.url ? (
            <a href={item.url} target="_blank" rel="noreferrer">
              Abrir <ExternalLink size={13} />
            </a>
          ) : null}
        </span>
      </div>
    </article>
  );
}

function AskView({ onSelectRun, onOpenRadar }: { onSelectRun: (runId: number) => void; onOpenRadar: () => void }) {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<BaseAnswer | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const submit = async () => {
    if (!question.trim()) return;
    setBusy(true);
    setMessage(null);
    try {
      const response = await requestJson<BaseAnswer>("/api/ask", {
        method: "POST",
        body: JSON.stringify({ question, limit: 20 }),
      });
      setAnswer(response);
    } catch (exc) {
      setMessage(exc instanceof Error ? exc.message : "Falha na consulta.");
    } finally {
      setBusy(false);
    }
  };

  const rows = answer?.items || answer?.results || [];
  const examples = [
    "healthtechs com NeMo Guardrails",
    "startups com stack concorrente",
    "fintechs AI-native com prioridade alta",
  ];
  return (
    <section className="form-page">
      <div className="form-card wide">
        <p className="eyebrow">Filtro operacional</p>
        <h3>Encontrar startups na carteira</h3>
        <p className="muted-copy">Busca apenas nos perfis ja mapeados e retorna uma lista auditavel.</p>
        <div className="ask-row">
          <input
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Ex: startups de saude que precisam de guardrails"
            onKeyDown={(event) => {
              if (event.key === "Enter") submit();
            }}
          />
          <button className="primary-button" onClick={submit} disabled={busy}>
            <Send size={16} />
            {busy ? "Buscando..." : "Buscar"}
          </button>
        </div>
        <div className="example-row">
          {examples.map((example) => (
            <button key={example} className="ghost-button compact" onClick={() => setQuestion(example)}>
              {example}
            </button>
          ))}
        </div>
        {answer?.filters ? <p className="muted-copy">Filtros aplicados: {JSON.stringify(answer.filters)}</p> : null}
        {message ? <Alert text={message} tone="danger" /> : null}
      </div>
      <div className="result-grid">
        {rows.map((run) => (
          <button
            className="result-card as-button"
            key={run.run_id}
            onClick={() => {
              onSelectRun(run.run_id);
              onOpenRadar();
            }}
          >
            <h4>{run.nome || "Startup"}</h4>
            <p>{run.profile?.produto_descricao || run.setor || "Sem resumo."}</p>
            <span className="class-badge" style={{ color: classColor(run.classificacao), borderColor: classColor(run.classificacao) }}>
              {run.classificacao || "indeterminado"}
            </span>
          </button>
        ))}
      </div>
    </section>
  );
}

function KnowledgeView() {
  const [stats, setStats] = useState<KnowledgeStats | null>(null);
  const [query, setQuery] = useState("NVIDIA NIM custo latencia inferencia");
  const [hits, setHits] = useState<KnowledgeHit[]>([]);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const loadStats = useCallback(async () => {
    setStats(await requestJson<KnowledgeStats>("/api/knowledge/stats"));
  }, []);

  useEffect(() => {
    loadStats().catch(() => undefined);
  }, [loadStats]);

  const search = async () => {
    if (!query.trim()) return;
    setBusy(true);
    try {
      const response = await requestJson<KnowledgeSearchResponse>(`/api/knowledge/search?q=${encodeURIComponent(query)}&limit=7`);
      setHits(response.items || []);
    } catch (exc) {
      setMessage(exc instanceof Error ? exc.message : "Falha na busca.");
    } finally {
      setBusy(false);
    }
  };

  const rebuild = async (kind: "ingest" | "rebuild") => {
    setBusy(true);
    setMessage(null);
    try {
      await requestJson(`/api/knowledge/${kind}`, { method: "POST", body: JSON.stringify({}) });
      await loadStats();
      setMessage("Base atualizada.");
    } catch (exc) {
      setMessage(exc instanceof Error ? exc.message : "Falha ao atualizar.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="form-page">
      <div className="metrics-row">
        <MetricCard label="Trechos" value={numberFormat(Number(stats?.["ch" + "unk_count"] || 0))} />
        <MetricCard label="Documentos" value={numberFormat(Number(stats?.document_count || 0))} />
        <MetricCard label="Fontes" value={numberFormat(Number(stats?.source_count || 0))} />
      </div>
      <div className="form-card wide">
        <p className="eyebrow">Conhecimento NVIDIA</p>
        <h3>Buscar com citacao de fonte</h3>
        <div className="ask-row">
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Buscar tecnologia ou caso de uso" />
          <button className="primary-button" onClick={search} disabled={busy}>
            <Search size={16} />
            Buscar
          </button>
        </div>
        <div className="form-row">
          <button className="ghost-button" onClick={() => rebuild("ingest")} disabled={busy}>
            Ingerir fontes
          </button>
          <button className="ghost-button" onClick={() => rebuild("rebuild")} disabled={busy}>
            Recriar indice
          </button>
        </div>
        {message ? <Alert text={message} tone="info" /> : null}
      </div>
      <div className="result-grid">
        {hits.map((hit, index) => (
          <article className="result-card" key={`${hit.fonte_url}-${index}`}>
            <h4>{hit.tecnologia || hit.title || "Fonte NVIDIA"}</h4>
            <p>{firstText(hit).slice(0, 360)}</p>
            {hit.fonte_url ? (
              <a href={hit.fonte_url} target="_blank" rel="noreferrer">
                Fonte <ExternalLink size={13} />
              </a>
            ) : null}
          </article>
        ))}
      </div>
    </section>
  );
}

function ReviewView({ onUpdated }: { onUpdated: () => void }) {
  const [items, setItems] = useState<StartupRun[]>([]);
  const [status, setStatus] = useState("pendente");
  const [note, setNote] = useState<Record<number, string>>({});
  const [message, setMessage] = useState<string | null>(null);

  const load = useCallback(async () => {
    const response = await requestJson<ReviewResponse>(`/api/review?status=${status}&limit=100`);
    setItems(response.items || []);
  }, [status]);

  useEffect(() => {
    load().catch((exc) => setMessage(exc instanceof Error ? exc.message : "Falha ao carregar revisoes."));
  }, [load]);

  const update = async (runId: number, reviewStatus: "aprovado" | "rejeitado") => {
    await requestJson(`/api/review/${runId}`, {
      method: "PATCH",
      body: JSON.stringify({ review_status: reviewStatus, review_nota: note[runId] || null }),
    });
    await load();
    onUpdated();
  };

  return (
    <section className="form-page">
      <div className="filter-bar inline">
        <select value={status} onChange={(event) => setStatus(event.target.value)}>
          <option value="pendente">Pendentes</option>
          <option value="aprovado">Aprovados</option>
          <option value="rejeitado">Rejeitados</option>
          <option value="todos">Todos</option>
        </select>
      </div>
      {message ? <Alert text={message} tone="danger" /> : null}
      <div className="review-list">
        {items.length ? (
          items.map((run) => (
            <article className={`review-card ${run.review_status || "pendente"}`} key={run.run_id}>
              <div>
                <h4>{run.nome || "Startup"}</h4>
                <p>{(run.judge_motivos || []).join(" ") || "Marcada para revisao."}</p>
              </div>
              <textarea
                value={note[run.run_id] || ""}
                onChange={(event) => setNote((current) => ({ ...current, [run.run_id]: event.target.value }))}
                placeholder="Nota opcional"
              />
              <div className="form-row">
                <button className="ghost-button success" onClick={() => update(run.run_id, "aprovado")}>
                  <Check size={15} /> Aprovar
                </button>
                <button className="ghost-button danger" onClick={() => update(run.run_id, "rejeitado")}>
                  <X size={15} /> Rejeitar
                </button>
              </div>
            </article>
          ))
        ) : (
          <div className="list-empty">Nao ha itens nesta fila.</div>
        )}
      </div>
    </section>
  );
}

function ActivityView() {
  const [summary, setSummary] = useState<TimelineItem[]>([]);
  const [quality, setQuality] = useState<QualityResponse | null>(null);

  useEffect(() => {
    Promise.all([requestJson<ActivityResponse>("/api/observability"), requestJson<QualityResponse>("/api/quality")])
      .then(([activityData, qualityData]) => {
        setSummary(activityData.items || []);
        setQuality(qualityData);
      })
      .catch(() => undefined);
  }, []);

  return (
    <section className="form-page">
      <div className="metrics-row">
        <MetricCard label="Cobertura media" value={percentFormat(quality?.averages.evidence_coverage)} />
        <MetricCard label="Prontos para envio" value={percentFormat(quality?.averages.export_ready_rate)} />
        <MetricCard label="Com lacunas" value={percentFormat(quality?.averages.unsupported_claim_rate)} />
      </div>
      <div className="profile-section">
        <h3>Linha do tempo agregada</h3>
        <div className="activity-table">
          <div className="activity-head">
            <span>Etapa</span>
            <span>Modo</span>
            <span>Sucesso</span>
            <span>Media</span>
            <span>Total</span>
          </div>
          {summary.length ? (
            summary.map((item, index) => (
              <div className="activity-row" key={`${item.step}-${index}`}>
                <span>{publicStepName(item.step)}</span>
                <span>{publicModeName(item.mode)}</span>
                <span>{percentFormat(Number(item.success_rate || 0))}</span>
                <span>{numberFormat(item.avg_latency_ms || 0)} ms</span>
                <span>{numberFormat(Number(item.count || 0))}</span>
              </div>
            ))
          ) : (
            <div className="list-empty">Sem dados de atividade.</div>
          )}
        </div>
      </div>
    </section>
  );
}

function SetupView({ readiness }: { readiness: Readiness | null }) {
  const [sources, setSources] = useState<{ items: Record<string, unknown>[] } | null>(null);
  const [playbooks, setPlaybooks] = useState<{ items: Record<string, unknown>[] } | null>(null);

  useEffect(() => {
    Promise.all([requestJson<{ items: Record<string, unknown>[] }>("/api/sources"), requestJson<{ items: Record<string, unknown>[] }>("/api/playbooks")])
      .then(([sourceData, playbookData]) => {
        setSources(sourceData);
        setPlaybooks(playbookData);
      })
      .catch(() => undefined);
  }, []);

  return (
    <section className="form-page">
      <div className="setup-grid">
        {(readiness?.checks || []).map((check) => (
          <article className="setup-card" key={check.id}>
            <span className={`status-dot ${check.status === "ok" ? "ok" : check.status === "attention" ? "attention" : "optional"}`} />
            <h4>{check.label}</h4>
            <p>{check.detail}</p>
          </article>
        ))}
      </div>
      <div className="two-columns">
        <div className="profile-section">
          <h3>Fontes governadas</h3>
          <div className="source-list">
            {(sources?.items || []).map((item) => (
              <article key={String(item.id)}>
                <strong>{String(item.label || item.id)}</strong>
                <span>{String(item.category || "")}</span>
                <p>{String(item.allowed_use || "")}</p>
              </article>
            ))}
          </div>
        </div>
        <div className="profile-section">
          <h3>Roteiros de ativacao</h3>
          <div className="source-list">
            {(playbooks?.items || []).map((item) => (
              <article key={String(item.id)}>
                <strong>{String(item.title || item.id)}</strong>
                <p>{String(item.next_action || "")}</p>
              </article>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <article className="metric-card">
      <span>{label}</span>
      <strong>{value}</strong>
    </article>
  );
}

export default App;
