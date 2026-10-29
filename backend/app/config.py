"""Central configuration.

Every threshold that influences system behaviour is exposed here so that the
ablation study and the evaluation harness can vary a single knob at a time.
Values may be overridden through environment variables or a ``.env`` file.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(BACKEND_ROOT / ".env", BACKEND_ROOT.parent / ".env"),
        env_prefix="SELFRAG_",
        extra="ignore",
    )

    # ---------------------------------------------------------------- runtime
    app_name: str = "Enhanced Self-RAG"
    log_level: str = "INFO"
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]
    )

    # ------------------------------------------------------------ persistence
    data_dir: Path = DATA_DIR
    upload_dir: Path = DATA_DIR / "uploads"
    index_dir: Path = DATA_DIR / "index"
    results_dir: Path = DATA_DIR / "results"
    demo_dir: Path = DATA_DIR / "demo"

    @model_validator(mode="after")
    def _derive_dirs(self) -> "Settings":
        """Keep derived directories under ``data_dir`` unless they were overridden."""

        default_root = DATA_DIR.resolve()
        if Path(self.upload_dir).resolve() == (default_root / "uploads").resolve():
            self.upload_dir = Path(self.data_dir) / "uploads"
        if Path(self.index_dir).resolve() == (default_root / "index").resolve():
            self.index_dir = Path(self.data_dir) / "index"
        if Path(self.results_dir).resolve() == (default_root / "results").resolve():
            self.results_dir = Path(self.data_dir) / "results"
        if Path(self.demo_dir).resolve() == (default_root / "demo").resolve():
            self.demo_dir = Path(self.data_dir) / "demo"
        if Path(self.chroma_path).resolve() == (default_root / "index" / "chroma").resolve():
            self.chroma_path = Path(self.index_dir) / "chroma"
        # Honour unprefixed USE_LLM_JUDGE when SELFRAG_USE_LLM_JUDGE is unset.
        raw = os.environ.get("USE_LLM_JUDGE")
        if raw is not None and os.environ.get("SELFRAG_USE_LLM_JUDGE") is None:
            self.use_llm_judge = raw.strip().lower() in {"1", "true", "yes", "on"}
        return self

    # -------------------------------------------------------------- providers
    # ``auto`` resolves to the best provider that is actually available.
    llm_provider: str = "auto"
    embedding_provider: str = "auto"
    vector_store: str = "numpy"

    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_base_url: str | None = None

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"

    sentence_transformer_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    pinecone_api_key: str | None = None
    pinecone_index: str = "enhanced-self-rag"
    chroma_path: Path = DATA_DIR / "index" / "chroma"

    llm_temperature: float = 0.0
    llm_max_tokens: int = 700

    # -------------------------------------------------------------- ingestion
    chunk_size: int = 900
    chunk_overlap: int = 150
    min_chunk_chars: int = 80

    # -------------------------------------------------------------- retrieval
    initial_top_k: int = 5
    max_top_k: int = 12
    top_k_increment: int = 4
    rrf_k: int = 60
    dense_weight: float = 0.5
    bm25_weight: float = 0.5

    # ----------------------------------------------------------- evidence gate
    # Weights of the evidence score components. Normalised at use time.
    ev_weight_relevance: float = 0.40
    ev_weight_source_quality: float = 0.15
    ev_weight_coverage: float = 0.30
    ev_weight_consistency: float = 0.15

    evidence_threshold: float = 0.55
    evidence_low_threshold: float = 0.35
    min_supporting_chunks: int = 2
    chunk_relevance_floor: float = 0.18

    # -------------------------------------------------------------- self loops
    max_retrieval_attempts: int = 3
    max_correction_loops: int = 2

    # ----------------------------------------------------- claim verification
    # False (default): DeBERTa/DistilBERT NLI. True: lexical baseline (llm_judge_verify).
    # Env: SELFRAG_USE_LLM_JUDGE or USE_LLM_JUDGE.
    use_llm_judge: bool = False
    nli_model: str = "cross-encoder/nli-deberta-v3-xsmall"
    nli_fallback_model: str = "typeform/distilbert-base-uncased-mnli"
    nli_allow_download: bool = True
    claim_support_threshold: float = 0.55
    claim_partial_threshold: float = 0.32
    contradiction_threshold: float = 0.55
    claim_support_rate_target: float = 0.75

    # ------------------------------------------------------- topic clustering
    cluster_min_similarity: float = 0.42
    cluster_max_themes: int = 8
    cluster_sample_chars: int = 4000

    # ---------------------------------------------------------- originality
    plagiarism_min_ngram: int = 12
    plagiarism_uncited_ratio: float = 0.20
    plagiarism_quoted_ratio: float = 0.40

    # ------------------------------------------------------------- confidence
    conf_weight_evidence: float = 0.30
    conf_weight_claim_support: float = 0.35
    conf_weight_source_agreement: float = 0.15
    conf_weight_coverage: float = 0.20
    abstain_confidence_threshold: float = 0.35

    # ---------------------------------------------------------------- auth
    auth_required: bool = True
    session_ttl_seconds: int = 60 * 60 * 24 * 7
    google_client_id: str = ""

    # ------------------------------------------------------------ supabase
    # Optional Postgres via Supabase PostgREST. When both are set, auth users,
    # sessions, and Write stories use the remote DB; otherwise local JSON.
    supabase_url: str = ""
    supabase_service_key: str = ""

    # ---------------------------------------------------------- paper search
    semantic_scholar_api_key: str | None = None
    tavily_api_key: str | None = None
    paper_search_timeout_s: float = 20.0
    paper_pdf_max_bytes: int = 15_000_000
    unpaywall_email: str = "selfrag@localhost"

    def ensure_dirs(self) -> None:
        for path in (
            self.data_dir,
            self.upload_dir,
            self.index_dir,
            self.results_dir,
            self.demo_dir,
        ):
            Path(path).mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings


settings = get_settings()
