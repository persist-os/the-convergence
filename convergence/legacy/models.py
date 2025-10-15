"""
Data models for the legacy system.

These models are completely RL-agnostic and work with any API type.
They track optimization runs, winners, and decision provenance.
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class TrackingBackend(str, Enum):
    """Available tracking backends."""
    BUILTIN = "builtin"  # SQLite + CSV (always available)
    MLFLOW = "mlflow"  # Free, self-hosted (pip install mlflow)
    AIM = "aim"  # Free, lightweight (pip install aim)
    WEAVE = "weave"  # Optional, paid (Weights & Biases)


class LegacyConfig(BaseModel):
    """Configuration for legacy tracking."""
    enabled: bool = True
    session_id: Optional[str] = None  # Auto-generated if not provided
    tracking_backend: TrackingBackend = TrackingBackend.BUILTIN
    
    # Storage paths
    sqlite_path: str = "./data/legacy.db"
    export_dir: str = "./legacy"
    
    # Export settings
    export_formats: List[str] = Field(default_factory=lambda: ["winners_only", "full_audit"])
    
    # External tracker configs (optional)
    mlflow_config: Dict[str, Any] = Field(default_factory=dict)
    aim_config: Dict[str, Any] = Field(default_factory=dict)
    weave_config: Dict[str, Any] = Field(default_factory=dict)


class Session(BaseModel):
    """A collection of related optimization runs."""
    session_id: str
    name: Optional[str] = None
    api_name: str  # "openai", "apify", "tavily", etc.
    api_endpoint: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    config_fingerprint: str  # Hash of search space + evaluation config
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TestCaseResult(BaseModel):
    """Result for a single test case in a run."""
    result_id: str
    run_id: str
    test_case_id: str
    config: Dict[str, Any]  # The config that was tested
    score: float
    metrics: Dict[str, float]
    latency_ms: float
    cost_usd: float = 0.0
    response_text: Optional[str] = None
    full_response: Optional[Dict[str, Any]] = None
    success: bool = True
    error: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class OptimizationRun(BaseModel):
    """A single optimization run (one generation)."""
    run_id: str
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    api_name: str
    api_endpoint: str
    config: Dict[str, Any]  # The configuration tested
    test_case_ids: List[str]  # Which test cases were used
    test_results: List[TestCaseResult]  # Detailed per-test results
    aggregate_score: float
    aggregate_metrics: Dict[str, float]
    duration_ms: float
    cost_usd: float = 0.0
    generation: int = 0  # For evolutionary tracking
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TestCaseWinner(BaseModel):
    """Current best configuration for a specific test case."""
    winner_id: str
    test_case_id: str
    api_name: str
    best_config: Dict[str, Any]
    best_score: float
    best_run_id: str
    previous_winner_id: Optional[str] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    improvement: float = 0.0  # Improvement over previous winner
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class RunLineage(BaseModel):
    """Tracks how configurations evolved from each other."""
    lineage_id: str
    parent_run_id: Optional[str] = None
    child_run_id: str
    relationship_type: str  # "evolution", "mutation", "crossover", "manual", "mab_selection"
    changes: Dict[str, Any]  # What changed from parent to child
    improvement: float = 0.0  # Score improvement
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DecisionLog(BaseModel):
    """Audit trail entry for a decision."""
    decision_id: str
    run_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    decision_type: str  # "winner_update", "config_selected", "session_created", etc.
    reasoning: str  # Human-readable explanation
    data: Dict[str, Any] = Field(default_factory=dict)  # Supporting data
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

