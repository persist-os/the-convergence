"""
Legacy storage layer - uses SQLite + CSV export.

This module handles:
- Session management
- Run tracking
- Winner updates
- CSV export (winners + audit trail)

Default backend is SQLite (no external dependencies).
Optional external trackers can be added (MLflow, Aim, Weave).
"""
import json
import hashlib
import uuid
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

import aiosqlite

from convergence.legacy.models import (
    LegacyConfig,
    Session,
    OptimizationRun,
    TestCaseResult,
    TestCaseWinner,
    RunLineage,
    DecisionLog,
    TrackingBackend,
)


class LegacyStore:
    """
    Storage layer for optimization legacy.
    
    Uses SQLite for structured queries + CSV for simple exports.
    Completely RL-agnostic and works with any API type.
    """
    
    def __init__(self, config: LegacyConfig):
        """
        Initialize legacy store.
        
        Args:
            config: Legacy configuration
        """
        self.config = config
        self.db_path = Path(config.sqlite_path)
        self.export_dir = Path(config.export_dir)
        
        # Ensure directories exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
        # Optional external tracker (Future: MLflow, Aim, Weave)
        self.tracker = None
        if config.tracking_backend != TrackingBackend.BUILTIN:
            print(f"📝 Note: {config.tracking_backend} tracker will be implemented in future release")
            print(f"   Using builtin (SQLite + CSV) for now")
        
        self._conn: Optional[aiosqlite.Connection] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _connect(self) -> None:
        """Establish database connection and create tables."""
        if self._conn is not None:
            return
        
        self._conn = await aiosqlite.connect(self.db_path)
        await self._create_tables()
    
    async def close(self) -> None:
        """Close database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None
    
    async def _create_tables(self) -> None:
        """Create all required tables if they don't exist."""
        # Sessions table
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                name TEXT,
                api_name TEXT,
                api_endpoint TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                config_fingerprint TEXT,
                metadata JSON
            )
        """)
        
        # Runs table
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                session_id TEXT,
                timestamp TIMESTAMP,
                api_name TEXT,
                api_endpoint TEXT,
                config JSON,
                test_case_ids JSON,
                aggregate_score REAL,
                aggregate_metrics JSON,
                duration_ms REAL,
                cost_usd REAL,
                generation INTEGER,
                metadata JSON,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        
        # Test case results table
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS test_case_results (
                result_id TEXT PRIMARY KEY,
                run_id TEXT,
                test_case_id TEXT,
                config JSON,
                score REAL,
                metrics JSON,
                latency_ms REAL,
                cost_usd REAL,
                response_text TEXT,
                full_response JSON,
                success BOOLEAN,
                error TEXT,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
        """)
        
        # Winners table
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS test_case_winners (
                winner_id TEXT PRIMARY KEY,
                test_case_id TEXT,
                api_name TEXT,
                best_config JSON,
                best_score REAL,
                best_run_id TEXT,
                previous_winner_id TEXT,
                updated_at TIMESTAMP,
                improvement REAL,
                FOREIGN KEY (best_run_id) REFERENCES runs(run_id),
                UNIQUE(test_case_id, api_name)
            )
        """)
        
        # Lineage table
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS run_lineage (
                lineage_id TEXT PRIMARY KEY,
                parent_run_id TEXT,
                child_run_id TEXT,
                relationship_type TEXT,
                changes JSON,
                improvement REAL,
                FOREIGN KEY (parent_run_id) REFERENCES runs(run_id),
                FOREIGN KEY (child_run_id) REFERENCES runs(run_id)
            )
        """)
        
        # Decision log table
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS decision_log (
                decision_id TEXT PRIMARY KEY,
                run_id TEXT,
                timestamp TIMESTAMP,
                decision_type TEXT,
                reasoning TEXT,
                data JSON,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
        """)
        
        # Create indexes for fast queries
        await self._conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_session ON runs(session_id)")
        await self._conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_api ON runs(api_name)")
        await self._conn.execute("CREATE INDEX IF NOT EXISTS idx_results_run ON test_case_results(run_id)")
        await self._conn.execute("CREATE INDEX IF NOT EXISTS idx_results_test_case ON test_case_results(test_case_id)")
        await self._conn.execute("CREATE INDEX IF NOT EXISTS idx_winners_api_test ON test_case_winners(api_name, test_case_id)")
        
        await self._conn.commit()
    
    async def create_or_get_session(
        self,
        session_id: Optional[str],
        api_name: str,
        api_endpoint: str,
        config_fingerprint: str,
        name: Optional[str] = None
    ) -> Session:
        """
        Create new session or get existing one.
        
        Args:
            session_id: Optional session ID (auto-generated if None)
            api_name: API name (e.g., "openai", "apify")
            api_endpoint: API endpoint URL
            config_fingerprint: Hash of search space + evaluation config
            name: Optional human-readable name
        
        Returns:
            Session object
        """
        if not session_id:
            session_id = f"session_{uuid.uuid4().hex[:12]}"
        
        # Check if session exists
        cursor = await self._conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?",
            (session_id,)
        )
        row = await cursor.fetchone()
        
        if row:
            # Session exists, return it
            return Session(
                session_id=row[0],
                name=row[1],
                api_name=row[2],
                api_endpoint=row[3],
                created_at=datetime.fromisoformat(row[4]),
                updated_at=datetime.fromisoformat(row[5]),
                config_fingerprint=row[6],
                metadata=json.loads(row[7]) if row[7] else {}
            )
        
        # Create new session
        session = Session(
            session_id=session_id,
            name=name,
            api_name=api_name,
            api_endpoint=api_endpoint,
            config_fingerprint=config_fingerprint
        )
        
        await self._conn.execute("""
            INSERT INTO sessions (session_id, name, api_name, api_endpoint, created_at, updated_at, config_fingerprint, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session.session_id,
            session.name,
            session.api_name,
            session.api_endpoint,
            session.created_at.isoformat(),
            session.updated_at.isoformat(),
            session.config_fingerprint,
            json.dumps(session.metadata)
        ))
        await self._conn.commit()
        
        return session
    
    async def record_run(self, run: OptimizationRun) -> None:
        """
        Record an optimization run and its test case results.
        
        Args:
            run: OptimizationRun object with all data
        """
        # Insert run
        await self._conn.execute("""
            INSERT INTO runs (
                run_id, session_id, timestamp, api_name, api_endpoint,
                config, test_case_ids, aggregate_score, aggregate_metrics,
                duration_ms, cost_usd, generation, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run.run_id,
            run.session_id,
            run.timestamp.isoformat(),
            run.api_name,
            run.api_endpoint,
            json.dumps(run.config),
            json.dumps(run.test_case_ids),
            run.aggregate_score,
            json.dumps(run.aggregate_metrics),
            run.duration_ms,
            run.cost_usd,
            run.generation,
            json.dumps(run.metadata)
        ))
        
        # Insert test case results
        for result in run.test_results:
            await self._conn.execute("""
                INSERT INTO test_case_results (
                    result_id, run_id, test_case_id, config, score, metrics,
                    latency_ms, cost_usd, response_text, full_response, success, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.result_id,
                result.run_id,
                result.test_case_id,
                json.dumps(result.config),
                result.score,
                json.dumps(result.metrics),
                result.latency_ms,
                result.cost_usd,
                result.response_text,
                json.dumps(result.full_response) if result.full_response else None,
                result.success,
                result.error
            ))
        
        await self._conn.commit()
        
        # Update winners
        await self._update_winners(run)
    
    async def _update_winners(self, run: OptimizationRun) -> None:
        """Update winners if this run has better results."""
        for result in run.test_results:
            # Get current winner
            cursor = await self._conn.execute("""
                SELECT winner_id, best_score FROM test_case_winners
                WHERE test_case_id = ? AND api_name = ?
            """, (result.test_case_id, run.api_name))
            row = await cursor.fetchone()
            
            should_update = False
            previous_winner_id = None
            improvement = 0.0
            
            if not row:
                # No previous winner
                should_update = True
                improvement = result.score
            elif result.score > row[1]:
                # Better than previous winner
                should_update = True
                previous_winner_id = row[0]
                improvement = result.score - row[1]
            
            if should_update:
                winner = TestCaseWinner(
                    winner_id=f"winner_{uuid.uuid4().hex[:12]}",
                    test_case_id=result.test_case_id,
                    api_name=run.api_name,
                    best_config=result.config,
                    best_score=result.score,
                    best_run_id=run.run_id,
                    previous_winner_id=previous_winner_id,
                    improvement=improvement
                )
                
                # Upsert winner
                await self._conn.execute("""
                    INSERT OR REPLACE INTO test_case_winners (
                        winner_id, test_case_id, api_name, best_config, best_score,
                        best_run_id, previous_winner_id, updated_at, improvement
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    winner.winner_id,
                    winner.test_case_id,
                    winner.api_name,
                    json.dumps(winner.best_config),
                    winner.best_score,
                    winner.best_run_id,
                    winner.previous_winner_id,
                    winner.updated_at.isoformat(),
                    winner.improvement
                ))
                await self._conn.commit()
    
    async def get_winner(self, test_case_id: str, api_name: str) -> Optional[TestCaseWinner]:
        """
        Get current best configuration for a test case.
        
        Args:
            test_case_id: Test case identifier
            api_name: API name
        
        Returns:
            TestCaseWinner if exists, None otherwise
        """
        cursor = await self._conn.execute("""
            SELECT * FROM test_case_winners
            WHERE test_case_id = ? AND api_name = ?
        """, (test_case_id, api_name))
        row = await cursor.fetchone()
        
        if not row:
            return None
        
        return TestCaseWinner(
            winner_id=row[0],
            test_case_id=row[1],
            api_name=row[2],
            best_config=json.loads(row[3]),
            best_score=row[4],
            best_run_id=row[5],
            previous_winner_id=row[6],
            updated_at=datetime.fromisoformat(row[7]),
            improvement=row[8]
        )
    
    async def get_top_winners(
        self,
        session_id: str,
        limit: int = 10
    ) -> List[TestCaseWinner]:
        """
        Get top N winners from a session for warm-start.
        
        Returns the best performing configs across all test cases in this session,
        ordered by score descending. Useful for initializing next optimization run
        with proven configurations.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of winners to return
            
        Returns:
            List of TestCaseWinner objects, ordered by best_score descending
        """
        cursor = await self._conn.execute("""
            SELECT DISTINCT
                w.winner_id,
                w.test_case_id,
                w.api_name,
                w.best_config,
                w.best_score,
                w.best_run_id,
                w.previous_winner_id,
                w.updated_at,
                w.improvement
            FROM test_case_winners w
            INNER JOIN runs r ON w.best_run_id = r.run_id
            WHERE r.session_id = ?
            ORDER BY w.best_score DESC
            LIMIT ?
        """, (session_id, limit))
        
        rows = await cursor.fetchall()
        
        winners = []
        for row in rows:
            winners.append(TestCaseWinner(
                winner_id=row[0],
                test_case_id=row[1],
                api_name=row[2],
                best_config=json.loads(row[3]),
                best_score=row[4],
                best_run_id=row[5],
                previous_winner_id=row[6],
                updated_at=datetime.fromisoformat(row[7]),
                improvement=row[8]
            ))
        
        return winners
    
    async def get_max_generation(self, session_id: str) -> Optional[int]:
        """
        Get the maximum generation number for a session.
        
        This allows subsequent optimization runs to continue numbering
        from where the previous run left off.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Maximum generation number, or None if no runs exist
        """
        cursor = await self._conn.execute("""
            SELECT MAX(generation) FROM runs
            WHERE session_id = ?
        """, (session_id,))
        row = await cursor.fetchone()
        
        if row and row[0] is not None:
            return row[0]
        return None
    
    async def get_experiment_count(self, session_id: str) -> int:
        """
        Get the total number of experiments (runs) for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Total number of experiments/runs
        """
        cursor = await self._conn.execute("""
            SELECT COUNT(*) FROM runs
            WHERE session_id = ?
        """, (session_id,))
        row = await cursor.fetchone()
        
        return row[0] if row else 0
    
    async def export_winners_csv(self, api_name: Optional[str] = None) -> Path:
        """
        Export winners to simple CSV format.
        
        Args:
            api_name: Optional filter by API name
        
        Returns:
            Path to exported CSV file
        """
        output_path = self.export_dir / f"winners_{api_name or 'all'}.csv"
        
        # Query winners
        if api_name:
            cursor = await self._conn.execute("""
                SELECT test_case_id, api_name, best_config, best_score, updated_at
                FROM test_case_winners
                WHERE api_name = ?
                ORDER BY test_case_id
            """, (api_name,))
        else:
            cursor = await self._conn.execute("""
                SELECT test_case_id, api_name, best_config, best_score, updated_at
                FROM test_case_winners
                ORDER BY api_name, test_case_id
            """)
        
        rows = await cursor.fetchall()
        
        # Write CSV
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['test_case_id', 'api_name', 'best_config', 'best_score', 'updated_at'])
            
            for row in rows:
                writer.writerow([
                    row[0],  # test_case_id
                    row[1],  # api_name
                    row[2],  # best_config (JSON string)
                    f"{row[3]:.4f}",  # best_score
                    row[4]  # updated_at
                ])
        
        return output_path
    
    async def export_audit_csv(self, session_id: Optional[str] = None) -> Path:
        """
        Export full audit trail to CSV.
        
        Args:
            session_id: Optional filter by session
        
        Returns:
            Path to exported CSV file
        """
        output_path = self.export_dir / f"audit_{session_id or 'all'}.csv"
        
        # Query test case results with run info
        if session_id:
            cursor = await self._conn.execute("""
                SELECT 
                    r.timestamp, r.session_id, r.run_id, r.api_name,
                    tcr.test_case_id, tcr.config, tcr.score, tcr.success
                FROM test_case_results tcr
                JOIN runs r ON tcr.run_id = r.run_id
                WHERE r.session_id = ?
                ORDER BY r.timestamp
            """, (session_id,))
        else:
            cursor = await self._conn.execute("""
                SELECT 
                    r.timestamp, r.session_id, r.run_id, r.api_name,
                    tcr.test_case_id, tcr.config, tcr.score, tcr.success
                FROM test_case_results tcr
                JOIN runs r ON tcr.run_id = r.run_id
                ORDER BY r.timestamp
            """)
        
        rows = await cursor.fetchall()
        
        # Write CSV
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'session_id', 'run_id', 'api_name', 'test_case_id', 'config', 'score', 'success'])
            
            for row in rows:
                writer.writerow(row)
        
        return output_path

