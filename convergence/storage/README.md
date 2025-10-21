# Convergence Storage Backends

Pluggable storage system for The Convergence optimization framework.

## Available Backends

### Built-in Backends

1. **SQLiteStorage** - Fast local database with relational queries
2. **FileStorage** - Human-readable JSON files for audit trails
3. **MemoryStorage** - In-memory cache for hot data
4. **MultiBackendStorage** - Dual-write to multiple backends (redundancy)

### Optional Backends

5. **ConvexStorage** - Real-time serverless database integration

## ConvexStorage - Generic Integration

The `ConvexStorage` backend provides a flexible bridge to any Convex setup.

### Quick Start

```python
from convergence.storage import ConvexStorage
from convergence import ConvergenceSDK

# Option 1: Auto-import (if you have standard backend structure)
storage = ConvexStorage()

# Option 2: Inject your Convex client
from my_app.convex import my_client
storage = ConvexStorage(client=my_client)

# Option 3: Inject custom functions
storage = ConvexStorage(
    save_rl_data_fn=my_save_fn,
    save_experiment_fn=my_experiment_fn,
)

# Use with Convergence
sdk = ConvergenceSDK(storage=storage)
await sdk.optimize("my_system", test_cases)
```

### Required Methods

Your Convex client must implement these async methods:

```python
# RL Training Data
async def save_rl_data(rl_key, rl_record_type, agent_id, episode_timestamp, 
                       rl_episode_data, **optional_fields) -> dict

async def get_rl_data_by_key(rl_key: str) -> dict

# Optimization Experiments
async def save_experiment(experiment_id, optimization_run_id, system_name,
                         algorithm_name, test_case_id, tested_config,
                         experiment_score, test_passed, experiment_timestamp,
                         **optional_fields) -> dict

# Optimization Runs
async def start_optimization_run(run_id, system_name, algorithm_name) -> dict

async def complete_optimization_run(run_id, **optional_fields) -> dict

async def get_optimization_run(run_id: str) -> dict
```

All methods should return: `{"success": bool, "data": any, "error": str}`

### Key Routing

Data is automatically routed by key prefix:

- `episode:*` → RL training data
- `trajectory:*` → RL trajectories
- `agent_legacy:*` → Agent learned knowledge
- `training_run:*` → RL policy training sessions
- `experiment:*` → Optimization experiment results
- `run:*` → Optimization run metadata

### Example: Direct HTTP Integration

```python
import httpx
from convergence.storage import ConvexStorage

class MyConvexClient:
    def __init__(self, url: str, api_key: str):
        self.url = url
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    async def save_rl_data(self, **kwargs):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.url}/api/convergence/storage/rl-data",
                json=kwargs,
                headers=self.headers
            )
            return response.json()
    
    async def save_experiment(self, **kwargs):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.url}/api/convergence/storage/experiments",
                json=kwargs,
                headers=self.headers
            )
            return response.json()
    
    # ... implement other methods

# Use it
client = MyConvexClient("https://my-app.convex.cloud", "api-key")
storage = ConvexStorage(client=client)
```

### Example: Function Injection

```python
from convergence.storage import ConvexStorage

async def save_to_my_db(rl_key, rl_record_type, agent_id, **kwargs):
    # Your custom database logic
    await my_database.insert("rl_data", {
        "key": rl_key,
        "type": rl_record_type,
        "agent_id": agent_id,
        **kwargs
    })
    return {"success": True}

storage = ConvexStorage(save_rl_data_fn=save_to_my_db)
```

## Other Storage Backends

### SQLite + File (Legacy Multi-Backend)

```python
from convergence.storage import get_legacy_storage

# Dual-write to SQLite + Files for redundancy
storage = get_legacy_storage(cache_enabled=True)
```

### Custom Storage

Implement the `StorageProtocol`:

```python
from convergence.storage import StorageProtocol

class MyCustomStorage:
    async def save(self, key: str, value: Any) -> None:
        ...
    
    async def load(self, key: str) -> Any:
        ...
    
    async def exists(self, key: str) -> bool:
        ...
    
    # ... other methods

storage = MyCustomStorage()
```

## Data Structure

### RL Training Data

```python
{
    "agent_id": "agent_123",
    "station": "web_playground",
    "civilization_id": "civ_001",
    "timestamp": 1234567890,
    "reward": 0.85,
    "fitness_score": 0.90,
    "success": True,
    # Full episode data
}
```

### Optimization Experiments

```python
{
    "experiment_id": "exp_001",
    "optimization_run_id": "run_abc",
    "system_name": "context_enrichment",
    "algorithm_name": "mab_evolution",
    "test_case_id": "test_001",
    "tested_config": {"threshold": 0.75, "limit": 10},
    "score": 0.92,
    "passed": True,
    "timestamp": 1234567890,
    "generation": 3,
    "latency_ms": 150,
    "cost_usd": 0.002,
    "metrics": {...}
}
```

### Optimization Runs

```python
{
    "run_id": "run_abc",
    "system_name": "context_enrichment",
    "algorithm_name": "mab_evolution",
    "status": "completed",  # "started" | "completed"
    "winning_config_id": "config_xyz",
    "winning_config": {...},
    "total_generations": 5,
    "convergence_achieved": True
}
```

## Testing Your Integration

```python
from convergence.storage import ConvexStorage

# Initialize
storage = ConvexStorage(client=your_client)

# Test save
await storage.save("episode:test:001", {
    "agent_id": "test_agent",
    "timestamp": 123456,
    "reward": 0.9
})

# Test load
data = await storage.load("episode:test:001")
print(data)

# Test exists
exists = await storage.exists("episode:test:001")
print(f"Exists: {exists}")
```

## Best Practices

1. **Use Auto-Import** if you have standard backend structure
2. **Inject Client** for maximum flexibility
3. **Inject Functions** for custom routing logic
4. **Return Proper Format**: `{"success": bool, "data": any, "error": str}`
5. **Handle Errors Gracefully**: Don't crash optimization runs
6. **Test Integration**: Verify all required methods work

## Support

- See `CONVERGENCE_CONVEX_INTEGRATION.md` for detailed integration guide
- Check `examples/` for reference implementations
- Implement `StorageProtocol` for custom backends

