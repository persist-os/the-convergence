# Agno Reddit Agent Optimization with Azure OpenAI

Test and optimize Agno agents using Reddit's social tools with Azure-deployed models to find the best configurations for Reddit data retrieval tasks.

## Overview

This example uses **Multi-Armed Bandits (MAB)** with Thompson Sampling to efficiently test Agno agents across different Azure model deployments and configurations, determining which combinations work best for Reddit tool usage.

### Key Features

- **MAB-Based Model Selection**: Automatically discovers best Azure models via Thompson Sampling
- **Reddit Tool Testing**: Validates `search_subreddits`, `get_subreddit_info`, `get_post_details`
- **Azure Integration**: Direct integration with Azure AI Foundry deployments
- **Comprehensive Evaluation**: 4 metrics (accuracy, completeness, latency, token efficiency)
- **Scalable Architecture**: Add new Azure deployments as you create them

### Tools Tested

- `search_subreddits`: Find subreddits by topic
- `get_subreddit_info`: Get detailed subreddit information
- `get_post_details`: Retrieve post data
- `get_subreddit_posts`: Get posts from a subreddit (integration test)

### Agent Parameters Optimized

- **Model** (MAB arms): Any Azure deployment (gpt-4-1, o4-mini, gpt-4o, etc.)
- **Temperature**: 0.0 - 1.0 (reasoning consistency)
- **Max Completion Tokens**: 500 - 4000 (response length)
- **Instruction Style**: minimal, detailed, structured (prompt engineering)
- **Tool Strategy**: include_specific, include_all (tool selection)

### Quick Start Examples

```bash
# Single model optimization (gpt-4-1)
convergence optimize reddit_agent_optimization.yaml

# Multi-model comparison (uncomment models in YAML first)
# Tests gpt-4-1 vs o4-mini to find best model
convergence optimize reddit_agent_optimization.yaml

# Switch to a different model
# Edit YAML: change model.values to ["o4-mini"] instead of ["gpt-4-1"]
convergence optimize reddit_agent_optimization.yaml
```

---

## Setup

### 1. Install Dependencies

```bash
# Install the-convergence (if not already installed)
pip install the-convergence

# Install Agno for Reddit agents
pip install agno

# Install PRAW (Reddit API wrapper, required by Agno)
pip install praw
```

### 2. Configure Models (Model Registry)

The optimization system uses a **model registry** in `reddit_agent_optimization.yaml` to manage multiple Azure deployments. This makes it easy to:
- Switch between models without changing environment variables
- Test multiple models simultaneously
- Keep all model configs in one place

**Edit `reddit_agent_optimization.yaml`**:

```yaml
agent:
  models:
    gpt-4-1:  # Your primary model
      azure_deployment: "gpt-4.1"
      azure_endpoint: "https://heycontext-openai.cognitiveservices.azure.com"
      api_key_env: "AZURE_API_KEY"
      api_version: "2025-01-01-preview"
    
    o4-mini:  # Add as many models as you want
      azure_deployment: "o4-mini"
      azure_endpoint: "https://YOUR_AZURE_OPENAI_RESOURCE.openai.azure.com"
      api_key_env: "AZURE_API_KEY"
      api_version: "2025-01-01-preview"

search_space:
  parameters:
    model:
      type: "categorical"
      values:
        - "gpt-4-1"  # Active model(s)
        # - "o4-mini"  # Uncomment to test multiple models
```

**Benefits**:
- ✅ No more `export AZURE_OPENAI_ENDPOINT` needed!
- ✅ Easy model switching: just change the `values` list
- ✅ Per-model API versions (o4-mini vs gpt-4.1)
- ✅ MAB-ready for multi-model comparison

### 3. Get Reddit API Credentials

Reddit requires OAuth credentials for API access:

1. Go to https://www.reddit.com/prefs/apps
2. Click **"Create App"** or **"Create Another App"**
3. Select **"script"** as the app type
4. Fill in:
   - **name**: anything (e.g., "reddit-agent-tester")
   - **description**: optional
   - **redirect uri**: http://localhost:8080 (required but not used for scripts)
5. Click **Create app**
6. Note your credentials:
   - **client_id**: String under "personal use script" (14 characters)
   - **client_secret**: Longer string labeled "secret"

### 4. Set Environment Variables

```bash
# Reddit API credentials (required)
export REDDIT_CLIENT_ID="your_client_id_here"
export REDDIT_CLIENT_SECRET="your_client_secret_here"

# Azure OpenAI API key (required)
# Note: Endpoints are now configured in the YAML model registry
export AZURE_API_KEY="your_azure_api_key"

# Optional: For write operations (not needed for read-only tests)
# export REDDIT_USERNAME="your_reddit_username"
# export REDDIT_PASSWORD="your_reddit_password"

# Optional: Use different API keys for different models
# export AZURE_API_KEY_GPT4="your_gpt4_specific_key"
```

**What changed?**: The model registry (step 2) eliminates the need for `AZURE_OPENAI_ENDPOINT` environment variables. All model configurations (endpoints, API versions, deployments) are now in the YAML file.

### 5. Test Reddit Connection (Optional)

```bash
cd examples/agno_agents/reddit
python reddit_agent_runner.py
```

This will verify your Reddit credentials are working.

### 6. Run Optimization

```bash
# From the-convergence root directory
convergence optimize examples/agno_agents/reddit/reddit_agent_optimization.yaml
```

---

## Test Cases

### Test 1: Search Technology Subreddits
- **Function**: `search_subreddits`
- **Task**: Find technology-related subreddits
- **Validation**: 
  - Schema (name, subscribers, description fields)
  - Contains r/technology
  - Subscriber counts present and reasonable (>1K)
  - Keywords: "technology", "tech"
- **Category**: Unit test
- **Difficulty**: Easy

### Test 2: Get r/technology Info
- **Function**: `get_subreddit_info`
- **Task**: Retrieve detailed information about r/technology
- **Validation**: 
  - Complete fields (display_name, title, description, subscribers, created_utc, url)
  - Subscriber count accurate (>10M for r/technology)
  - Description contains tech-related keywords
- **Category**: Unit test
- **Difficulty**: Easy

### Test 3: Get Top Post Details (Integration Test)
- **Functions**: `get_subreddit_posts` → `get_post_details`
- **Task**: Find top post from r/technology and get its details
- **Validation**: 
  - Multi-tool workflow executed correctly
  - Complete post data (id, title, author, score, comments, url)
  - Post is from r/technology
  - Engagement metrics present
- **Category**: Integration test
- **Difficulty**: Medium

### Test 4: AI Subreddit Search (Edge Case)
- **Function**: `search_subreddits`
- **Task**: Find AI-related discussion communities
- **Validation**: 
  - Multiple relevant subreddits found (≥3)
  - Contains expected subs (r/artificial, r/MachineLearning, etc.)
  - Diverse results across different AI topics
  - Keywords: "AI", "artificial intelligence", "machine learning"
- **Category**: Unit test
- **Difficulty**: Medium

---

## Evaluation Metrics

The optimizer evaluates each agent configuration across 4 metrics:

### 1. Accuracy (40%) - Most Important
**What it measures**: Tool usage correctness and result relevance

**Components**:
- Correct tool(s) called (30%)
- Tool parameters correct (20%)
- Results contain expected data (30%)
- Keyword matching in results (20%)

**Perfect score**: Agent calls correct Reddit function with correct parameters and returns relevant results

### 2. Completeness (30%) - Second Most Important
**What it measures**: Data field presence and population

**Components**:
- Required fields present (70%)
- Optional fields present (30% bonus)
- Field values non-empty
- Data richness

**Perfect score**: All required fields present with valid values, plus many optional fields

### 3. Latency (20%)
**What it measures**: Response time

**Scoring**:
- < 5s: Excellent (1.0)
- < 10s: Good (0.8)
- < 20s: Acceptable (0.6)
- < 30s: Slow (0.4)
- ≥ 30s: Very slow (0.2)

**Perfect score**: Agent responds in under 5 seconds

### 4. Token Efficiency (10%)
**What it measures**: Cost-effectiveness (value per token)

**Scoring**:
- ≤ 70% of estimated: Excellent (1.0)
- ≤ 100% of estimated: Good (0.9)
- ≤ 130% of estimated: Acceptable (0.7)
- ≤ 150% of estimated: Verbose (0.5)
- > 150% of estimated: Very verbose (0.3)

**Perfect score**: Agent uses minimal tokens while delivering complete results

---

## MAB Architecture for Model Selection

The system uses **Thompson Sampling** (a Bayesian MAB strategy) to efficiently discover the best Azure model deployment:

### How It Works

1. **Exploration Phase**: Initially tries all Azure deployments equally
2. **Learning Phase**: Builds Bayesian belief about each model's performance
3. **Exploitation Phase**: Increasingly favors better-performing models
4. **Balance**: Continues exploring to avoid local optima

### Adding New Models

As you deploy new models to Azure AI Foundry, simply add them to the config:

```yaml
search_space:
  parameters:
    azure_deployment:
      values:
        - "o4-mini"                    # Current
        - "gpt-4o-mini-deployment"     # Add this
        - "gpt-4o-deployment"          # Add this
        - "claude-3-5-sonnet"          # Add this
```

The MAB algorithm will automatically:
- Test each new deployment
- Learn its performance characteristics
- Determine optimal use cases
- Recommend best deployment for Reddit tasks

### MAB Benefits

- **Efficient**: Fewer tests needed than exhaustive search
- **Adaptive**: Learns from each test
- **Robust**: Handles variability in API responses
- **Scalable**: Easy to add new models

---

## Results

Results are saved to `./results/reddit_agent_optimization/`:

### Generated Files

- **`best_config.json`**: Optimal agent configuration discovered
  - Best Azure deployment
  - Best temperature
  - Best max_completion_tokens
  - Best instruction_style
  - Best tool_strategy
  
- **`detailed_results.json`**: All experiments with full data
  - Every configuration tested
  - Score for each metric
  - Tool calls and responses
  
- **`report.md`**: Human-readable summary
  - Best configuration explanation
  - Performance comparison
  - Recommendations
  
- **`experiments.csv`**: Spreadsheet format for analysis
  - Easy filtering and sorting
  - Pivot tables in Excel/Google Sheets
  
- **`best_reddit_agent_config.py`**: Production-ready Python config

### Interpreting Results

**Best overall score** = Weighted average:
```
Score = (Accuracy × 0.40) + (Completeness × 0.30) + (Latency × 0.20) + (Token Efficiency × 0.10)
```

**Example winning config**:
```yaml
azure_deployment: "o4-mini"
temperature: 0.3
max_completion_tokens: 1000
instruction_style: "detailed"
tool_strategy: "include_all"

Final Score: 0.87
  - Accuracy: 0.92 (excellent)
  - Completeness: 0.85 (very good)
  - Latency: 0.80 (good)
  - Token Efficiency: 0.75 (acceptable)
```

---

## Troubleshooting

### "Reddit credentials not found"

```bash
# Check if environment variables are set
echo $REDDIT_CLIENT_ID
echo $REDDIT_CLIENT_SECRET

# If empty, set them
export REDDIT_CLIENT_ID="your_client_id"
export REDDIT_CLIENT_SECRET="your_client_secret"
```

### "Azure API call failed"

1. Verify your Azure API key: `echo $AZURE_API_KEY`
2. Check your endpoint URL in `reddit_agent_optimization.yaml`
3. Ensure deployment name matches your Azure deployment
4. Test with Azure OpenAI Studio first

### "Agno package not found"

```bash
pip install agno praw
```

### "Rate limit exceeded"

Reddit API allows 60 requests/minute. The config is set to 50 req/min by default.

To slow down further:
```yaml
rate_limiting:
  requests_per_minute: 30  # More conservative
```

### Test fails with empty results

Check that test expectations match reality:
1. Visit r/technology manually
2. Verify current subscriber count (test expects >10M)
3. Update test case expectations if needed

---

## Next Steps

### Phase 1: Validate Basic Functionality ✅
Run the current 4 test cases to ensure all tools work correctly with o4-mini.

### Phase 2: Add More Azure Models
Deploy additional models to Azure Foundry:
- gpt-4o-mini (balanced performance/cost)
- gpt-4o (best accuracy)
- claude-3-5-sonnet (alternative provider)

Update `reddit_agent_optimization.yaml` with new deployments.

### Phase 3: Expand Test Coverage
Add more test cases:
- Complex multi-tool workflows
- Edge cases (deleted posts, private subreddits)
- Different subreddits for diversity
- Write operations (create_post, create_comment) - requires auth

### Phase 4: Production Deployment
Use best configuration in production:
```python
from reddit_agent_runner import RedditAgentRunner

# Load best config
with open('results/best_reddit_agent_config.py') as f:
    best_params = eval(f.read())

# Create production agent
runner = RedditAgentRunner(config)
agent = runner.create_agent(best_params)

# Use in production
response = agent.print_response("Find AI discussions in r/technology")
```

---

## Architecture Notes

### File Structure
```
examples/agno_agents/reddit/
├── reddit_agent_optimization.yaml   # Main config (Azure + MAB)
├── reddit_test_cases.json          # 4 test cases with validation
├── reddit_evaluator.py             # Custom scoring (4 metrics)
├── reddit_agent_runner.py          # Agno wrapper (Azure support)
├── README.md                       # This file
└── results/                        # Auto-generated
    └── reddit_agent_optimization/
        ├── best_config.json
        ├── detailed_results.json
        ├── report.md
        └── experiments.csv
```

### Design Principles

1. **MAB-Ready**: Easy to add new Azure models - just update YAML
2. **Comprehensive Validation**: Schema, keywords, completeness, thresholds
3. **Production-Grade Evaluation**: Robust error handling, graceful degradation
4. **Extensible**: Easy to add more test cases, metrics, or tools
5. **Azure-First**: Designed for Azure AI Foundry workflows

---

## Support

- **Issues**: https://github.com/persist-os/the-convergence/issues
- **Documentation**: `the-convergence/README.md`
- **Reddit API**: https://www.reddit.com/dev/api
- **Agno Docs**: https://docs.agno.com/

---

## License

MIT License - see `the-convergence/LICENSE`

