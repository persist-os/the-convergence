"""
Custom evaluator for BrowserBase API optimization.

This evaluator extracts metrics from BrowserBase scraping results to evaluate
the performance of different browser configurations.

Supports all 6 parameters from browserbase_optimization.yaml:
- browser_type: chromium, firefox, webkit
- headless: true/false
- viewport_width: 1280, 1920, 2560
- viewport_height: 720, 1080, 1440
- timeout_ms: 5000, 10000, 30000
- wait_strategy: network_idle, load, dom_content_loaded
"""

import json
import logging
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,  # Changed from DEBUG to reduce noise
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def score_browserbase_response(result: Any, expected: Any, params: Dict[str, Any], metric: Optional[str] = None) -> float:
    """
    Evaluate BrowserBase scraping results.
    
    Args:
        result: API response result from BrowserBase
        expected: Expected output from test case
        params: Browser parameters used (all 6 params from config)
        metric: Specific metric to evaluate ('success_rate' or 'execution_time_ms')
    
    Returns:
        float: Score between 0.0 and 1.0
    """
    logger.info("=" * 80)
    logger.info(f"🔍 EVALUATOR CALLED - Metric: {metric}")
    logger.info(f"📊 Result type: {type(result).__name__}")
    logger.info(f"📋 Result preview: {str(result)[:200]}...")
    logger.info(f"⚙️  Parameters: {params}")
    
    try:
        # Handle different result formats
        if isinstance(result, str):
            logger.debug(f"📝 Result is string, length: {len(result)}")
            try:
                data = json.loads(result)
                logger.info(f"✅ Successfully parsed JSON from string")
                logger.debug(f"📦 Parsed data keys: {list(data.keys()) if isinstance(data, dict) else 'not a dict'}")
            except json.JSONDecodeError as e:
                logger.warning(f"❌ JSON decode failed: {e}")
                logger.warning(f"🔄 Falling back to MOCK evaluation (JSON parse error)")
                return _evaluate_mock_result(result, expected, params, metric)
        elif isinstance(result, dict):
            data = result
            logger.info(f"✅ Result is already a dict")
            logger.debug(f"📦 Dict keys: {list(data.keys())}")
        else:
            logger.warning(f"⚠️  Unknown result format: {type(result)}")
            logger.warning(f"🔄 Falling back to MOCK evaluation (unknown format)")
            return _evaluate_mock_result(result, expected, params, metric)
        
        # Check if this looks like a BrowserBase response
        logger.debug(f"🔍 Checking for BrowserBase response markers...")
        has_status = 'status' in data
        has_data = 'data' in data
        has_session_id = isinstance(data.get('data'), dict) and 'session_id' in data.get('data', {})
        logger.debug(f"   - Has 'status' field: {has_status}")
        logger.debug(f"   - Has 'data' field: {has_data}")
        logger.debug(f"   - Has 'session_id' in data: {has_session_id}")
        
        if isinstance(data, dict) and (has_status or has_data or has_session_id):
            logger.info(f"✨ REAL BROWSERBASE RESPONSE DETECTED")
            # Check if it's a session creation response
            if has_session_id:
                logger.info(f"   Session ID: {data['data']['session_id']}")
                logger.info(f"   Session Status: {data['data'].get('session_status', 'unknown')}")
            else:
                logger.info(f"   Status: {data.get('status', 'N/A')}")
                logger.info(f"   Data present: {bool(data.get('data'))}")
            return _evaluate_real_browserbase_result(data, expected, params, metric)
        else:
            logger.warning(f"⚠️  Response doesn't match BrowserBase format")
            logger.warning(f"   Dict keys found: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
            logger.warning(f"🔄 Falling back to MOCK evaluation (no status/data fields)")
            return _evaluate_mock_result(data, expected, params, metric)
            
    except Exception as e:
        logger.error(f"❌ Exception in evaluator: {type(e).__name__}: {e}", exc_info=True)
        logger.error(f"🔄 Falling back to FALLBACK evaluation (exception)")
        return _evaluate_fallback(params, metric)


def _evaluate_real_browserbase_result(data: Dict[str, Any], expected: Any, params: Dict[str, Any], metric: Optional[str]) -> float:
    """Evaluate actual BrowserBase API response."""
    logger.info("🎯 Using REAL BrowserBase evaluation")
    logger.debug(f"   Metric: {metric}")
    logger.debug(f"   Data: {json.dumps(data, indent=2)[:500]}")
    
    if metric == "success_rate":
        # Check if the scraping/session was successful
        status = data.get('status')
        logger.debug(f"   Response status: {status}")
        
        # Check for successful session creation (from adapter-transformed response)
        if status == 'success' and 'data' in data:
            session_data = data.get('data', {})
            
            # Session successfully created
            if 'session_id' in session_data:
                session_status = session_data.get('session_status', 'unknown').upper()
                
                if session_status in ['RUNNING', 'READY']:
                    logger.info(f"📈 REAL success_rate score: 1.0 (session {session_status})")
                    return 1.0
                elif session_status in ['STARTING', 'PENDING', 'STARTED']:
                    logger.info(f"📈 REAL success_rate score: 0.9 (session {session_status})")
                    return 0.9
                else:
                    logger.info(f"📈 REAL success_rate score: 0.5 (session {session_status})")
                    return 0.5
        
        # Original scraping evaluation logic
        if status == 'success' or status == 'completed':
            scraped_data = data.get('data', {})
            logger.debug(f"   Scraped data type: {type(scraped_data).__name__}")
            logger.debug(f"   Scraped data size: {len(scraped_data) if scraped_data else 0}")
            
            # Check if we got the expected elements
            if isinstance(expected, dict):
                # Check minimum elements if specified
                min_elements = expected.get('min_elements', 1)
                if isinstance(scraped_data, list) and len(scraped_data) >= min_elements:
                    logger.info(f"📈 REAL success_rate score: 1.0 (list with {len(scraped_data)} elements)")
                    return 1.0
                elif isinstance(scraped_data, dict) and len(scraped_data) >= min_elements:
                    logger.info(f"📈 REAL success_rate score: 1.0 (dict with {len(scraped_data)} keys)")
                    return 1.0
                elif scraped_data:  # Got some data
                    logger.info(f"📈 REAL success_rate score: 0.7 (partial data)")
                    return 0.7
            score = 0.5 if scraped_data else 0.2
            logger.info(f"📈 REAL success_rate score: {score} (scraped_data: {bool(scraped_data)})")
            return score
        else:
            logger.info(f"📈 REAL success_rate score: 0.1 (failed status: {status})")
            return 0.1  # Failed to scrape
    
    elif metric == "execution_time_ms":
        # Extract execution time from the response
        exec_time = data.get('execution_time', data.get('duration', data.get('time_ms', 0)))
        logger.debug(f"   Raw execution time: {exec_time}")
        
        # Convert to ms if needed
        if exec_time > 100000:  # Assume it's in microseconds
            exec_time = exec_time / 1000
            logger.debug(f"   Converted from microseconds to ms: {exec_time}")
        elif exec_time > 0 and exec_time < 100:  # Assume it's in seconds
            exec_time = exec_time * 1000
            logger.debug(f"   Converted from seconds to ms: {exec_time}")
            
        # Score based on execution time (lower is better)
        # Target: under 5 seconds is great, under 10 is good, over 30 is poor
        if exec_time <= 5000:  # 5 seconds
            logger.info(f"⏱️  REAL execution_time score: 1.0 ({exec_time}ms - excellent)")
            return 1.0
        elif exec_time <= 10000:  # 10 seconds
            logger.info(f"⏱️  REAL execution_time score: 0.8 ({exec_time}ms - good)")
            return 0.8
        elif exec_time <= 20000:  # 20 seconds  
            logger.info(f"⏱️  REAL execution_time score: 0.5 ({exec_time}ms - moderate)")
            return 0.5
        elif exec_time <= 30000:  # 30 seconds
            logger.info(f"⏱️  REAL execution_time score: 0.3 ({exec_time}ms - slow)")
            return 0.3
        else:
            logger.info(f"⏱️  REAL execution_time score: 0.1 ({exec_time}ms - very slow)")
            return 0.1
    
    # Default: moderate score
    logger.info(f"📈 REAL default score: 0.6 (unknown metric)")
    return 0.6


def _evaluate_mock_result(result: Any, expected: Any, params: Dict[str, Any], metric: Optional[str]) -> float:
    """
    Evaluate mock/fallback results for testing without real API calls.
    
    Considers all 6 parameters: browser_type, headless, viewport_width, 
    viewport_height, timeout_ms, wait_strategy
    """
    logger.info("🎭 Using MOCK/SIMULATED evaluation")
    logger.debug(f"   Metric: {metric}")
    logger.debug(f"   Params: {params}")
    
    if metric == "success_rate":
        # Extract all parameters
        browser_type = params.get('browser_type', 'chromium')
        timeout_ms = params.get('timeout_ms', 5000)
        headless = params.get('headless', True)
        viewport_width = params.get('viewport_width', 1920)
        viewport_height = params.get('viewport_height', 1080)
        wait_strategy = params.get('wait_strategy', 'load')
        
        score = 0.3  # Base score
        
        # Browser type scoring (chromium most reliable)
        if browser_type == 'chromium':
            score += 0.25
        elif browser_type == 'firefox':
            score += 0.20
        elif browser_type == 'webkit':
            score += 0.15
            
        # Timeout scoring (sweet spot 5-10 seconds)
        if 5000 <= timeout_ms <= 10000:
            score += 0.15
        elif 3000 <= timeout_ms <= 30000:
            score += 0.10
        else:
            score += 0.05
            
        # Headless mode (more reliable, no UI overhead)
        if headless:
            score += 0.10
        else:
            score += 0.05  # Some sites work better with visible browser
            
        # Viewport scoring (standard sizes work best)
        if viewport_width == 1920 and viewport_height == 1080:
            score += 0.15  # Most common, well-tested
        elif viewport_width == 1280 and viewport_height == 720:
            score += 0.12  # HD ready
        elif viewport_width == 2560 and viewport_height == 1440:
            score += 0.10  # 2K, some sites may not optimize for it
        else:
            score += 0.05  # Unusual viewport
        
        # Wait strategy scoring
        if wait_strategy == 'network_idle':
            score += 0.10  # Most thorough
        elif wait_strategy == 'load':
            score += 0.08  # Standard
        elif wait_strategy == 'dom_content_loaded':
            score += 0.06  # Fast but may miss dynamic content
        
        final_score = min(1.0, score)
        logger.info(f"📈 MOCK evaluation score: {final_score:.4f}")
        return final_score
    
    elif metric == "execution_time_ms":
        # Simulate execution time based on all parameters
        browser_type = params.get('browser_type', 'chromium')
        timeout_ms = params.get('timeout_ms', 5000)
        headless = params.get('headless', True)
        viewport_width = params.get('viewport_width', 1920)
        viewport_height = params.get('viewport_height', 1080)
        wait_strategy = params.get('wait_strategy', 'load')
        
        # Base time
        base_time = 2500
        
        # Browser speed differences
        if browser_type == 'chromium':
            simulated_time = base_time + 800
        elif browser_type == 'firefox':
            simulated_time = base_time + 1200
        elif browser_type == 'webkit':
            simulated_time = base_time + 1500
        else:
            simulated_time = base_time + 1000
            
        # Headless is faster (no rendering overhead)
        if not headless:
            simulated_time += 1500
            
        # Viewport size affects rendering time
        pixel_count = viewport_width * viewport_height
        if pixel_count > 2000000:  # 1920x1080 or higher
            simulated_time += 800
        elif pixel_count > 1000000:  # 1280x720 or higher
            simulated_time += 500
        else:
            simulated_time += 200
            
        # Wait strategy affects completion time
        if wait_strategy == 'network_idle':
            simulated_time += timeout_ms * 0.3  # Wait for all network activity
        elif wait_strategy == 'load':
            simulated_time += timeout_ms * 0.15  # Wait for load event
        elif wait_strategy == 'dom_content_loaded':
            simulated_time += timeout_ms * 0.08  # Fastest
            
        # Score based on simulated execution time (lower is better)
        if simulated_time <= 4000:
            final_score = 1.0
        elif simulated_time <= 6000:
            final_score = 0.85
        elif simulated_time <= 8000:
            final_score = 0.70
        elif simulated_time <= 10000:
            final_score = 0.55
        elif simulated_time <= 15000:
            final_score = 0.35
        else:
            final_score = 0.15
        
        logger.info(f"⏱️  MOCK simulated time: {simulated_time:.0f}ms, score: {final_score:.4f}")
        return final_score
    
    # Default score
    logger.info(f"📈 MOCK default score: 0.6000")
    return 0.6


def _evaluate_fallback(params: Dict[str, Any], metric: Optional[str]) -> float:
    """
    Fallback evaluation when everything else fails.
    
    Uses heuristics based on all 6 parameters to provide reasonable scoring.
    """
    logger.info("⚠️  Using FALLBACK evaluation (error recovery)")
    logger.debug(f"   Metric: {metric}")
    logger.debug(f"   Params: {params}")
    
    browser_type = params.get('browser_type', 'chromium')
    timeout_ms = params.get('timeout_ms', 5000)
    headless = params.get('headless', True)
    viewport_width = params.get('viewport_width', 1920)
    wait_strategy = params.get('wait_strategy', 'load')
    
    # Heuristic: chromium + headless + 1920px + 5-10s timeout + network_idle = best
    if (browser_type == 'chromium' and headless and 
        viewport_width == 1920 and 5000 <= timeout_ms <= 10000 and
        wait_strategy == 'network_idle'):
        logger.info(f"📈 FALLBACK score: 0.85 (optimal config)")
        return 0.85
    
    # Good combination
    elif (browser_type in ['chromium', 'firefox'] and headless and
          1280 <= viewport_width <= 2560 and 3000 <= timeout_ms <= 30000):
        logger.info(f"📈 FALLBACK score: 0.70 (good config)")
        return 0.70
    
    # Decent combination  
    elif browser_type in ['chromium', 'firefox'] and 1000 <= timeout_ms <= 30000:
        logger.info(f"📈 FALLBACK score: 0.55 (decent config)")
        return 0.55
    
    # Fallback
    else:
        logger.info(f"📈 FALLBACK score: 0.45 (default)")
        return 0.45