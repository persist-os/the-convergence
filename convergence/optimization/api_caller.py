"""
Generic API caller for any HTTP/gRPC endpoint.
"""
import os
import time
import asyncio
import json
from typing import Dict, Any, Optional
from pathlib import Path
import httpx

from convergence.optimization.models import APIResponse

# Load environment variables from .env FIRST
try:
    from dotenv import load_dotenv
    # Try loading from multiple locations
    env_paths = [
        Path.cwd() / ".env",  # Current directory
        Path.home() / ".env",  # Home directory
        Path(__file__).parent.parent.parent / ".env",  # Project root (convergence/../.env)
    ]
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path, override=False)  # Don't override already-set vars
            break
    else:
        # Try without explicit path (searches up directory tree)
        load_dotenv(override=False)
except ImportError:
    # python-dotenv not installed, will use system env vars only
    pass

# Weave integration for observability
try:
    import weave
    WEAVE_AVAILABLE = True
except ImportError:
    WEAVE_AVAILABLE = False
    weave = None


class APICaller:
    """Makes calls to any API endpoint."""
    
    def __init__(self, timeout: int = 60):
        """
        Initialize API caller.
        
        Args:
            timeout: Default timeout in seconds
        """
        self.client = httpx.AsyncClient(timeout=timeout)
    
    async def call(
        self,
        endpoint: str,
        method: str,
        params: Dict[str, Any],
        auth: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
    ) -> APIResponse:
        """
        Make API call with given parameters.
        
        Args:
            endpoint: API endpoint URL
            method: HTTP method (GET, POST, etc.)
            params: Parameters to send (body for POST, query for GET)
            auth: Authentication config
            headers: Custom headers
            timeout: Override default timeout
            
        Returns:
            APIResponse with result, latency, cost, success flag
        """
        # Use Weave-tracked version if available
        if WEAVE_AVAILABLE and weave:
            return await self._call_with_weave(endpoint, method, params, auth, headers, timeout)
        else:
            return await self._call_impl(endpoint, method, params, auth, headers, timeout)
    
    async def _call_impl(
        self,
        endpoint: str,
        method: str,
        params: Dict[str, Any],
        auth: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
    ) -> APIResponse:
        """Internal implementation of API call."""
        start_time = time.time()
        
        try:
            # Build request headers
            request_headers = headers or {}
            
            # Handle authentication
            if auth:
                token = None
                if auth.get("type") == "bearer":
                    token_env = auth.get("token_env")
                    if token_env:
                        token = os.getenv(token_env)
                        if token:
                            request_headers["Authorization"] = f"Bearer {token}"
                
                elif auth.get("type") == "api_key":
                    key_env = auth.get("token_env")
                    if key_env:
                        api_key = os.getenv(key_env)
                        if api_key:
                            # Use or-coalescing to handle None values correctly
                            header_name = auth.get("header_name") or "x-api-key"
                            request_headers[header_name] = api_key
                        else:
                            # FAIL HARD: No API key found
                            raise ValueError(
                                f"Environment variable '{key_env}' not found!\n"
                                f"Please set {key_env} in your .env file or environment.\n"
                                f"Checked locations:\n"
                                f"  - {Path.cwd() / '.env'}\n"
                                f"  - {Path.home() / '.env'}\n"
                                f"  - Project root .env"
                            )
            
            # Make request
            method_upper = method.upper()
            
            if method_upper == "POST":
                response = await self.client.post(
                    endpoint,
                    json=params,
                    headers=request_headers,
                    timeout=timeout or 60.0
                )
            elif method_upper == "GET":
                response = await self.client.get(
                    endpoint,
                    params=params,
                    headers=request_headers,
                    timeout=timeout or 60.0
                )
            else:
                response = await self.client.request(
                    method_upper,
                    endpoint,
                    json=params if method_upper in ["POST", "PUT", "PATCH"] else None,
                    params=params if method_upper == "GET" else None,
                    headers=request_headers,
                    timeout=timeout or 60.0
                )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Parse response - FAIL HARD if not JSON
            try:
                response.raise_for_status()
                result = response.json()
            except httpx.HTTPStatusError as e:
                # Re-raise HTTP errors (will be caught by outer handler)
                raise
            except json.JSONDecodeError as e:
                # JSON parsing failed - this is a real error
                latency_ms = (time.time() - start_time) * 1000
                return APIResponse(
                    success=False,
                    result=None,
                    latency_ms=latency_ms,
                    error=f"Invalid JSON response: {response.text[:200]}"
                )
            
            return APIResponse(
                success=True,
                result=result,
                latency_ms=latency_ms,
                estimated_cost_usd=self._estimate_cost(result)
            )
            
        except httpx.TimeoutException as e:
            latency_ms = (time.time() - start_time) * 1000
            return APIResponse(
                success=False,
                result=None,
                latency_ms=latency_ms,
                error=f"Timeout: {str(e)}"
            )
        
        except httpx.HTTPStatusError as e:
            latency_ms = (time.time() - start_time) * 1000
            return APIResponse(
                success=False,
                result=None,
                latency_ms=latency_ms,
                error=f"HTTP {e.response.status_code}: {e.response.text}"
            )
        
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return APIResponse(
                success=False,
                result=None,
                latency_ms=latency_ms,
                error=f"Error: {str(e)}"
            )
    
    def _estimate_cost(self, result: Any) -> float:
        """
        Estimate cost from API response.
        
        For LLM APIs, looks for token usage.
        Can be extended for other API types.
        
        Args:
            result: API response result
            
        Returns:
            Estimated cost in USD
        """
        # Simple heuristic for LLM APIs
        if isinstance(result, dict):
            # OpenAI-style usage
            if "usage" in result:
                usage = result["usage"]
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                
                # Rough estimate for GPT-4 (adjust for your API)
                # GPT-4: $0.03/1K prompt, $0.06/1K completion
                cost = (prompt_tokens * 0.00003) + (completion_tokens * 0.00006)
                return cost
            
            # Anthropic-style usage
            if "usage" in result and "input_tokens" in result["usage"]:
                usage = result["usage"]
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
                
                # Claude: $0.015/1K input, $0.075/1K output
                cost = (input_tokens * 0.000015) + (output_tokens * 0.000075)
                return cost
        
        return 0.0
    
    async def _call_with_weave(
        self,
        endpoint: str,
        method: str,
        params: Dict[str, Any],
        auth: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
    ) -> APIResponse:
        """Weave-tracked version of API call."""
        if not WEAVE_AVAILABLE or not weave:
            return await self._call_impl(endpoint, method, params, auth, headers, timeout)
        
        # Decorate the implementation with proper parameters for Weave tracking
        @weave.op()
        async def tracked_call(
            endpoint: str,
            method: str,
            params: Dict[str, Any],
            auth: Optional[Dict[str, str]] = None,
            headers: Optional[Dict[str, str]] = None,
            timeout: Optional[int] = None,
        ):
            # Add metadata for better trace visibility
            weave_context = {
                "api_endpoint": endpoint,
                "http_method": method,
                "request_params": params,
                "auth_type": auth.get("type") if auth else None,
                "timeout": timeout,
                "timestamp": time.time()
            }
            
            result = await self._call_impl(endpoint, method, params, auth, headers, timeout)
            
            # Add response metadata
            weave_context.update({
                "success": result.success,
                "latency_ms": result.latency_ms,
                "cost_usd": result.estimated_cost_usd,
                "error": result.error
            })
            
            return result
        
        return await tracked_call(endpoint, method, params, auth, headers, timeout)
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

