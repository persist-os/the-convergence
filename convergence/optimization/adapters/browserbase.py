"""BrowserBase API adapter."""
import os
import asyncio
from typing import Dict, Any, Optional
from . import APIAdapter
from ..models import APIResponse
import httpx


class BrowserBaseAdapter(APIAdapter):
    """Transforms generic optimization into BrowserBase session API."""
    
    def __init__(self):
        """
        Initialize BrowserBase adapter.
        
        Raises:
            ValueError: If BROWSERBASE_PROJECT_ID env var is not set
        """
        self.project_id = os.getenv("BROWSERBASE_PROJECT_ID")
        if not self.project_id:
            raise ValueError(
                "BROWSERBASE_PROJECT_ID environment variable required "
                "for BrowserBase optimization. Please set it to your BrowserBase project ID."
            )
    
    def transform_request(
        self,
        optimization_params: Dict[str, Any],
        test_case: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Transform params into BrowserBase session request.
        
        Converts optimization parameters to BrowserBase /v1/sessions API format:
        - browser_type → browserSettings.browserType
        - headless → browserSettings.headless
        - viewport_width/height → browserSettings.viewport.width/height
        - timeout_ms → browserSettings.timeout (converted to seconds)
        
        Args:
            optimization_params: Optimization parameters being tested
            test_case: Test case (not used for session creation, but available)
            
        Returns:
            BrowserBase session creation payload
        """
        # Build browserSettings from optimization params
        browser_settings = {
            "browserType": optimization_params.get("browser_type", "chromium"),
            "headless": optimization_params.get("headless", True),
            "viewport": {
                "width": optimization_params.get("viewport_width", 1920),
                "height": optimization_params.get("viewport_height", 1080)
            }
        }
        
        # Add timeout if provided (convert ms to seconds for BrowserBase)
        timeout_ms = optimization_params.get("timeout_ms")
        if timeout_ms:
            browser_settings["timeout"] = timeout_ms // 1000
        
        # Build final request with projectId and browserSettings
        return {
            "projectId": self.project_id,
            "browserSettings": browser_settings
        }
    
    def transform_response(
        self,
        api_response: APIResponse,
        optimization_params: Dict[str, Any]
    ) -> APIResponse:
        """
        Transform session response for evaluator.
        
        Converts BrowserBase session creation response to evaluator-friendly format:
        - {id: "session_xxx", status: "running", ...}
        → {status: "success", data: {session_id: "...", ...}, metadata: {...}}
        
        Args:
            api_response: Raw BrowserBase API response
            optimization_params: Optimization parameters used in request
            
        Returns:
            Transformed APIResponse for evaluator
        """
        if not api_response.success:
            return api_response
        
        # Session response: {id: "session_xxx", status: "running", ...}
        result = api_response.result
        
        if isinstance(result, dict) and "id" in result:
            # Transform to evaluator-friendly format
            # BrowserBase API returns status in UPPERCASE (e.g., "RUNNING")
            raw_status = result.get("status", "").upper()
            
            # Session is successful if it's running, pending, or started
            is_success = raw_status in ["RUNNING", "PENDING", "STARTED", "READY"]
            
            transformed_result = {
                "status": "success" if is_success else "failed",
                "data": {
                    "session_id": result.get("id"),
                    "browser_type": optimization_params.get("browser_type"),
                    "session_status": raw_status,  # Store uppercase for consistency
                    "created_at": result.get("createdAt")
                },
                "metadata": {
                    "browserbase_session": True,
                    "original_response": result
                }
            }
            
            return APIResponse(
                success=True,
                result=transformed_result,
                latency_ms=api_response.latency_ms,
                estimated_cost_usd=api_response.estimated_cost_usd
            )
        
        return api_response
    
    async def cleanup_session(
        self,
        api_response: APIResponse,
        optimization_params: Dict[str, Any]
    ) -> bool:
        """
        Clean up BrowserBase session after evaluation.
        
        Deletes the session to free up concurrent session slots.
        This prevents hitting the 25 concurrent session limit.
        
        Args:
            api_response: The transformed API response containing session_id
            optimization_params: Optimization parameters (unused but kept for consistency)
            
        Returns:
            bool: True if cleanup succeeded, False otherwise
        """
        try:
            # Extract session_id from transformed response
            if not api_response.success or not api_response.result:
                return False
            
            result = api_response.result
            
            # Get session_id from the data field
            if isinstance(result, dict) and 'data' in result:
                session_data = result.get('data', {})
                session_id = session_data.get('session_id')
                
                if not session_id:
                    return False
                
                # Get API key from environment
                api_key = os.getenv("BROWSERBASE_API_KEY")
                if not api_key:
                    return False
                
                # Delete the session
                delete_url = f"https://api.browserbase.com/v1/sessions/{session_id}"
                
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.delete(
                        delete_url,
                        headers={"x-bb-api-key": api_key}
                    )
                    
                    if response.status_code in [200, 204, 404]:
                        # 200/204: Successfully deleted
                        # 404: Already deleted/expired (still success)
                        return True
                    else:
                        # Log but don't fail
                        print(f"   [dim]⚠️  Session cleanup warning: HTTP {response.status_code}[/dim]")
                        return False
            
            return False
            
        except httpx.TimeoutException:
            # Timeout is not critical - session will auto-expire
            print(f"   [dim]⚠️  Session cleanup timeout (session will auto-expire)[/dim]")
            return False
        except Exception as e:
            # Any other error - log but don't fail the optimization
            print(f"   [dim]⚠️  Session cleanup error: {e}[/dim]")
            return False

