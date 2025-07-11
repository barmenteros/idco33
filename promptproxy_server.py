"""
PromptProxy HTTP Server with MQL5 Detection, AWS RAG Integration, and Full Circuit-Breaker Logic
Task D24: Complete implementation with robust fallback & circuit-breaker logic
Module: PromptProxy

HTTP server that acts as a local proxy between Claude Desktop CLI
and the MQL5 RAG system. Includes intelligent MQL5 prompt detection,
AWS RAG endpoint integration, prompt augmentation, and production-ready
circuit-breaker pattern with comprehensive fallback mechanisms.
"""

import json
import logging
import time
import asyncio
import random
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from enum import Enum

import httpx
import yaml
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel, Field

# Import MQL5 prompt detection from Task D19
try:
    from prompt_detector import is_mql5_prompt, get_mql5_detection_details
    PROMPT_DETECTOR_AVAILABLE = True
    logging.getLogger(__name__).info("✅ MQL5 PromptDetector imported successfully")
except ImportError as e:
    PROMPT_DETECTOR_AVAILABLE = False
    logging.getLogger(__name__).warning(f"⚠️ MQL5 PromptDetector not available: {e}")
    
    # Fallback function for when detector is not available
    def is_mql5_prompt(text: str) -> bool:
        return False
    
    def get_mql5_detection_details(text: str) -> dict:
        return {'is_mql5': False, 'error': 'PromptDetector not available'}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states for robust failure handling."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # All requests blocked
    HALF_OPEN = "half_open" # Testing recovery


class ErrorType(Enum):
    """Classification of error types for better handling."""
    TIMEOUT = "timeout"
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    SERVER_ERROR = "server_error"
    UNKNOWN = "unknown"


class PromptRequest(BaseModel):
    """Request model for prompt processing."""
    prompt: str = Field(..., description="The user's prompt text")
    user: Optional[str] = Field(default=None, description="User identifier")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")


class PromptResponse(BaseModel):
    """Response model for processed prompts."""
    success: bool = Field(..., description="Whether processing was successful")
    prompt: str = Field(..., description="Original or augmented prompt")
    augmented: bool = Field(default=False, description="Whether prompt was augmented with context")
    mql5_detected: bool = Field(default=False, description="Whether MQL5 content was detected")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional response metadata")
    error: Optional[str] = Field(default=None, description="Error message if processing failed")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Current timestamp")
    version: str = Field(..., description="Service version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class CircuitBreakerMetrics(BaseModel):
    """Circuit breaker metrics for monitoring."""
    state: str = Field(..., description="Current circuit breaker state")
    failure_count: int = Field(..., description="Current failure count")
    success_count: int = Field(..., description="Success count since last reset")
    last_failure_time: Optional[str] = Field(default=None, description="Last failure timestamp")
    last_success_time: Optional[str] = Field(default=None, description="Last success timestamp")
    total_requests: int = Field(..., description="Total requests processed")
    total_failures: int = Field(..., description="Total failures recorded")
    circuit_open_count: int = Field(..., description="Times circuit has opened")
    average_response_time_ms: float = Field(..., description="Average response time")


class PromptProxyServer:
    """
    HTTP server that provides the PromptProxy functionality.
    Acts as a local middleware between Claude Desktop CLI and the RAG system.
    Includes intelligent MQL5 prompt detection, AWS RAG integration, prompt augmentation,
    and production-ready circuit-breaker pattern with comprehensive fallback mechanisms.
    """
    
    def __init__(self, host: str = "localhost", port: int = 8080, config: Optional[Dict] = None):
        self.host = host
        self.port = port
        self.config = config or {}
        self.app = FastAPI(
            title="MQL5 PromptProxy Server",
            description="Local proxy server for MQL5 prompt enrichment with AWS RAG integration and circuit-breaker",
            version="1.4.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        self.start_time = time.time()
        
        # AWS RAG configuration
        self.aws_config = self.config.get('aws', {})
        self.api_gateway_url = self.aws_config.get('api_gateway_url', '')
        self.api_key = self.aws_config.get('api_key', '')
        self.rag_timeout = self.aws_config.get('timeout_seconds', 2.0)
        
        # Augmentation configuration
        self.augmentation_config = self.config.get('augmentation', {})
        self.max_snippets = self.augmentation_config.get('max_snippets', 5)
        self.snippet_separator = self.augmentation_config.get('snippet_separator', '\n\n')
        
        # Circuit breaker configuration (Task D24)
        self.circuit_config = self.config.get('circuit_breaker', {})
        self.failure_threshold = self.circuit_config.get('failure_threshold', 3)
        self.success_threshold = self.circuit_config.get('success_threshold', 2)
        self.cooldown_duration = self.circuit_config.get('cooldown_duration', 300)  # 5 minutes
        self.half_open_max_requests = self.circuit_config.get('half_open_max_requests', 3)
        
        # Retry configuration
        self.retry_config = self.config.get('retry', {})
        self.max_retries = self.retry_config.get('max_retries', 1)
        self.retry_delay_ms = self.retry_config.get('delay_ms', 100)
        self.retry_backoff_multiplier = self.retry_config.get('backoff_multiplier', 2.0)
        
        # Circuit breaker state (Task D24)
        self.circuit_state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.circuit_open_time = 0
        self.half_open_requests = 0
        self.total_requests = 0
        self.total_failures = 0
        self.circuit_open_count = 0
        self.last_failure_time = None
        self.last_success_time = None
        self.response_times = []
        
        # HTTP client for AWS calls
        self.http_client = None
        
        # Add CORS middleware for Claude Desktop CLI compatibility
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
        )
        
        self._setup_routes()
        self._setup_middleware()
        
        logger.info(f"PromptProxy server initialized on {self.host}:{self.port}")
        logger.info(f"AWS RAG endpoint: {self.api_gateway_url or 'Not configured'}")
        logger.info(f"RAG timeout: {self.rag_timeout}s")
        logger.info(f"Max snippets for augmentation: {self.max_snippets}")
        logger.info(f"Circuit breaker: {self.failure_threshold} failures, {self.cooldown_duration}s cooldown")
        logger.info(f"Retry policy: {self.max_retries} retries, {self.retry_delay_ms}ms delay")
    
    def _setup_routes(self):
        """Set up HTTP routes for the proxy server."""
        
        @self.app.get("/", response_model=Dict[str, str])
        async def root():
            """Root endpoint with basic service information."""
            return {
                "service": "MQL5 PromptProxy",
                "status": "active",
                "version": "1.4.0",
                "mql5_detector": "available" if PROMPT_DETECTOR_AVAILABLE else "unavailable",
                "aws_rag": "configured" if self.api_gateway_url else "not_configured",
                "augmentation": "enabled",
                "circuit_breaker": self.circuit_state.value,
                "endpoints": {
                    "health": "/health",
                    "process": "/process",
                    "rag": "/rag",
                    "detect": "/detect",
                    "config": "/config",
                    "circuit": "/circuit",
                    "docs": "/docs"
                }
            }
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint for monitoring."""
            uptime = time.time() - self.start_time
            return HealthResponse(
                status="healthy",
                timestamp=datetime.now().isoformat(),
                version="1.4.0",
                uptime_seconds=round(uptime, 2)
            )
        
        @self.app.get("/circuit", response_model=CircuitBreakerMetrics)
        async def circuit_metrics():
            """Circuit breaker metrics endpoint for monitoring."""
            avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0.0
            
            return CircuitBreakerMetrics(
                state=self.circuit_state.value,
                failure_count=self.failure_count,
                success_count=self.success_count,
                last_failure_time=self.last_failure_time,
                last_success_time=self.last_success_time,
                total_requests=self.total_requests,
                total_failures=self.total_failures,
                circuit_open_count=self.circuit_open_count,
                average_response_time_ms=round(avg_response_time, 2)
            )
        
        @self.app.post("/process", response_model=PromptResponse)
        async def process_prompt(request: PromptRequest):
            """
            Main endpoint for prompt processing with MQL5 detection, AWS RAG integration,
            augmentation template application, and full circuit-breaker protection.
            Task D24: Complete circuit-breaker and fallback implementation.
            """
            start_time = time.time()
            
            try:
                logger.info(f"Received prompt processing request: {request.prompt[:100]}...")
                
                # Step 1: Detect if the prompt is MQL5-related (Task D21)
                is_mql5 = is_mql5_prompt(request.prompt)
                detection_details = None
                
                if PROMPT_DETECTOR_AVAILABLE:
                    detection_details = get_mql5_detection_details(request.prompt)
                    logger.info(f"MQL5 detection: {is_mql5} (confidence: {detection_details.get('confidence', 'N/A')})")
                else:
                    logger.warning("MQL5 detection unavailable - treating as non-MQL5")
                
                # Step 2: Conditional processing based on detection
                if is_mql5:
                    logger.info("🎯 MQL5 prompt detected - attempting RAG with circuit-breaker protection")
                    
                    # Task D24: Call AWS RAG with full circuit-breaker protection
                    try:
                        rag_response = await self._call_aws_rag_with_circuit_breaker(request.prompt)
                        if rag_response and rag_response.get('success') and rag_response.get('snippets'):
                            # Task D23: Apply augmentation template to retrieved snippets
                            augmented_prompt = self._apply_augmentation_template(
                                original_prompt=request.prompt,
                                snippets=rag_response.get('snippets', [])
                            )
                            
                            processed_prompt = augmented_prompt
                            augmented = True
                            
                            # Store RAG results
                            metadata = {
                                "rag_results": {
                                    "snippets_count": len(rag_response.get('snippets', [])),
                                    "retrieval_time_ms": rag_response.get('processing_time_ms', 0),
                                    "lambda_success": True,
                                    "augmentation_applied": True,
                                    "original_prompt_length": len(request.prompt),
                                    "augmented_prompt_length": len(augmented_prompt),
                                    "circuit_breaker_state": self.circuit_state.value
                                }
                            }
                            logger.info(f"✅ RAG retrieval and augmentation successful: {len(rag_response.get('snippets', []))} snippets")
                            logger.info(f"📝 Prompt augmented: {len(request.prompt)} → {len(augmented_prompt)} chars")
                            
                        else:
                            logger.warning("⚠️ RAG endpoint returned no snippets - using fallback")
                            processed_prompt = request.prompt
                            augmented = False
                            metadata = {
                                "rag_results": {
                                    "lambda_success": rag_response.get('success', False) if rag_response else False,
                                    "augmentation_applied": False,
                                    "fallback_reason": "no_snippets",
                                    "circuit_breaker_state": self.circuit_state.value
                                }
                            }
                    
                    except Exception as rag_error:
                        logger.warning(f"⚠️ RAG system failed: {rag_error} - using fallback")
                        processed_prompt = request.prompt
                        augmented = False
                        metadata = {
                            "rag_results": {
                                "lambda_success": False, 
                                "augmentation_applied": False,
                                "fallback_reason": "circuit_breaker_protection",
                                "circuit_breaker_state": self.circuit_state.value,
                                "error": str(rag_error)
                            }
                        }
                
                else:
                    logger.info("📝 Non-MQL5 prompt - passing through unchanged")
                    processed_prompt = request.prompt
                    augmented = False
                    metadata = {
                        "augmentation_applied": False, 
                        "reason": "non_mql5_prompt",
                        "circuit_breaker_state": self.circuit_state.value
                    }
                
                processing_time = (time.time() - start_time) * 1000
                
                # Build response with detection results
                base_metadata = {
                    "user": request.user,
                    "session_id": request.session_id,
                    "timestamp": datetime.now().isoformat(),
                    "server_version": "1.4.0",
                    "detector_available": PROMPT_DETECTOR_AVAILABLE,
                    "detection_method": "integrated" if PROMPT_DETECTOR_AVAILABLE else "fallback",
                    "circuit_breaker_state": self.circuit_state.value,
                    "total_requests": self.total_requests,
                    "total_failures": self.total_failures
                }
                
                # Merge metadata
                base_metadata.update(metadata)
                
                # Add detection details if available
                if detection_details:
                    base_metadata["detection_details"] = {
                        "confidence": detection_details.get('confidence', 0.0),
                        "matched_keywords": detection_details.get('matched_keywords', []),
                        "matched_patterns": len(detection_details.get('matched_patterns', [])),
                        "matched_contexts": len(detection_details.get('matched_contexts', []))
                    }
                
                response = PromptResponse(
                    success=True,
                    prompt=processed_prompt,
                    augmented=augmented,
                    mql5_detected=is_mql5,
                    processing_time_ms=round(processing_time, 2),
                    metadata=base_metadata
                )
                
                logger.info(f"Prompt processed successfully in {processing_time:.2f}ms (augmented: {augmented})")
                return response
                
            except Exception as e:
                processing_time = (time.time() - start_time) * 1000
                logger.error(f"Error processing prompt: {e}")
                
                return PromptResponse(
                    success=False,
                    prompt=request.prompt,
                    augmented=False,
                    mql5_detected=False,
                    processing_time_ms=round(processing_time, 2),
                    error=str(e),
                    metadata={
                        "timestamp": datetime.now().isoformat(),
                        "error_type": type(e).__name__,
                        "detector_available": PROMPT_DETECTOR_AVAILABLE,
                        "circuit_breaker_state": self.circuit_state.value
                    }
                )
        
        @self.app.get("/detect", response_model=Dict[str, Any])
        async def detect_mql5(prompt: str):
            """
            Endpoint for testing MQL5 detection functionality.
            Useful for debugging and validation.
            """
            try:
                if not PROMPT_DETECTOR_AVAILABLE:
                    return {
                        "error": "PromptDetector not available",
                        "available": False
                    }
                
                is_mql5 = is_mql5_prompt(prompt)
                details = get_mql5_detection_details(prompt)
                
                return {
                    "prompt": prompt,
                    "is_mql5": is_mql5,
                    "detection_details": details,
                    "detector_available": True
                }
                
            except Exception as e:
                logger.error(f"Detection test failed: {e}")
                return {
                    "error": str(e),
                    "available": PROMPT_DETECTOR_AVAILABLE
                }
        
        @self.app.get("/config", response_model=Dict[str, Any])
        async def get_config():
            """
            Endpoint for checking configuration status.
            Useful for debugging and validation.
            """
            return {
                "aws_configured": bool(self.api_gateway_url and self.api_key),
                "api_gateway_url": self.api_gateway_url[:50] + "..." if len(self.api_gateway_url) > 50 else self.api_gateway_url,
                "api_key_configured": bool(self.api_key),
                "timeout_seconds": self.rag_timeout,
                "augmentation": {
                    "max_snippets": self.max_snippets,
                    "snippet_separator": repr(self.snippet_separator)
                },
                "circuit_breaker": {
                    "state": self.circuit_state.value,
                    "failure_threshold": self.failure_threshold,
                    "success_threshold": self.success_threshold,
                    "cooldown_duration": self.cooldown_duration,
                    "half_open_max_requests": self.half_open_max_requests,
                    "current_failures": self.failure_count,
                    "current_successes": self.success_count
                },
                "retry": {
                    "max_retries": self.max_retries,
                    "delay_ms": self.retry_delay_ms,
                    "backoff_multiplier": self.retry_backoff_multiplier
                },
                "detector_available": PROMPT_DETECTOR_AVAILABLE
            }
        
        @self.app.post("/rag", response_model=PromptResponse)
        async def rag_endpoint(request: PromptRequest):
            """
            Alternative endpoint that matches the API Gateway path.
            Provides compatibility with the AWS Lambda RAG endpoint structure.
            """
            # Delegate to the main process endpoint
            return await process_prompt(request)
    
    def _setup_middleware(self):
        """Set up middleware for logging and monitoring."""
        
        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            """Log all incoming requests."""
            start_time = time.time()
            
            # Log request details
            logger.info(
                f"Request: {request.method} {request.url.path} "
                f"from {request.client.host if request.client else 'unknown'}"
            )
            
            # Process request
            response = await call_next(request)
            
            # Log response details
            processing_time = (time.time() - start_time) * 1000
            logger.info(
                f"Response: {response.status_code} "
                f"processed in {processing_time:.2f}ms"
            )
            
            return response
    
    def _apply_augmentation_template(self, original_prompt: str, snippets: List[Dict[str, Any]]) -> str:
        """
        Apply augmentation template to retrieved snippets.
        
        Task D23: Core augmentation template functionality.
        
        Args:
            original_prompt: The user's original prompt
            snippets: List of retrieved documentation snippets
            
        Returns:
            Augmented prompt with context snippets
        """
        if not snippets:
            logger.warning("No snippets provided for augmentation")
            return original_prompt
        
        # Limit snippets to configured maximum
        limited_snippets = snippets[:self.max_snippets]
        
        # Format snippets according to augmentation template
        context_lines = ["/* Context: MQL5 documentation snippets */"]
        
        for i, snippet in enumerate(limited_snippets, 1):
            snippet_text = snippet.get('snippet', '').strip()
            source = snippet.get('source', 'Unknown source')
            score = snippet.get('score', 0.0)
            
            if snippet_text:  # Only include snippets with actual content
                # Add snippet with source attribution
                context_lines.append(f"// Snippet {i} (relevance: {score:.2f}) - {source}")
                context_lines.append(snippet_text)
                
                # Add separator between snippets (except last one)
                if i < len(limited_snippets):
                    context_lines.append("")
        
        # Build the final augmented prompt
        augmented_prompt = "\n".join(context_lines)
        augmented_prompt += "\n\n/* Original Prompt */\n"
        augmented_prompt += original_prompt
        
        logger.info(f"🔧 Applied augmentation template: {len(limited_snippets)} snippets, {len(augmented_prompt)} chars")
        
        return augmented_prompt

    async def _call_aws_rag_with_circuit_breaker(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Call AWS RAG endpoint with full circuit-breaker protection.
        
        Task D24: Complete circuit-breaker implementation with all states.
        
        Args:
            prompt: The user's prompt text
            
        Returns:
            Dictionary with RAG results or None if circuit is open/failed
        """
        self.total_requests += 1
        
        # Check circuit breaker state
        if self.circuit_state == CircuitBreakerState.OPEN:
            if time.time() - self.circuit_open_time >= self.cooldown_duration:
                # Transition to HALF_OPEN
                self._transition_to_half_open()
            else:
                logger.warning(f"🚫 Circuit breaker OPEN - blocking request (cooldown: {self.cooldown_duration - (time.time() - self.circuit_open_time):.1f}s remaining)")
                return None
        
        # Handle HALF_OPEN state
        if self.circuit_state == CircuitBreakerState.HALF_OPEN:
            if self.half_open_requests >= self.half_open_max_requests:
                logger.warning("🚫 Circuit breaker HALF_OPEN - max test requests reached")
                return None
            
            self.half_open_requests += 1
            logger.info(f"🔄 Circuit breaker HALF_OPEN - test request {self.half_open_requests}/{self.half_open_max_requests}")
        
        # Validate configuration
        if not self.api_gateway_url or not self.api_key:
            logger.error("❌ AWS RAG not configured - missing API Gateway URL or API key")
            self._record_failure(ErrorType.AUTHENTICATION, "Missing configuration")
            return None
        
        # Call RAG endpoint with retry logic
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    # Calculate exponential backoff delay
                    delay = self.retry_delay_ms * (self.retry_backoff_multiplier ** (attempt - 1))
                    # Add jitter to prevent thundering herd
                    jitter = random.uniform(0.1, 0.3) * delay
                    delay_with_jitter = delay + jitter
                    
                    logger.info(f"🔄 Retry attempt {attempt}/{self.max_retries} after {delay_with_jitter:.0f}ms")
                    await asyncio.sleep(delay_with_jitter / 1000)
                
                result = await self._make_rag_request(prompt)
                
                if result:
                    # Success - record and potentially close circuit
                    self._record_success()
                    return result
                else:
                    # Failure but no exception - don't retry
                    self._record_failure(ErrorType.SERVER_ERROR, "Empty response")
                    return None
                    
            except httpx.TimeoutException as e:
                error_type = ErrorType.TIMEOUT
                error_msg = f"Request timed out after {self.rag_timeout}s"
                logger.warning(f"⏰ {error_msg}")
                
                if attempt == self.max_retries:
                    self._record_failure(error_type, error_msg)
                    return None
                    
            except httpx.ConnectError as e:
                error_type = ErrorType.NETWORK
                error_msg = f"Network connection failed: {e}"
                logger.warning(f"🔌 {error_msg}")
                
                if attempt == self.max_retries:
                    self._record_failure(error_type, error_msg)
                    return None
                    
            except Exception as e:
                error_type = ErrorType.UNKNOWN
                error_msg = f"Unexpected error: {e}"
                logger.error(f"💥 {error_msg}")
                
                if attempt == self.max_retries:
                    self._record_failure(error_type, error_msg)
                    return None
        
        # All retries exhausted
        return None
    
    async def _make_rag_request(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Make the actual RAG request to AWS API Gateway.
        
        Args:
            prompt: The user's prompt text
            
        Returns:
            Dictionary with RAG results or None if failed
        """
        # Prepare request
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "User-Agent": "MQL5-PromptProxy/1.4.0"
        }
        
        payload = {
            "prompt": prompt,
            "max_snippets": self.max_snippets,
            "timestamp": datetime.now().isoformat()
        }
        
        # Initialize HTTP client if needed
        if self.http_client is None:
            self.http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.rag_timeout),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
            )
        
        logger.info(f"🌐 Calling AWS RAG endpoint: {self.api_gateway_url}")
        start_time = time.time()
        
        response = await self.http_client.post(
            self.api_gateway_url,
            json=payload,
            headers=headers
        )
        
        call_duration = (time.time() - start_time) * 1000
        self.response_times.append(call_duration)
        
        # Keep only last 100 response times for average calculation
        if len(self.response_times) > 100:
            self.response_times.pop(0)
        
        logger.info(f"⏱️ AWS RAG call completed in {call_duration:.2f}ms (status: {response.status_code})")
        
        # Handle response
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ RAG success: {len(result.get('snippets', []))} snippets retrieved")
            return result
        
        elif response.status_code == 429:
            raise Exception("Rate limited by API Gateway")
        
        elif response.status_code == 403:
            raise Exception("Authentication failed - check API key")
        
        else:
            raise Exception(f"Unexpected response status: {response.status_code}")
    
    def _transition_to_half_open(self):
        """Transition circuit breaker to HALF_OPEN state."""
        self.circuit_state = CircuitBreakerState.HALF_OPEN
        self.half_open_requests = 0
        logger.info("🔄 Circuit breaker transitioned to HALF_OPEN - testing recovery")
    
    def _record_success(self):
        """Record a successful RAG call and manage circuit breaker state."""
        self.success_count += 1
        self.last_success_time = datetime.now().isoformat()
        
        if self.circuit_state == CircuitBreakerState.HALF_OPEN:
            if self.success_count >= self.success_threshold:
                # Close circuit after sufficient successes
                self._close_circuit()
            else:
                logger.info(f"🔄 Circuit breaker HALF_OPEN - success {self.success_count}/{self.success_threshold}")
        
        elif self.circuit_state == CircuitBreakerState.CLOSED:
            # Reset failure count on success
            if self.failure_count > 0:
                logger.info(f"✅ RAG success - resetting failure count (was {self.failure_count})")
                self.failure_count = 0
    
    def _close_circuit(self):
        """Close the circuit breaker and reset counters."""
        self.circuit_state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_requests = 0
        logger.info("✅ Circuit breaker CLOSED - normal operation resumed")

    def _record_failure(self, error_type: ErrorType, error_msg: str):
        """Record a RAG call failure and manage circuit breaker state."""
        self.failure_count += 1
        self.total_failures += 1
        self.last_failure_time = datetime.now().isoformat()
        
        logger.warning(f"❌ RAG failure recorded: {error_type.value} - {error_msg} (count: {self.failure_count})")
        
        if self.circuit_state == CircuitBreakerState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                self._open_circuit()
        
        elif self.circuit_state == CircuitBreakerState.HALF_OPEN:
            # Any failure in HALF_OPEN immediately opens circuit
            self._open_circuit()
    
    def _open_circuit(self):
        """Open the circuit breaker."""
        self.circuit_state = CircuitBreakerState.OPEN
        self.circuit_open_time = time.time()
        self.circuit_open_count += 1
        self.success_count = 0
        self.half_open_requests = 0
        
        logger.warning(
            f"🚫 Circuit breaker OPENED after {self.failure_count} failures. "
            f"Cooldown for {self.cooldown_duration}s. Total opens: {self.circuit_open_count}"
        )
    
    def start(self):
        """Start the HTTP server."""
        logger.info(f"Starting PromptProxy server on http://{self.host}:{self.port}")
        logger.info("Available endpoints:")
        logger.info(f"  - Health: http://{self.host}:{self.port}/health")
        logger.info(f"  - Process: http://{self.host}:{self.port}/process")
        logger.info(f"  - RAG: http://{self.host}:{self.port}/rag")
        logger.info(f"  - Detect: http://{self.host}:{self.port}/detect")
        logger.info(f"  - Config: http://{self.host}:{self.port}/config")
        logger.info(f"  - Circuit: http://{self.host}:{self.port}/circuit")
        logger.info(f"  - Docs: http://{self.host}:{self.port}/docs")
        logger.info(f"MQL5 PromptDetector: {'✅ Available' if PROMPT_DETECTOR_AVAILABLE else '❌ Not Available'}")
        logger.info(f"AWS RAG Integration: {'✅ Configured' if (self.api_gateway_url and self.api_key) else '❌ Not Configured'}")
        logger.info(f"Prompt Augmentation: ✅ Enabled (max {self.max_snippets} snippets)")
        logger.info(f"Circuit Breaker: ✅ Enabled ({self.failure_threshold} failures → {self.cooldown_duration}s cooldown)")
        logger.info(f"Retry Policy: ✅ Enabled ({self.max_retries} retries, {self.retry_delay_ms}ms delay)")
        
        try:
            uvicorn.run(
                self.app,
                host=self.host,
                port=self.port,
                log_level="info",
                access_log=True
            )
        except KeyboardInterrupt:
            logger.info("Server shutdown requested")
        except Exception as e:
            logger.error(f"Server error: {e}")
            raise
        finally:
            # Clean up HTTP client
            if self.http_client:
                asyncio.create_task(self.http_client.aclose())
    
    def stop(self):
        """Stop the HTTP server gracefully."""
        logger.info("Stopping PromptProxy server...")
        if self.http_client:
            asyncio.create_task(self.http_client.aclose())


# Configuration class for server settings
class PromptProxyConfig:
    """Configuration settings for the PromptProxy server."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.host = "localhost"
        self.port = 8080
        self.log_level = "INFO"
        self.cors_origins = ["*"]
        self.max_prompt_length = 10000
        
        # AWS RAG configuration
        self.aws = {
            'api_gateway_url': '',
            'api_key': '',
            'timeout_seconds': 2.0
        }
        
        # Augmentation configuration
        self.augmentation = {
            'max_snippets': 5,
            'snippet_separator': '\n\n'
        }
        
        # Circuit breaker configuration (Task D24)
        self.circuit_breaker = {
            'failure_threshold': 3,
            'success_threshold': 2,
            'cooldown_duration': 300,
            'half_open_max_requests': 3
        }
        
        # Retry configuration (Task D24)
        self.retry = {
            'max_retries': 1,
            'delay_ms': 100,
            'backoff_multiplier': 2.0
        }
        
        if config_file and Path(config_file).exists():
            self._load_from_file(config_file)
    
    def _load_from_file(self, config_file: str):
        """Load configuration from YAML file."""
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            self.host = config.get('host', self.host)
            self.port = config.get('port', self.port)
            self.log_level = config.get('log_level', self.log_level)
            self.cors_origins = config.get('cors_origins', self.cors_origins)
            self.max_prompt_length = config.get('max_prompt_length', self.max_prompt_length)
            
            # AWS configuration
            if 'aws' in config:
                aws_config = config['aws']
                self.aws.update({
                    'api_gateway_url': aws_config.get('api_gateway_url', ''),
                    'api_key': aws_config.get('api_key', ''),
                    'timeout_seconds': aws_config.get('timeout_seconds', 2.0)
                })
            
            # Augmentation configuration
            if 'augmentation' in config:
                aug_config = config['augmentation']
                self.augmentation.update({
                    'max_snippets': aug_config.get('max_snippets', 5),
                    'snippet_separator': aug_config.get('snippet_separator', '\n\n')
                })
            
            # Circuit breaker configuration (Task D24)
            if 'circuit_breaker' in config:
                cb_config = config['circuit_breaker']
                self.circuit_breaker.update({
                    'failure_threshold': cb_config.get('failure_threshold', 3),
                    'success_threshold': cb_config.get('success_threshold', 2),
                    'cooldown_duration': cb_config.get('cooldown_duration', 300),
                    'half_open_max_requests': cb_config.get('half_open_max_requests', 3)
                })
            
            # Retry configuration (Task D24)
            if 'retry' in config:
                retry_config = config['retry']
                self.retry.update({
                    'max_retries': retry_config.get('max_retries', 1),
                    'delay_ms': retry_config.get('delay_ms', 100),
                    'backoff_multiplier': retry_config.get('backoff_multiplier', 2.0)
                })
            
            logger.info(f"Configuration loaded from {config_file}")
            
        except ImportError:
            logger.warning("PyYAML not installed, using default configuration")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format."""
        return {
            'host': self.host,
            'port': self.port,
            'log_level': self.log_level,
            'cors_origins': self.cors_origins,
            'max_prompt_length': self.max_prompt_length,
            'aws': self.aws,
            'augmentation': self.augmentation,
            'circuit_breaker': self.circuit_breaker,
            'retry': self.retry
        }


# Factory function for easy server creation
def create_promptproxy_server(
    host: str = "localhost", 
    port: int = 8080,
    config_file: Optional[str] = None
) -> PromptProxyServer:
    """
    Factory function to create a PromptProxy server instance.
    
    Args:
        host: Server host address
        port: Server port number
        config_file: Optional configuration file path
        
    Returns:
        Configured PromptProxyServer instance
    """
    if config_file:
        config = PromptProxyConfig(config_file)
        return PromptProxyServer(
            host=config.host, 
            port=config.port, 
            config=config.to_dict()
        )
    
    return PromptProxyServer(host=host, port=port)


# CLI entry point
def main():
    """Main entry point for running the server from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MQL5 PromptProxy Server with Full Circuit-Breaker Protection")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--config", help="Configuration file path (YAML)")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    # Create server with configuration
    server = create_promptproxy_server(
        host=args.host,
        port=args.port,
        config_file=args.config
    )
    
    # Display configuration status
    logger.info(f"Server configuration:")
    logger.info(f"  Host: {server.host}")
    logger.info(f"  Port: {server.port}")
    logger.info(f"  AWS configured: {bool(server.api_gateway_url and server.api_key)}")
    logger.info(f"  Augmentation: max {server.max_snippets} snippets")
    logger.info(f"  Circuit breaker: {server.failure_threshold} failures → {server.cooldown_duration}s cooldown")
    logger.info(f"  Retry policy: {server.max_retries} retries")
    if args.config:
        logger.info(f"  Config file: {args.config}")
    
    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
   exit(main())