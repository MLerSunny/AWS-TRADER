"""
Circuit breaker pattern implementation for handling external service failures.

This module provides a CircuitBreaker class that can wrap around
external service calls to prevent cascading failures.
"""
import asyncio
import functools
import logging
import time
from enum import Enum
from typing import Any, Callable, Dict, Optional, TypeVar, Union, cast

# Import configuration
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import config

logger = logging.getLogger(__name__)

T = TypeVar("T")  # Return type for the wrapped function

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"      # Circuit is open, requests immediately fail
    HALF_OPEN = "half_open"  # Testing if service is recovered


class CircuitBreaker:
    """
    Circuit breaker for protecting against external service failures.
    
    Attributes:
        name (str): Name for the circuit breaker
        failure_threshold (int): Number of failures before opening circuit
        recovery_timeout (int): Seconds to wait before trying recovery
        expected_exceptions (tuple): Exception types to count as failures
        fallback_function (Callable): Function to call when circuit is open
    """
    
    # Class-level storage of circuit breakers by name
    _circuit_breakers: Dict[str, 'CircuitBreaker'] = {}
    
    @classmethod
    def get_or_create(cls, name: str, **kwargs) -> 'CircuitBreaker':
        """
        Get an existing circuit breaker by name or create a new one.
        
        Args:
            name: Name of the circuit breaker
            **kwargs: Arguments for creating a new circuit breaker
            
        Returns:
            CircuitBreaker: Existing or new circuit breaker instance
        """
        if name not in cls._circuit_breakers:
            cls._circuit_breakers[name] = CircuitBreaker(name, **kwargs)
        return cls._circuit_breakers[name]
    
    def __init__(
        self,
        name: str,
        failure_threshold: Optional[int] = None,
        recovery_timeout: Optional[int] = None,
        expected_exceptions: tuple = (Exception,),
        fallback_function: Optional[Callable] = None,
    ):
        """
        Initialize the circuit breaker.
        
        Args:
            name: Name for the circuit breaker
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before trying recovery
            expected_exceptions: Exception types to count as failures
            fallback_function: Function to call when circuit is open
        """
        self.name = name
        self.failure_threshold = failure_threshold or config.CIRCUIT_BREAKER_FAILURE_THRESHOLD
        self.recovery_timeout = recovery_timeout or config.CIRCUIT_BREAKER_RECOVERY_TIMEOUT
        self.expected_exceptions = expected_exceptions
        self.fallback_function = fallback_function
        
        # State
        self._failure_count = 0
        self._state = CircuitState.CLOSED
        self._last_failure_time = 0
        
        # Register in class storage
        CircuitBreaker._circuit_breakers[name] = self
        
        logger.info(
            f"Circuit breaker '{name}' initialized with "
            f"failure_threshold={self.failure_threshold}, "
            f"recovery_timeout={self.recovery_timeout}s"
        )
    
    @property
    def state(self) -> CircuitState:
        """
        Get the current state of the circuit breaker.
        
        The state automatically transitions from OPEN to HALF_OPEN
        after the recovery timeout period.
        
        Returns:
            CircuitState: Current state
        """
        # Check if we should transition from OPEN to HALF_OPEN
        if (
            self._state == CircuitState.OPEN and
            time.time() - self._last_failure_time >= self.recovery_timeout
        ):
            logger.info(f"Circuit '{self.name}' transitioning from OPEN to HALF_OPEN")
            self._state = CircuitState.HALF_OPEN
        
        return self._state
    
    def __call__(self, func):
        """
        Decorate a function with the circuit breaker.
        
        Args:
            func: The function to wrap
            
        Returns:
            Callable: Wrapped function
        """
        is_async = asyncio.iscoroutinefunction(func)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            """Async wrapper around the function."""
            return await self._handle_async_call(func, *args, **kwargs)
            
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            """Synchronous wrapper around the function."""
            return self._handle_sync_call(func, *args, **kwargs)
        
        return async_wrapper if is_async else sync_wrapper
    
    async def _handle_async_call(self, func, *args, **kwargs):
        """Handle an async function call with circuit breaker logic."""
        current_state = self.state
        
        # If circuit is open, fail fast
        if current_state == CircuitState.OPEN:
            logger.warning(f"Circuit '{self.name}' is OPEN - fast failing")
            if self.fallback_function:
                if asyncio.iscoroutinefunction(self.fallback_function):
                    return await self.fallback_function(*args, **kwargs)
                return self.fallback_function(*args, **kwargs)
            raise CircuitOpenError(f"Circuit '{self.name}' is open")
            
        try:
            # Call the function
            result = await func(*args, **kwargs)
            
            # If successful in HALF_OPEN, reset to CLOSED
            if current_state == CircuitState.HALF_OPEN:
                logger.info(f"Circuit '{self.name}' recovery successful, transitioning to CLOSED")
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                
            return result
            
        except self.expected_exceptions as e:
            # Handle failure based on current state
            return await self._handle_failure(e)
    
    def _handle_sync_call(self, func, *args, **kwargs):
        """Handle a synchronous function call with circuit breaker logic."""
        current_state = self.state
        
        # If circuit is open, fail fast
        if current_state == CircuitState.OPEN:
            logger.warning(f"Circuit '{self.name}' is OPEN - fast failing")
            if self.fallback_function:
                return self.fallback_function(*args, **kwargs)
            raise CircuitOpenError(f"Circuit '{self.name}' is open")
            
        try:
            # Call the function
            result = func(*args, **kwargs)
            
            # If successful in HALF_OPEN, reset to CLOSED
            if current_state == CircuitState.HALF_OPEN:
                logger.info(f"Circuit '{self.name}' recovery successful, transitioning to CLOSED")
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                
            return result
            
        except self.expected_exceptions as e:
            # Handle failure based on current state
            return self._handle_failure_sync(e)
    
    async def _handle_failure(self, exception):
        """
        Handle a service call failure.
        
        Args:
            exception: The exception that was raised
            
        Raises:
            CircuitOpenError: If in HALF_OPEN state
            exception: The original exception in CLOSED state
        """
        current_state = self.state
        self._last_failure_time = time.time()
        
        if current_state == CircuitState.HALF_OPEN:
            # Failed recovery attempt, back to OPEN
            logger.warning(
                f"Circuit '{self.name}' recovery attempt failed, "
                f"remaining in OPEN state for another {self.recovery_timeout}s"
            )
            self._state = CircuitState.OPEN
            
            if self.fallback_function:
                if asyncio.iscoroutinefunction(self.fallback_function):
                    return await self.fallback_function()
                return self.fallback_function()
            raise CircuitOpenError(f"Circuit '{self.name}' recovery failed")
            
        # In CLOSED state, increment failure count
        self._failure_count += 1
        logger.warning(
            f"Circuit '{self.name}' failure count: {self._failure_count}/{self.failure_threshold}"
        )
        
        # Check if threshold reached
        if self._failure_count >= self.failure_threshold:
            logger.error(
                f"Circuit '{self.name}' failure threshold reached, "
                f"transitioning to OPEN for {self.recovery_timeout}s"
            )
            self._state = CircuitState.OPEN
            
            if self.fallback_function:
                if asyncio.iscoroutinefunction(self.fallback_function):
                    return await self.fallback_function()
                return self.fallback_function()
            raise CircuitOpenError(f"Circuit '{self.name}' opened after {self._failure_count} failures")
        
        # Otherwise, propagate the original exception
        raise exception
    
    def _handle_failure_sync(self, exception):
        """Synchronous version of _handle_failure."""
        current_state = self.state
        self._last_failure_time = time.time()
        
        if current_state == CircuitState.HALF_OPEN:
            # Failed recovery attempt, back to OPEN
            logger.warning(
                f"Circuit '{self.name}' recovery attempt failed, "
                f"remaining in OPEN state for another {self.recovery_timeout}s"
            )
            self._state = CircuitState.OPEN
            
            if self.fallback_function:
                return self.fallback_function()
            raise CircuitOpenError(f"Circuit '{self.name}' recovery failed")
            
        # In CLOSED state, increment failure count
        self._failure_count += 1
        logger.warning(
            f"Circuit '{self.name}' failure count: {self._failure_count}/{self.failure_threshold}"
        )
        
        # Check if threshold reached
        if self._failure_count >= self.failure_threshold:
            logger.error(
                f"Circuit '{self.name}' failure threshold reached, "
                f"transitioning to OPEN for {self.recovery_timeout}s"
            )
            self._state = CircuitState.OPEN
            
            if self.fallback_function:
                return self.fallback_function()
            raise CircuitOpenError(f"Circuit '{self.name}' opened after {self._failure_count} failures")
        
        # Otherwise, propagate the original exception
        raise exception
    
    def reset(self):
        """Reset the circuit breaker to its initial closed state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0
        logger.info(f"Circuit '{self.name}' manually reset to CLOSED state")


class CircuitOpenError(Exception):
    """Exception raised when a circuit is open."""
    pass 