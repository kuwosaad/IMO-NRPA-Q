"""
trace.py

Minimal tracing decorator for the IMO25 pipeline.
Provides lightweight instrumentation for key functions without business logic changes.
"""

import functools
import time
from typing import Any, Callable, Optional
from .logging_utils import log_print


def trace(func: Optional[Callable] = None, *, name: Optional[str] = None) -> Callable:
    """
    Decorator that traces function entry/exit with timing and argument summary.
    
    Args:
        func: Function to trace
        name: Optional custom name for the function in logs
    
    Usage:
        @trace
        def my_function(...): ...
        
        @trace(name="custom_name")
        def another_function(...): ...
    """
    if func is None:
        # Called with arguments: @trace(name="...")
        return lambda f: trace(f, name=name)
    
    func_name = name or func.__name__
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        # Build argument summary (truncate long values)
        args_summary = []
        for i, arg in enumerate(args):
            arg_str = str(arg)
            if len(arg_str) > 50:
                arg_str = arg_str[:47] + "..."
            args_summary.append(f"arg{i}={arg_str}")
        
        kwargs_summary = []
        for k, v in kwargs.items():
            v_str = str(v)
            if len(v_str) > 50:
                v_str = v_str[:47] + "..."
            kwargs_summary.append(f"{k}={v_str}")
        
        all_args = ", ".join(args_summary + kwargs_summary)
        if len(all_args) > 200:
            all_args = all_args[:197] + "..."
        
        log_print(f"[TRACE] ENTER {func_name}({all_args})")
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            # Build result summary
            result_str = str(result)
            if len(result_str) > 200:
                result_str = result_str[:197] + "..."
            
            log_print(f"[TRACE] EXIT {func_name} -> {result_str} ({elapsed:.2f}ms)")
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            log_print(f"[TRACE] ERROR {func_name} -> {type(e).__name__}: {e} ({elapsed:.2f}ms)")
            raise
    
    return wrapper