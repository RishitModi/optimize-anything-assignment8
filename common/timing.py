"""Wall-clock performance profiling utilities using time.perf_counter."""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, Optional, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])


class Timer:
    """A context manager and decorator to measure elapsed wall-clock time.

    Uses `time.perf_counter` for high-resolution measurement.

    Attributes:
        label: Descriptive name for the measured code block.
        start_time: Timestamp when measurement began (in seconds).
        end_time: Timestamp when measurement ended (in seconds).
        elapsed_sec: Total duration in seconds (computed upon completion).

    Examples:
        Context manager usage:
        >>> with Timer("optimization_step") as timer:
        ...     # some computation
        ...     pass
        >>> print(f"Elapsed: {timer.elapsed_sec:.4f}s")

        Decorator usage:
        >>> @Timer("solve_epoch")
        ... def solve_epoch():
        ...     pass
    """

    def __init__(self, label: Optional[str] = None, print_on_exit: bool = False) -> None:
        """Initialize the Timer.

        Args:
            label: Optional label describing the measured section.
            print_on_exit: If True, prints elapsed time to stdout on exit.
        """
        self.label: Optional[str] = label
        self.print_on_exit: bool = print_on_exit
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed_sec: float = 0.0

    def __enter__(self) -> Timer:
        """Start the timer when entering context."""
        self.start_time = time.perf_counter()
        self.end_time = None
        self.elapsed_sec = 0.0
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Stop the timer when exiting context and calculate duration."""
        self.end_time = time.perf_counter()
        if self.start_time is not None:
            self.elapsed_sec = self.end_time - self.start_time

        if self.print_on_exit:
            name = f"[{self.label}] " if self.label else ""
            print(f"{name}Elapsed time: {self.elapsed_sec:.6f} seconds")

    def __call__(self, func: F) -> F:
        """Allow Timer to be used directly as a function decorator."""
        label = self.label or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with Timer(label=label, print_on_exit=self.print_on_exit) as t:
                result = func(*args, **kwargs)
            # Store last elapsed duration on the wrapper function for easy access
            setattr(wrapper, "last_elapsed_sec", t.elapsed_sec)
            return result

        return cast(F, wrapper)


def timed(label: Optional[str] = None, print_on_exit: bool = False) -> Callable[[F], F]:
    """Decorator to measure and log function execution time using time.perf_counter.

    Args:
        label: Custom name for the function. Defaults to the function's `__name__`.
        print_on_exit: Whether to print elapsed time upon function return.

    Returns:
        Decorated function with execution time profiling.

    Example:
        >>> @timed("Gradient Descent")
        ... def optimize():
        ...     pass
    """
    def decorator(func: F) -> F:
        name = label or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - t0
                setattr(wrapper, "last_elapsed_sec", elapsed)
                if print_on_exit:
                    print(f"[{name}] Finished in {elapsed:.6f} seconds")

        return cast(F, wrapper)

    return decorator
