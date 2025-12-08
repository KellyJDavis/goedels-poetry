from collections.abc import Callable
from functools import wraps
from typing import Any, Concatenate, TypeVar

from typing_extensions import ParamSpec

P = ParamSpec("P")
R = TypeVar("R")


def maybe_save(n: int = 1) -> Callable[[Callable[Concatenate[Any, P], R]], Callable[Concatenate[Any, P], R]]:
    """
    Decorator to auto-save state after every n method calls.

    Each decorated method maintains its own independent counter.

    Parameters
    ----------
    n : int
        Save frequency - save after every n calls (default: 1, set to 0 to disable)

    Returns
    -------
    Callable[[Callable[Concatenate[Any, P], R]], Callable[Concatenate[Any, P], R]]
        Decorated function
    """

    def decorator(func: Callable[Concatenate[Any, P], R]) -> Callable[Concatenate[Any, P], R]:
        # Initialize counter for this specific method (stored on the function to preserve behavior)
        func._save_counter = 0  # type: ignore[attr-defined]
        func._save_frequency = n  # type: ignore[attr-defined]

        @wraps(func)
        def wrapper(self: Any, *args: P.args, **kwargs: P.kwargs) -> R:
            # Execute the original method
            result = func(self, *args, **kwargs)

            # Handle auto-save logic if enabled
            if func._save_frequency > 0:  # type: ignore[attr-defined]
                func._save_counter += 1  # type: ignore[attr-defined]
                if func._save_counter >= func._save_frequency:  # type: ignore[attr-defined]
                    func._save_counter = 0  # type: ignore[attr-defined]
                    self._state.save()

            return result

        return wrapper

    return decorator
