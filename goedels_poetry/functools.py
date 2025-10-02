from functools import wraps


def maybe_save(n: int = 1):
    """
    Decorator to auto-save state after every n method calls.

    Each decorated method maintains its own independent counter.

    Parameters
    ----------
    n : int
        Save frequency - save after every n calls (default: 1, set to 0 to disable)
    """

    def decorator(func):
        # Initialize counter for this specific method
        func._save_counter = 0
        func._save_frequency = n

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Execute the original method
            result = func(self, *args, **kwargs)

            # Handle auto-save logic if enabled
            if func._save_frequency > 0:
                func._save_counter += 1
                if func._save_counter >= func._save_frequency:
                    func._save_counter = 0  # Reset counter
                    self._state.save()

            return result

        return wrapper

    return decorator
