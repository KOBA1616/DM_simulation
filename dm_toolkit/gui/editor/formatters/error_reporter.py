from contextlib import contextmanager
from typing import List

class ErrorReporter:
    """
    Context manager and error reporter for JSON path tracking during text generation.
    It tracks the current path being processed and can wrap executions to re-raise
    exceptions with the path embedded in the error message.
    """
    def __init__(self):
        self._path: List[str] = []

    @contextmanager
    def path_segment(self, segment: str):
        """Context manager to push and pop a segment from the current path."""
        self._path.append(segment)
        try:
            yield
        except Exception as e:
            # Re-raise with path info if it's not already wrapped
            if not getattr(e, '_has_path_info', False):
                path_str = self.current_path
                new_msg = f"{str(e)} (at path: {path_str})"
                new_e = ValueError(new_msg) if not isinstance(e, ValueError) else ValueError(new_msg)
                new_e.__cause__ = e
                new_e._has_path_info = True
                raise new_e from e
            raise
        finally:
            self._path.pop()

    @property
    def current_path(self) -> str:
        """Returns the current path as a dotted string."""
        return " -> ".join(self._path) if self._path else "root"
