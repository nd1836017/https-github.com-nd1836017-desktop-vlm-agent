"""Run-scoped variable store + ``{{var.<name>}}`` placeholder substitution.

Variables are populated by the ``REMEMBER`` primitive (either with a literal
value or by asking the VLM to extract a value from the current screen) and
read by the ``RECALL`` primitive or by the ``{{var.<name>}}`` placeholder
embedded in any other step's natural-language text.

Substitution happens at runtime (just before sending a step to the planner)
because variables are not known at tasks-load time \u2014 they are populated as
the run progresses. This is intentionally separate from the
``{{row.<field>}}`` placeholder, which IS resolved at load time because the
CSV is known up-front.

The store is included in the run checkpoint so that resuming a long run
preserves any values learned before the crash.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


# Match ``{{var.name}}`` or ``{{var.name|default}}``. Whitespace inside the
# braces is tolerated (e.g. ``{{ var.email | none@example.com }}``). Variable
# names follow the same alphanum/underscore identifier shape as REMEMBER.
_VAR_PLACEHOLDER_RE = re.compile(
    r"\{\{\s*var\.([A-Za-z_][\w]*)\s*(?:\|([^}]*))?\s*\}\}"
)


class UnknownVariableError(KeyError):
    """Raised when a ``{{var.<name>}}`` placeholder references an unknown name.

    Distinct from a generic KeyError so the agent loop can catch it and turn
    the failure into a clean replan (or HALT) rather than crashing.
    """

    def __init__(self, name: str, available: tuple[str, ...]) -> None:
        self.name = name
        self.available = available
        msg = (
            f"Unknown variable {name!r}. "
            f"Available: {sorted(available) if available else '(none)'}"
        )
        super().__init__(msg)


@dataclass
class VariableStore:
    """Mutable string -> string mapping populated by REMEMBER.

    The store is intentionally minimal: just a dict, plus serialization
    helpers for the checkpoint. Values are always strings \u2014 the planner
    interpolates them as text into the next prompt.
    """

    _values: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------ basic
    def __len__(self) -> int:
        return len(self._values)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._values

    def __iter__(self):
        return iter(self._values)

    def names(self) -> tuple[str, ...]:
        return tuple(self._values.keys())

    # ---------------------------------------------------------------- mutate
    def set(self, name: str, value: str) -> None:
        """Store ``value`` under ``name``. Overwrites any prior value."""
        if not name:
            raise ValueError("Variable name must be non-empty.")
        if not isinstance(value, str):
            raise TypeError(
                f"Variable values must be str, not {type(value).__name__}."
            )
        self._values[name] = value
        log.info("Stored variable %r (%d chars)", name, len(value))

    def get(self, name: str, default: str | None = None) -> str:
        """Return the stored value, or ``default`` (or raise) if missing."""
        if name in self._values:
            return self._values[name]
        if default is not None:
            return default
        raise UnknownVariableError(name, tuple(self._values.keys()))

    # ------------------------------------------------------------ rendering
    def summary(self) -> str:
        """Compact one-line dump of stored variables for log/debug use."""
        if not self._values:
            return "(empty)"
        return ", ".join(
            f"{k}={_truncate(v, 40)!r}" for k, v in self._values.items()
        )

    # ------------------------------------------------------------- persist
    def to_dict(self) -> dict[str, str]:
        """Snapshot for checkpoint serialization."""
        return dict(self._values)

    @classmethod
    def from_dict(cls, data: Mapping[str, object] | None) -> VariableStore:
        """Rehydrate from a checkpoint snapshot. Tolerates ``None`` / bad shapes."""
        store = cls()
        if not data:
            return store
        for k, v in data.items():
            if not isinstance(k, str):
                continue
            store._values[k] = "" if v is None else str(v)
        return store


def substitute_variables(text: str, store: VariableStore) -> str:
    """Replace ``{{var.<name>}}`` placeholders in ``text`` using ``store``.

    Supports ``{{var.<name>|<default>}}`` for fallback when the variable
    is unset \u2014 useful when a step is reachable both before and after a
    REMEMBER. When no default is provided and the variable is unset,
    raises ``UnknownVariableError`` so the caller can surface a clear
    error to the user.

    Returns ``text`` unchanged when no placeholders are present.
    """
    if "{{" not in text:
        return text

    def repl(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(2)
        if name in store:
            return store.get(name)
        if default is not None:
            return default.strip()
        raise UnknownVariableError(name, store.names())

    return _VAR_PLACEHOLDER_RE.sub(repl, text)


def text_uses_variables(text: str) -> bool:
    """Cheap test: does ``text`` contain any ``{{var.<name>}}`` placeholder?

    Used by feature inspection (``inspect_features``) and by tests.
    """
    return bool(_VAR_PLACEHOLDER_RE.search(text or ""))


def _truncate(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "\u2026"
