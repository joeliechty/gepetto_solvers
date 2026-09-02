"""Fetching a hand by name.

Mirrors the C++ ``HandKinematicsRegistry``: a hand names a kinematics
(``Hand.kinematics``) that must be registered on the C++ side, and this registry
maps a hand NAME to the Python object that describes one particular hand built on
it. Two tendon hands of different morphology are two entries here over one C++
kinematics.

Factories are stored rather than instances, and :func:`get_hand` builds a fresh
hand each call: a hand hands out solver configs that the environment layer
mutates in place, so a shared instance would leak one solve's constraints into
the next.
"""

from __future__ import annotations

from collections.abc import Callable

from .base import Hand

#: The hand used when a caller names none.
DEFAULT_HAND = "tendon_5f"

#: A hand factory. Must be callable with NO arguments -- that is how
#: :func:`get_hand` invokes it -- but is typed to accept more, because a hand
#: class is usually registered directly and constructors take optional
#: configuration (``TendonHand5F(dims=...)``). Narrowing this to
#: ``Callable[[], Hand]`` rejects exactly those classes.
HandFactory = Callable[..., Hand]

_FACTORIES: dict[str, HandFactory] = {}


def register_hand(name: str, factory: HandFactory) -> None:
    """Register ``factory`` under ``name``. Re-registering replaces.

    ``factory`` must be callable with no arguments; a hand CLASS whose
    constructor arguments all have defaults is the usual thing to pass.
    """
    _FACTORIES[name] = factory


def get_hand(name: str | None = None) -> Hand:
    """Build the hand registered as ``name`` (default :data:`DEFAULT_HAND`).

    Raises naming every registered hand if the name is unknown: a hand silently
    falling back to the default is exactly the failure this layer exists to
    prevent.
    """
    key = DEFAULT_HAND if name is None else name
    factory = _FACTORIES.get(key)
    if factory is None:
        known = ", ".join(repr(n) for n in registered_hands()) or "(none)"
        raise KeyError(
            f"no hand registered as {key!r}. Registered: {known}.")
    return factory()


def registered_hands() -> list[str]:
    """Every registered hand name, sorted."""
    return sorted(_FACTORIES)
