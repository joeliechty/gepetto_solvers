"""The emergency stop, and the gate that serialises access to the robot.

Both are plain Python with no viser in them, so the smoke tests can exercise the
admission logic headlessly.
"""

import threading
import traceback

# ---------------------------------------------------------------------------
# Software e-stop.
# ---------------------------------------------------------------------------

class Refused(Exception):
    """Raised by :meth:`EStop.admit` when a solve is not allowed to start --
    either the latch is engaged or another solve already holds it."""


class EStop:
    """Latching software e-stop, and the single admission gate for every solve.

    LATCHING, not momentary. A momentary stop lets the very next thing that
    touches a slider restart the hand through the live-FK hook, which is
    precisely what someone reaching for a stop button does not want. Tripped
    stays tripped until :meth:`rearm`.

    The stop is COOPERATIVE, and it lands at an Augmented Lagrangian iteration
    boundary. One outer iteration is a single call into C++ with no interrupt
    hook (GTSAM's inner loop, via ``WarmAugmentedLagrangianOptimizer``), so
    nothing in Python can break into it -- ~1.7 s is the measured worst case and
    it is a floor, not a tuning choice. What matters is that this is bounded and
    that NO STATE IS LOST: the stepper keeps its multipliers, its penalty weight
    and its whole history, so a rearm resumes the same solve rather than
    restarting it.

    What makes the button feel instant is the GIL release on the solve binding
    (``capabilities()["gil_release"]``). With it the click is serviced while the
    last iteration is still running -- the latch engages, the controls grey out
    and the status updates at once. Without it the interpreter is frozen for the
    whole iteration and none of that can happen until it ends.

    Also the ONE place that decides whether a solve may start. It used to be a
    bare ``_solving`` bool tested and set without a lock, which the live-FK hook
    read while the auto-solve worker was writing it; folding the latch and the
    busy flag into one lock-guarded object closes that race as a side effect.
    """

    def __init__(self):
        # RLock rather than Lock: _refresh callbacks read the state while
        # holding it, and re-entering must not deadlock the GUI thread.
        self._lock = threading.RLock()
        self._tripped = False
        self._reason = ""
        self._busy = None       # what currently holds the gate, or None
        self._listeners = []    # notified on every trip/rearm; see add_listener

    # -- the latch --

    def add_listener(self, fn):
        """Register ``fn(tripped, reason)``, called on every trip and rearm.

        This is how the latch reaches things that are not solves. In ROS mode the
        robot bridge registers here, so pressing E-STOP does not merely refuse the
        next solve -- it halts the servo publishers, which is the only part of
        this app that can move a physical robot. Anything registered must return
        promptly and must not raise; see :meth:`_notify`.
        """
        with self._lock:
            self._listeners.append(fn)

    def _notify(self, tripped, reason):
        """Fan the latch's new state out to the listeners.

        Called with the lock RELEASED, deliberately. A listener publishes ROS
        messages and may take locks of its own, and holding the e-stop's lock
        across that is how a stop button ends up deadlocked against the thing it
        is trying to stop. An exception is swallowed with a traceback for the
        same reason: one listener failing must not stop the others being told,
        and must never turn the stop button into a crash.
        """
        for fn in list(self._listeners):
            try:
                fn(tripped, reason)
            except Exception:
                traceback.print_exc()

    def trip(self, reason="E-STOP pressed"):
        """Engage the latch. Runs on a viser callback thread, so it must never
        block: it takes the lock only to flip two fields, and deliberately does
        NOT wait for the running solve to notice."""
        with self._lock:
            if not self._tripped:
                self._tripped = True
                self._reason = reason
        self._notify(True, reason)
        return True

    def rearm(self):
        """Release the latch. Refuses while a solve is still winding down, so
        the GUI cannot come back to life around a solve that has not yet
        returned -- the operator would rearm into a hand still moving."""
        with self._lock:
            if self._busy is not None:
                return False
            self._tripped = False
            self._reason = ""
        self._notify(False, "")
        return True

    def is_tripped(self):
        """The poll predicate; handed straight to ``HandIKStepper.run`` as its
        ``should_stop``. A method rather than a property because that is the
        shape run() wants."""
        with self._lock:
            return self._tripped

    @property
    def reason(self):
        with self._lock:
            return self._reason

    @property
    def busy(self):
        with self._lock:
            return self._busy

    def check(self):
        """Raise if the latch is engaged. For polling inside a loop that has
        work of its own between solver calls."""
        if self.is_tripped():
            raise Refused(self._reason)

    # -- admission --

    def admit(self, what):
        """Claim the gate for one solve, or raise :class:`Refused`.

        Claims EAGERLY -- the refusal comes out of this call, not out of a later
        ``__enter__`` -- because the auto-solve hands its gate to a worker
        thread and so cannot express the claim as a ``with`` block. Callers that
        can still use one: ``with estop.admit(...)`` releases on the way out.

        The check and the claim happen under one lock, which is the point: two
        callbacks arriving on different viser worker threads cannot both see a
        free gate and both start solving.
        """
        with self._lock:
            if self._tripped:
                raise Refused(f"E-STOP engaged: {self._reason}")
            if self._busy is not None:
                raise Refused(f"already running: {self._busy}")
            self._busy = what
        return _Gate(self)

    def _release(self):
        with self._lock:
            self._busy = None


class _Gate:
    """The claim :meth:`EStop.admit` hands back. Releasing twice is harmless, so
    a caller may ``release()`` early and still let a ``with`` block unwind."""

    def __init__(self, estop):
        self._estop = estop

    def release(self):
        self._estop._release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
        return False
