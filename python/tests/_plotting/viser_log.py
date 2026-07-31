"""A scrolling console-style log window for the viser demo apps.

The headless scripts (``ctrl_5f_phases.py``) print a running tick-by-tick log --
iteration counts, per-family constraint violations, the independent geometric
clearance check -- and that log is how you actually tell what a phase is doing.
The interactive apps had only a three-line status readout, so the moment you
switched to the visualizer you lost the history: you could see the current
violation but not whether it was falling, stalling or oscillating.

This renders that same text as a fixed-position pane on the LEFT of the viewport,
opposite viser's own control panel, via :meth:`viser.GuiApi.add_html`. Two
consequences of using raw HTML worth knowing:

  * Everything appended is HTML-escaped here, so a stray ``<`` in a message
    cannot break the pane.
  * ``position: fixed`` puts the pane outside viser's panel in the viewport. If
    a future viser version wraps GUI content in a transformed ancestor (which
    creates a containing block and would trap it), the pane still renders and
    still scrolls -- it just does so inside the right-hand panel.

Auto-scroll is done with ``flex-direction: column-reverse`` and the lines fed in
newest-first, so the newest line is pinned in view with no JavaScript: the
browser anchors a reversed column at what is visually the bottom.

Usage::

    log = ViserLog(server)
    log.rule("Phase 1: support contact")
    log.write("  tick  0   58.1 ms  iters=4")
    log.capture(geometric_report, solver, result, origin, normal, mask)
"""

import contextlib
import html
import io


# Cap on retained lines. High enough to hold a long auto-run, low enough that
# the whole pane is re-serialized to the client cheaply on every tick.
DEFAULT_MAX_LINES = 400

# Colour keys applied per line, by prefix match on the raw text. A log with no
# structure at all is hard to scan at 4 ticks/s; this is the cheapest structure
# that helps (phase rules and failures stand out) without a markup language.
_LEVEL_COLORS = {"rule": "#7dd3fc",     # phase separators
                 "warn": "#fbbf24",
                 "error": "#f87171",
                 "good": "#86efac",
                 "info": "#d4d4d8"}


class ViserLog:
    """A capped, scrolling text pane pinned to the left of the viewport."""

    def __init__(self, server, *, max_lines=DEFAULT_MAX_LINES, visible=True,
                 width_px=560, title="log"):
        self.max_lines = int(max_lines)
        self.width_px = int(width_px)
        self.title = title
        self._lines = []          # (level, text), newest LAST
        self._visible = bool(visible)
        self._html = server.gui.add_html(self._render())

    # -- writing ----------------------------------------------------------

    def write(self, text="", level="info"):
        """Append ``text`` (may be multi-line) and push to the client."""
        for line in str(text).split("\n"):
            self._lines.append((level, line))
        if len(self._lines) > self.max_lines:
            del self._lines[:-self.max_lines]
        self._flush()

    def rule(self, text):
        """A phase separator, styled like ``ctrl_5f_phases``' ``=== Phase N ===``."""
        self.write(f"=== {text} " + "=" * max(0, 46 - len(text)), level="rule")

    def capture(self, fn, *args, level="info", **kwargs):
        """Run ``fn`` with stdout redirected into the log, and return its result.

        This is the point of the class: the headless reports
        (``geometric_report``, ``report_pregrasp_target``) print rather than
        return their text, so capturing stdout gets the visualizer the EXACT
        report the script produces, with no second copy to drift from it.
        """
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = fn(*args, **kwargs)
        text = buf.getvalue().rstrip("\n")
        if text:
            self.write(text, level=level)
        return out

    def clear(self):
        self._lines.clear()
        self._flush()

    # -- visibility -------------------------------------------------------

    @property
    def visible(self):
        return self._visible

    @visible.setter
    def visible(self, on):
        self._visible = bool(on)
        self._flush()

    # -- rendering --------------------------------------------------------

    def _flush(self):
        self._html.content = self._render()

    def _render(self):
        if not self._visible:
            return ""
        rows = "".join(
            f'<div style="color:{_LEVEL_COLORS.get(lvl, _LEVEL_COLORS["info"])};'
            f'white-space:pre">{html.escape(text) or "&nbsp;"}</div>'
            # Newest FIRST into a column-reverse flexbox, which is what pins the
            # newest line in view without any scroll scripting.
            for lvl, text in reversed(self._lines))
        return (
            f'<div style="position:fixed;left:12px;top:12px;'
            f'width:{self.width_px}px;max-width:42vw;height:calc(100vh - 24px);'
            f'z-index:1000;display:flex;flex-direction:column;'
            f'pointer-events:auto">'
            f'<div style="font:600 11px/1.6 ui-sans-serif,system-ui;'
            f'letter-spacing:.08em;text-transform:uppercase;color:#a1a1aa;'
            f'background:rgba(24,24,27,.92);padding:4px 10px;'
            f'border-radius:6px 6px 0 0">{html.escape(self.title)}</div>'
            f'<div style="flex:1;overflow-y:auto;overflow-x:auto;'
            f'display:flex;flex-direction:column-reverse;'
            f'background:rgba(24,24,27,.92);color:#d4d4d8;'
            f'font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;'
            f'padding:8px 10px;border-radius:0 0 6px 6px">{rows}</div></div>')
