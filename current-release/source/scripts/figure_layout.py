"""Explicit point-based figure geometry for the Tessera statistical figures.

Every figure is drawn on a canvas exactly as wide as the manuscript text block
(496.8 pt = 6.9 in), so LaTeX places it at 100% and every glyph keeps its
designed size. Axes, panel markers, titles and legends are positioned in
points from the canvas's upper-left corner; nothing is cropped with a tight
bounding box or moved by an automatic layout engine. Text widths are measured
once with the same LaTeX renderer to reserve margins, and `check_text_layout`
rejects any figure in which two text runs overlap or a run leaves the canvas.
"""
from __future__ import annotations
import itertools
import numpy as np
from matplotlib.font_manager import FontProperties

TEXT_WIDTH = 496.8          # manuscript \linewidth in points (letter, 0.8 in margins)
MARKER_SIZE = 13.5          # bold lowercase panel marker
TITLE_SIZE = 9.0
CELL_TITLE_SIZE = 8.5
INK = '#000000'
HEADER = 20.0               # block top -> axes top: marker/title band
MARKER_BASELINE = 13.0      # block top -> marker/title baseline


class Canvas:
    """A fixed-size figure whose children are placed in points from top-left."""

    def __init__(self, plt, height, width=TEXT_WIDTH):
        self.W, self.H = float(width), float(height)
        self.fig = plt.figure(figsize=(self.W / 72, self.H / 72))
        self.fig._tessera_layout = {'canvas_points': [self.W, self.H],
                                    'placement': 'explicit points from the upper-left corner',
                                    'tight_bbox_cropping': False}
        self._renderer = None

    # --- coordinates -------------------------------------------------------
    def rect(self, x, y, w, h):
        """Matplotlib figure fractions for a box given as top-left x/y and size."""
        return [x / self.W, 1 - (y + h) / self.H, w / self.W, h / self.H]

    def fx(self, x):
        return x / self.W

    def fy(self, y):
        return 1 - y / self.H

    # --- measurement -------------------------------------------------------
    def renderer(self):
        if self._renderer is None:
            self.fig.canvas.draw()
            self._renderer = self.fig.canvas.get_renderer()
        return self._renderer

    def text_width(self, strings, size):
        """Widest rendered width (points) among strings, using the LaTeX renderer."""
        renderer = self.renderer()
        prop = FontProperties(family='serif', size=size)
        scale = 72 / self.fig.dpi
        widths = [renderer.get_text_width_height_descent(str(s), prop, ismath=False)[0] * scale
                  for s in strings if str(s)]
        return max(widths) if widths else 0.0

    # --- artists -----------------------------------------------------------
    def axes(self, x, y, w, h, **kwargs):
        assert w > 0 and h > 0, (x, y, w, h)
        return self.fig.add_axes(self.rect(x, y, w, h), **kwargs)

    def map_axes(self, x, y, w, h, xlim, ylim, halign='left', valign='top', invert_y=True):
        """Equal-aspect axes sized to the data, fitted inside the box; returns (ax, box)."""
        dx = abs(xlim[1] - xlim[0]); dy = abs(ylim[1] - ylim[0])
        ratio = dy / dx
        mw, mh = (w, w * ratio) if w * ratio <= h else (h / ratio, h)
        mx = x + {'left': 0, 'center': (w - mw) / 2, 'right': w - mw}[halign]
        my = y + {'top': 0, 'center': (h - mh) / 2, 'bottom': h - mh}[valign]
        ax = self.axes(mx, my, mw, mh)
        ax.set_xlim(*xlim)
        ax.set_ylim(*(ylim[::-1] if invert_y else ylim))
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        return ax, (mx, my, mw, mh)

    def text(self, x, y, s, size=8, ha='left', va='baseline', gid=None, **kwargs):
        artist = self.fig.text(self.fx(x), self.fy(y), s, fontsize=size, ha=ha, va=va,
                               color=INK, fontweight='normal', **kwargs)
        if gid:
            artist.set_gid(gid)
        return artist

    def marker(self, x, y, letter):
        """Bold lowercase panel marker; (x, y) is its left edge and baseline."""
        artist = self.fig.text(self.fx(x), self.fy(y), letter.lower(), fontsize=MARKER_SIZE,
                               fontweight='bold', color=INK, ha='left', va='baseline')
        artist.set_gid('panel-label')
        return artist

    def title(self, x, y, s, size=TITLE_SIZE):
        artist = self.text(x, y, s, size=size)
        artist.set_gid('panel-title')
        return artist

    def header(self, x, y, letter, title):
        """Marker at the block's left edge and the title 16 pt to its right."""
        self.marker(x, y + MARKER_BASELINE, letter)
        self.title(x + 16, y + MARKER_BASELINE, title)

    def panel(self, x, y, w, h, letter, title, left, bottom, right=4.0, top=HEADER):
        """Header plus a plotting area inset by explicit tick/label allowances."""
        self.header(x, y, letter, title)
        return self.axes(x + left, y + top, w - left - right, h - top - bottom)

    def row_marker(self, x, y, letter, title, y_center):
        """Dense-grid convention: bold marker at the lane top, rotated row title below."""
        self.marker(x, y, letter)
        artist = self.text(x + 5, y_center, title, size=CELL_TITLE_SIZE, ha='center', va='center',
                           rotation=90)
        artist.set_gid('panel-title')
        return artist

    def legend(self, handles, x, y, ncol, size=7.5, **kwargs):
        """Legend whose upper-centre is at (x, y) in canvas points."""
        options = dict(frameon=False, columnspacing=1.0, handletextpad=.35, borderaxespad=0,
                       borderpad=0, fontsize=size)
        options.update(kwargs)
        return self.fig.legend(handles=handles, loc='upper center', ncol=ncol,
                               bbox_to_anchor=(self.fx(x), self.fy(y)),
                               bbox_transform=self.fig.transFigure, **options)

    def fitted_legend(self, handles, x, y, max_width, size=7.5, max_rows=None, **kwargs):
        """Fewest rows whose rendered width fits max_width points."""
        n = len(handles)
        for rows in range(1, n + 1):
            ncol = int(np.ceil(n / rows))
            legend = self.legend(handles, x, y, ncol, size=size, **kwargs)
            self.fig.canvas.draw()
            width = legend.get_window_extent(self.fig.canvas.get_renderer()).width * 72 / self.fig.dpi
            if width <= max_width or ncol == 1:
                return legend
            legend.remove()
        return legend


def _visible_texts(fig):
    """Every text run the renderer will draw: figure text, axis text, drawn ticks, legends."""
    texts = list(fig.texts)
    for legend in fig.legends:
        texts += legend.get_texts()
    for ax in fig.axes:
        texts += list(ax.texts) + [ax.xaxis.label, ax.yaxis.label, ax.title]
        for axis in (ax.xaxis, ax.yaxis):
            for tick in axis._update_ticks():
                texts += [label for label in (tick.label1, tick.label2) if label.get_visible()]
        if ax.get_legend() is not None:
            texts += ax.get_legend().get_texts()
    return [t for t in texts if t.get_visible() and t.get_text().strip()]


def check_text_layout(fig, name, tolerance=0.4, allow_inside=()):
    """Reject overlapping text runs and text outside the canvas (all in points)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    scale = 72 / fig.dpi
    width, height = fig.bbox.width * scale, fig.bbox.height * scale
    boxes = []
    for artist in _visible_texts(fig):
        box = artist.get_window_extent(renderer)
        if box.width == 0 or box.height == 0:
            continue
        x0, y0, x1, y1 = box.x0 * scale, box.y0 * scale, box.x1 * scale, box.y1 * scale
        if x0 < -tolerance or y0 < -tolerance or x1 > width + tolerance or y1 > height + tolerance:
            raise AssertionError(f'{name}: text leaves the canvas: {artist.get_text()!r} '
                                 f'({x0:.1f},{y0:.1f})-({x1:.1f},{y1:.1f}) on {width:.1f}x{height:.1f}')
        boxes.append((artist, (x0, y0, x1, y1)))
    overlaps = []
    for (a, p), (b, q) in itertools.combinations(boxes, 2):
        w = min(p[2], q[2]) - max(p[0], q[0]); h = min(p[3], q[3]) - max(p[1], q[1])
        if w > tolerance and h > tolerance:
            if any(rule(a, b) or rule(b, a) for rule in allow_inside):
                continue
            overlaps.append((a.get_text(), b.get_text(), round(w, 1), round(h, 1)))
    if overlaps:
        raise AssertionError(f'{name}: overlapping text runs: {overlaps[:6]}')
    return {'text_runs_checked': len(boxes), 'overlapping_text_runs': 0,
            'text_outside_canvas': 0, 'canvas_points': [round(width, 2), round(height, 2)]}
