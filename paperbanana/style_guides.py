DEFAULT_DIAGRAM_STYLE_GUIDE = """
NeurIPS-style methodology diagram guide:

1) Color and grouping
- Use light pastel containers to group phases (for example pale blue and pale orange).
- Reserve saturated colors only for key modules.
- Keep backgrounds clean (white or very light tones).

2) Shapes and layout
- Prefer rounded rectangles for process nodes.
- Use cylinders only for databases or memory-like components.
- Keep left-to-right narrative flow and avoid clutter.

3) Arrows and line semantics
- Solid lines for primary data flow.
- Dashed lines for auxiliary guidance or control signals.
- Avoid unnecessary arrow crossings.

4) Typography
- Sans-serif labels for module names.
- Serif italic for symbolic variables (S, C, P, I, R, E, G).
- Keep text concise and legible.

5) Domain fit
- For agent and LLM papers, icon-rich narrative style is acceptable.
- Keep the figure publication-ready and avoid default slide-template aesthetics.
""".strip()


DEFAULT_PLOT_STYLE_GUIDE = """
NeurIPS-style statistical plot guide:

1) Visual tone
- Use high-contrast, publication-ready styling.
- Prefer clear white backgrounds or subtle light gray alternatives.

2) Colors
- Use readable palettes; avoid rainbow/jet colormaps.
- Distinguish categories by color plus markers/patterns when possible.

3) Axes and grids
- Use light dashed grids behind data elements.
- Keep labels readable and uncluttered.

4) Plot-type conventions
- Line plots: include markers and clean legends.
- Bar plots: consistent spacing and optional error bars.
- Heatmaps: perceptually uniform colormaps and optional value annotations.

5) Publication quality
- Avoid chartjunk, heavy shadows, and outdated default styles.
- Prioritize clarity, faithfulness to values, and compact layout.
""".strip()
