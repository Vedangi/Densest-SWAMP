
"""
plot_nyc_solution_folium.py

Plot Densest-SWAMP / hypergraph solutions on an NYC subway map with
interactive parameter selection (n, p).

Solutions are stored as a tuple-keyed dict:
    SOLUTIONS[(n, p)] = [node_id, node_id, ...]

The output HTML contains two dropdowns (n and p) in the top-right.
Selecting a combination:
    - hides every solution layer
    - shows the layer for that (n, p)
    - rewrites the summary box on the left

Inputs:
    nyc_hypergraph_parent_ids.json
    nyc_node_positions.py
    nyc_line_colors.py
    nyc_route_shapes.json
    nyc_lines_hypergraph.txt

Output:
    nyc_solution_map.html
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import folium

from nyc_node_positions_cgpt import NODE_POS, ID_TO_NAME, ID_TO_PARENT
from nyc_line_colors_cgpt import LINE_COLORS
# from solution_list import SOLUTIONS
from solution_list import SOLUTIONS



# -----------------------------
# Config
# -----------------------------
HYPERGRAPH_JSON = "nyc_hypergraph_parent_ids.json"
LINES_FILE = "nyc_lines_hypergraph.txt"
ROUTE_SHAPES_JSON = "nyc_route_shapes.json"
OUT_HTML = "nyc_solution_map.html"


# Initial selection when the page opens
DEFAULT_N = 10
DEFAULT_P = 0.0

SHOW_ALL_STATIONS = True
SIZE_BY_TRANSFER_DEGREE = True


# -----------------------------
# Canonical formatting for the `p` parameter
# -----------------------------
def format_p(p):
    """
    Canonical string for `p`, used both as the dropdown label and as the
    lookup key in layer_name_by_key / summary_by_key.

    This is the SINGLE source of formatting truth. JS never reformats `p`;
    it just consumes the strings we hand it. That avoids cross-language
    disagreements like Python's '1e-08' vs JS's '1e-8' breaking lookups.

    Examples:
        0       -> "0"
        0.0     -> "0"
        0.9     -> "0.9"
        1e-3    -> "0.001"
        1e-8    -> "1e-8"      (leading zero in the exponent is stripped)
        2.5e-12 -> "2.5e-12"
    """
    pf = float(p)
    if pf == 0:
        return "0"
    s = f"{pf:g}"
    # Strip leading zeros in the exponent: '1e-08' -> '1e-8'
    if "e" in s:
        mantissa, exp = s.split("e")
        sign = exp[0] if exp[0] in "+-" else ""
        digits = exp.lstrip("+-").lstrip("0") or "0"
        s = f"{mantissa}e{sign}{digits}"
    return s


def make_key(n, p):
    """The string key used to look up a solution layer / summary."""
    return f"{n}_{format_p(p)}"


# -----------------------------
# Loaders
# -----------------------------
def load_hypergraph(path=HYPERGRAPH_JSON):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_line2stations_txt(path=LINES_FILE):
    line2stations = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split()
            line = parts[0]
            stations = [int(x) for x in parts[1:]]
            if stations:
                line2stations[line] = stations
    return line2stations


def load_route_shapes(path=ROUTE_SHAPES_JSON):
    if not Path(path).exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def station_lines_index(line2stations):
    """node_id -> sorted list of lines containing the station."""
    idx = defaultdict(list)
    for line, stations in line2stations.items():
        for v in stations:
            idx[int(v)].append(line)
    return {v: sorted(lines) for v, lines in idx.items()}


# -----------------------------
# Geometry helpers
# -----------------------------
def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


def map_center_for_nodes(node_ids, pos_ll):
    coords = [(pos_ll[v][1], pos_ll[v][0]) for v in node_ids if v in pos_ll]
    if coords:
        lat = sum(x[0] for x in coords) / len(coords)
        lon = sum(x[1] for x in coords) / len(coords)
        return [lat, lon]

    coords = [(lat, lon) for lon, lat in pos_ll.values()]
    if coords:
        lat = sum(x[0] for x in coords) / len(coords)
        lon = sum(x[1] for x in coords) / len(coords)
        return [lat, lon]

    return [40.7128, -74.0060]


# -----------------------------
# Static map layers (routes / fallback / all stations)
# -----------------------------
def add_route_shapes_layer(
    m, route_shapes, line_colors,
    name="Subway route geometry from shapes.txt",
    show=True, weight=4, opacity=0.55,
):
    fg = folium.FeatureGroup(name=name, show=show)
    if not route_shapes:
        fg.add_to(m)
        return fg

    for line, shape_records in route_shapes.items():
        color = line_colors.get(line, "#666666")
        for rec in shape_records:
            pts = rec.get("points", [])
            if len(pts) < 2:
                continue
            tooltip = (
                f"Line {line}"
                f"<br>direction: {rec.get('direction_id', '')}"
                # f"<br>shape_id: {rec.get('shape_id', '')}"
            )
            folium.PolyLine(
                locations=pts, color=color, weight=weight,
                opacity=opacity, tooltip=tooltip,
            ).add_to(fg)

    fg.add_to(m)
    return fg


def add_fallback_stop_segments_layer(
    m, line2stations, pos_ll, line_colors,
    name="Fallback stop-to-stop segments",
    show=False, weight=3, opacity=0.35,
):
    fg = folium.FeatureGroup(name=name, show=show)
    for line, stations in line2stations.items():
        color = line_colors.get(line, "#666666")
        stations = [int(v) for v in stations]
        for u, v in zip(stations, stations[1:]):
            if u not in pos_ll or v not in pos_ll:
                continue
            lon1, lat1 = pos_ll[u]
            lon2, lat2 = pos_ll[v]
            folium.PolyLine(
                locations=[(lat1, lon1), (lat2, lon2)],
                color=color, weight=weight, opacity=opacity,
                tooltip=f"{line}: {ID_TO_NAME.get(u, u)} → {ID_TO_NAME.get(v, v)}",
            ).add_to(fg)
    fg.add_to(m)
    return fg


def add_all_stations_layer(
    m, line2stations, pos_ll, st2lines,
    name="All physical stations", show=True,
    radius=3, opacity=0.55,
):
    fg = folium.FeatureGroup(name=name, show=show)
    all_nodes = sorted({v for stations in line2stations.values() for v in stations})

    for v in all_nodes:
        if v not in pos_ll:
            continue
        lon, lat = pos_ll[v]
        station_name = ID_TO_NAME.get(v, str(v))
        parent = ID_TO_PARENT.get(v, "")
        tooltip = (
            f"<b>{station_name}</b>"
            f"<br>node_id: {v}"
            f"<br>parent_station: {parent}"
            f"<br>lines: {', '.join(st2lines.get(v, []))}"
        )
        folium.CircleMarker(
            location=[lat, lon], radius=radius,
            color="#333333", weight=1, fill=True,
            fill_color="#666666", fill_opacity=opacity, opacity=opacity,
            tooltip=folium.Tooltip(tooltip, sticky=True),
        ).add_to(fg)
    fg.add_to(m)
    return fg


# -----------------------------
# One FeatureGroup per (n, p) solution
# -----------------------------
def build_solution_layer(m, solution_ids, pos_ll, st2lines, name, show=False):
    """
    Build a FeatureGroup of solution station markers. Returns the FeatureGroup
    so we can grab its JS variable name for the dropdown wiring.

    `control=False` keeps these layers OUT of the standard LayerControl, since
    they are managed exclusively by the n/p dropdowns.
    """
    fg = folium.FeatureGroup(name=name, show=show, control=False)
    sol = sorted(set(int(v) for v in solution_ids))

    for v in sol:
        if v not in pos_ll:
            print(f"[warn] solution node {v} has no position, skipping")
            continue
        lon, lat = pos_ll[v]
        station_name = ID_TO_NAME.get(v, str(v))
        parent = ID_TO_PARENT.get(v, "")
        lines_here = st2lines.get(v, [])
        deg = len(lines_here)

        if SIZE_BY_TRANSFER_DEGREE:
            radius = 7 + 2.5 * math.log1p(max(0, deg - 1))
        else:
            radius = 8

        tooltip = (
            f"<b>{station_name}</b>"
            f"<br>node_id: {v}"
            f"<br>parent_station: {parent}"
            f"<br>transfer degree: {deg}"
            f"<br>lines: {', '.join(lines_here)}"
        )
        folium.CircleMarker(
            location=[lat, lon], radius=radius,
            color="black", weight=2, fill=True,
            fill_color="#c60c30", fill_opacity=0.96, opacity=1.0,
            tooltip=folium.Tooltip(tooltip, sticky=True),
        ).add_to(fg)

    fg.add_to(m)
    return fg


def compute_summary(solution_ids, line2stations, st2lines, line_colors):
    """Precompute everything the summary box needs, per solution."""
    sol = set(int(v) for v in solution_ids)

    touched = []
    for line, stations in line2stations.items():
        hit = sum(1 for v in stations if int(v) in sol)
        if hit > 0:
            touched.append({
                "line": line,
                "hit": hit,
                "total": len(stations),
                "color": line_colors.get(line, "#777777"),
            })
    touched.sort(key=lambda x: (-x["hit"], x["line"]))

    transfer_ct = sum(1 for v in sol if len(st2lines.get(v, [])) >= 2)

    return {
        "size": len(sol),
        "lines_touched": len(touched),
        "transfer_count": transfer_ct,
        "top_lines": touched[:15],
    }


# -----------------------------
# Dropdown + dynamic summary UI
# -----------------------------
def add_interactive_controls(
    m, layer_name_by_key, summary_by_key,
    available_combinations, default_key,
    overlay_toggle_name=None,
):
    """
    Add two dropdowns (n, p) in the top-right and a dynamic summary box
    in the top-left. Their behaviour is wired in a small JS block:

        - change n  -> repopulate p with the p values available for that n
        - change p  -> hide every solution layer, show the matching one,
                        rewrite the summary box

    Parameters
    ----------
    layer_name_by_key : dict[str, str]
        "n_p" -> folium JS variable name of the FeatureGroup (from fg.get_name()).
    summary_by_key : dict[str, dict]
        "n_p" -> precomputed summary dict.
    available_combinations : list[tuple[int|float, float]]
    default_key : str
        The "n_p" key to show on load.
    """
    map_var = m.get_name()
    ns = sorted({n for n, _ in available_combinations})
    # Sort p values as floats (numeric order), then convert each to its canonical
    # string. JS will use those strings as both the option label and the lookup
    # key — it does not reformat them.
    ps_by_n_str = {
        str(n): [format_p(p) for p in sorted({p for nn, p in available_combinations if nn == n})]
        for n in ns
    }

    payload = {
        "layerNames": layer_name_by_key,
        "summaries": summary_by_key,
        "ns": ns,
        "psByN": ps_by_n_str,
        "defaultKey": default_key,
        "overlayToggleName": overlay_toggle_name,
    }
    payload_json = json.dumps(payload)

    # ---- top-right: dropdowns ----
    controls_html = """
    <div id="solution-controls" style="
        position: fixed; top: 10px; right: 10px; z-index: 10000;
        background: rgba(255,255,255,0.96);
        padding: 12px 14px; border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        font-family: Arial, sans-serif; font-size: 13px;
        min-width: 200px;">
        <div style="font-weight:700; margin-bottom:8px;">Solution selector</div>
        <div style="margin-bottom:6px;">
            <label for="n-select" style="display:inline-block; width:24px;">|S|</label>
            <select id="n-select" style="padding:3px 6px; min-width:120px;"></select>
        </div>
        <div>
            <label for="p-select" style="display:inline-block; width:24px;">p</label>
            <select id="p-select" style="padding:3px 6px; min-width:120px;"></select>
        </div>
        <div id="solution-status" style="margin-top:8px; color:#666; font-size:12px;"></div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(controls_html))

    # ---- top-left: summary box (filled by JS) ----
    summary_html = """
    <div id="solution-summary" style="
        position: fixed; top: 10px; left: 50px; z-index: 9999;
        background: rgba(255,255,255,0.94);
        padding: 10px 12px; border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.18);
        font-family: Arial, sans-serif; font-size: 12.5px;
        line-height: 1.35; min-width: 240px;">
        <em>Loading summary…</em>
    </div>
    """
    m.get_root().html.add_child(folium.Element(summary_html))

    # ---- JS wiring ----
    js = f"""
    <script>
    (function() {{
        const PAYLOAD = {payload_json};
        const MAP_VAR_NAME = "{map_var}";
        // Late-bound: folium declares `var {map_var} = L.map(...)` AFTER this
        // <script> tag, so we cannot dereference it at parse time.
        // window["{map_var}"] returns undefined instead of throwing,
        // and init() runs on a setTimeout so by then the map is ready.
        function getMap() {{ return window[MAP_VAR_NAME]; }}

        // Master toggle for the (n, p) solution overlay. Bound to the sentinel
        // FeatureGroup named "Solution overlay" in the bottom-right LayerControl.
        let showOverlay = true;

        // NOTE: `p` is delivered already-formatted by Python (see format_p).
        // We never reformat it here — that would risk Python/JS disagreeing on
        // things like "1e-08" vs "1e-8".

        function fillN() {{
            const sel = document.getElementById('n-select');
            sel.innerHTML = "";
            PAYLOAD.ns.forEach(function(n) {{
                const opt = document.createElement('option');
                opt.value = n;
                opt.textContent = n;
                sel.appendChild(opt);
            }});
        }}

        function fillP(n) {{
            const sel = document.getElementById('p-select');
            sel.innerHTML = "";
            const ps = PAYLOAD.psByN[String(n)] || [];
            ps.forEach(function(p) {{
                const opt = document.createElement('option');
                opt.value = p;        // already canonical, e.g. "0", "0.9", "1e-8"
                opt.textContent = p;
                sel.appendChild(opt);
            }});
        }}

        function hideAllSolutions() {{
            const mapObj = getMap();
            if (!mapObj) return;
            for (const key in PAYLOAD.layerNames) {{
                const name = PAYLOAD.layerNames[key];
                try {{
                    const layer = window[name];
                    if (layer && mapObj.hasLayer(layer)) {{
                        mapObj.removeLayer(layer);
                    }}
                }} catch (e) {{ /* ignore */ }}
            }}
        }}

        function showSolution(key) {{
            const mapObj = getMap();
            if (!mapObj) return false;
            const name = PAYLOAD.layerNames[key];
            if (!name) return false;
            const layer = window[name];
            if (!layer) return false;
            if (!mapObj.hasLayer(layer)) {{
                mapObj.addLayer(layer);
            }}
            return true;
        }}

        function renderSummary(key) {{
            const box = document.getElementById('solution-summary');
            const s = PAYLOAD.summaries[key];
            if (!s) {{
                box.innerHTML = "<em>No solution stored for this (n, p).</em>";
                return;
            }}
            let linesHtml = "";
            if (s.top_lines && s.top_lines.length) {{
                linesHtml = s.top_lines.map(function(row) {{
                    const sw = '<span style="display:inline-block;width:12px;height:12px;'
                             + 'background:' + row.color + ';border:1px solid #333;'
                             + 'margin-right:6px;vertical-align:middle;"></span>';
                    return sw + '<b>' + row.line + '</b>: ' + row.hit + '/' + row.total;
                }}).join("<br>");
            }} else {{
                linesHtml = "None";
            }}
            box.innerHTML =
                '<div style="font-size:14px; font-weight:700; margin-bottom:6px;">'
                + 'Hypergraph solution (|S|=' + s.size + ')</div>'
                + '<div><b>Total routes covered:</b> ' + s.lines_touched + '</div>'
                + '<div><b>Transfer stations:</b> ' + s.transfer_count + '/' + s.size + '</div>'
                + '<div style="font-weight:700; margin-top:8px;">Top covered routes:</div>'
                + '<div style="line-height:1.45;">' + linesHtml + '</div>';
        }}

        function applySelection() {{
            const n = document.getElementById('n-select').value;
            const p = document.getElementById('p-select').value;
            const key = n + "_" + p;
            const status = document.getElementById('solution-status');

            hideAllSolutions();
            if (showOverlay) {{
                const ok = showSolution(key);
                if (ok) {{
                    status.textContent = "Showing solution for |S|=" + n + ", p=" + p;
                    status.style.color = "#666";
                }} else {{
                    status.textContent = "No solution stored for |S|=" + n + ", p=" + p;
                    status.style.color = "#b00020";
                }}
            }} else {{
                status.textContent = "Overlay hidden (solution " + key + " selected)";
                status.style.color = "#888";
            }}
            renderSummary(key);
        }}

        function init() {{
            fillN();

            const dk = PAYLOAD.defaultKey || "";
            // Split on first "_" only — the n-part has no underscores, but
            // being explicit keeps us safe if format_p ever evolves.
            const sepIdx = dk.indexOf("_");
            const defN = sepIdx >= 0 ? dk.slice(0, sepIdx) : (PAYLOAD.ns[0] || "");
            const defP = sepIdx >= 0 ? dk.slice(sepIdx + 1) : "";

            document.getElementById('n-select').value = defN;
            fillP(defN);
            if (defP) document.getElementById('p-select').value = defP;

            document.getElementById('n-select').addEventListener('change', function() {{
                const n = document.getElementById('n-select').value;
                fillP(n);
                applySelection();
            }});
            document.getElementById('p-select').addEventListener('change', applySelection);

            // --- Wire the "Solution overlay" sentinel checkbox in LayerControl ---
            const mapObj = getMap();
            const sentinelName = PAYLOAD.overlayToggleName;
            if (mapObj && sentinelName) {{
                const sentinel = window[sentinelName];
                if (sentinel) {{
                    // Match the actual initial state of the sentinel on the map.
                    showOverlay = mapObj.hasLayer(sentinel);

                    mapObj.on('overlayadd', function(e) {{
                        if (e.layer === sentinel) {{
                            showOverlay = true;
                            applySelection();
                        }}
                    }});
                    mapObj.on('overlayremove', function(e) {{
                        if (e.layer === sentinel) {{
                            showOverlay = false;
                            applySelection();
                        }}
                    }});
                }}
            }}

            applySelection();
        }}

        // Defer one tick so folium has finished assigning `mapObj` and the layer vars.
        if (document.readyState === "loading") {{
            document.addEventListener('DOMContentLoaded', function() {{ setTimeout(init, 50); }});
        }} else {{
            setTimeout(init, 50);
        }}
    }})();
    </script>
    """
    m.get_root().html.add_child(folium.Element(js))


# -----------------------------
# Diagnostic (unchanged)
# -----------------------------
def print_long_stop_to_stop_jumps(line2stations, pos_ll, threshold_km=3.0):
    for line, stations in line2stations.items():
        stations = [int(x) for x in stations]
        for u, v in zip(stations, stations[1:]):
            if u not in pos_ll or v not in pos_ll:
                continue
            lon1, lat1 = pos_ll[u]
            lon2, lat2 = pos_ll[v]
            d = haversine_km(lon1, lat1, lon2, lat2)
            if d > threshold_km:
                print(
                    f"[jump?] {line}: "
                    f"{u}={ID_TO_NAME.get(u, u)} -> {v}={ID_TO_NAME.get(v, v)} "
                    f"({d:.2f} km)"
                )


# -----------------------------
# Main
# -----------------------------
def plot_solutions_folium(
    solutions=SOLUTIONS,
    default_n=DEFAULT_N,
    default_p=DEFAULT_P,
    out_html=OUT_HTML,
    hypergraph_json=HYPERGRAPH_JSON,
    lines_file=LINES_FILE,
    route_shapes_json=ROUTE_SHAPES_JSON,
    base_tiles="CartoDB positron",
):
    hg = load_hypergraph(hypergraph_json)
    line2stations = load_line2stations_txt(lines_file)
    route_shapes = load_route_shapes(route_shapes_json)
    st2lines = station_lines_index(line2stations)

    # Center over the union of every solution so any choice stays in view.
    union_ids = sorted({v for ids in solutions.values() for v in ids})
    center = map_center_for_nodes(union_ids, NODE_POS)

    m = folium.Map(
        location=center, zoom_start=11,
        tiles=base_tiles, control_scale=True,
    )
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="Dark").add_to(m)
    folium.TileLayer("CartoDB positron", name="Light").add_to(m)

    # --- Route geometry ---
    if route_shapes:
        add_route_shapes_layer(
            m, route_shapes=route_shapes, line_colors=LINE_COLORS,
            show=True, weight=4, opacity=0.58,
        )
    else:
        print("[warn] No route shapes found. Using fallback stop-to-stop segments.")
        add_fallback_stop_segments_layer(
            m, line2stations=line2stations, pos_ll=NODE_POS,
            line_colors=LINE_COLORS, show=True, weight=3, opacity=0.40,
        )

    if SHOW_ALL_STATIONS:
        add_all_stations_layer(
            m, line2stations=line2stations, pos_ll=NODE_POS,
            st2lines=st2lines, show=True,
        )

    # --- Sentinel FeatureGroup that powers the "Solution overlay" checkbox ---
    # It has no markers of its own. Its sole job is to appear in the LayerControl
    # so Leaflet emits `overlayadd` / `overlayremove` events when the user toggles
    # it. The JS listens for those events and shows/hides whichever (n, p) layer
    # is currently selected in the dropdowns. `show=True` makes the box ticked
    # on load, so the default solution is visible.
    overlay_toggle_fg = folium.FeatureGroup(
        name="Solution overlay",
        overlay=True,
        control=True,
        show=True,
    )
    overlay_toggle_fg.add_to(m)

    # --- Build one (hidden) FeatureGroup per (n, p), and remember its JS name ---
    # `make_key` is defined at module level so it stays consistent everywhere.
    layer_name_by_key = {}
    summary_by_key = {}

    for (n, p), ids in solutions.items():
        key = make_key(n, p)
        fg = build_solution_layer(
            m,
            solution_ids=ids,
            pos_ll=NODE_POS,
            st2lines=st2lines,
            name=f"Solution n={n}, p={format_p(p)}",
            show=False,
        )
        layer_name_by_key[key] = fg.get_name()
        summary_by_key[key] = compute_summary(ids, line2stations, st2lines, LINE_COLORS)

    # Resolve default selection
    default_key = make_key(default_n, default_p)
    if (default_n, default_p) not in solutions:
        first = next(iter(solutions))
        default_key = make_key(*first)
        print(
            f"[warn] default (n, p)=({default_n}, {default_p}) not in SOLUTIONS, "
            f"falling back to {first}"
        )

    # Standard LayerControl manages basemaps, static layers, and the sentinel.
    # (Per-(n, p) solution layers stay out of it via control=False.)
    folium.LayerControl(position="bottomright", collapsed=True).add_to(m)

    # Dropdowns + dynamic summary + overlay-toggle wiring
    add_interactive_controls(
        m,
        layer_name_by_key=layer_name_by_key,
        summary_by_key=summary_by_key,
        available_combinations=list(solutions.keys()),
        default_key=default_key,
        overlay_toggle_name=overlay_toggle_fg.get_name(),
    )

    m.save(out_html)
    print(f"[ok] saved {out_html}")
    print(f"[info] nodes in hypergraph: {len(hg.get('nodes', []))}")
    print(f"[info] hyperedges:          {len(hg.get('hyperedges', {}))}")
    print(f"[info] solutions stored:    {len(solutions)}")
    return m


if __name__ == "__main__":
    line2stations = load_line2stations_txt(LINES_FILE)
    print_long_stop_to_stop_jumps(line2stations, NODE_POS, threshold_km=3.0)

    plot_solutions_folium()
