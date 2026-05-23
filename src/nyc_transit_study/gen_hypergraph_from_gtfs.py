"""
Build a clean NYC subway hypergraph directly from a GTFS static feed.

Main modeling choice:
    - Node = one physical station, represented by the GTFS parent_station id
      when available; otherwise by the stop_id itself.
    - Hyperedge = one subway route/line, represented as an unordered set of
      physical stations served by that route.

This avoids the main source of the map jumps: matching stations only by name.
NYC has repeated names such as "86 St", "23 St", "Canal St", etc., so names
should be labels, not unique identifiers.

Outputs:
    nyc_hypergraph_parent_ids.json
        Main file for algorithms + plotting.

    nyc_routes_hypergraph.txt
        Simple route-to-node-id text format:
            ROUTE_ID node1 node2 node3 ...

    nyc_node_positions.py
        NODE_POS = {node_id: (lon, lat)}
        ID_TO_NAME = {node_id: station_name}
        ID_TO_PARENT = {node_id: parent_station_id}

    nyc_route_colors.py
        ROUTE_COLORS = {route_id: "#RRGGBB"}

    nyc_route_shapes.json
        Optional visualization geometry from GTFS shapes.txt.
        This is for drawing clean map lines, not for defining hyperedges.
"""

import csv
import json
import zipfile
from collections import defaultdict, Counter
from pathlib import Path


# ----------------------------
# Config
# ----------------------------
GTFS_ZIP_PATH = "gtfs_subway.zip"

OUT_HYPERGRAPH_JSON = "nyc_hypergraph_parent_ids.json"
OUT_ROUTES_TXT = "nyc_routes_hypergraph.txt"
OUT_NODE_POS_PY = "nyc_node_positions.py"
OUT_ROUTE_COLORS_PY = "nyc_route_colors.py"
OUT_ROUTE_SHAPES_JSON = "nyc_route_shapes.json"

# Usually False is best for the hypergraph:
#   False: one hyperedge per route/line, e.g., A, C, 1, 2, 3
#   True: one hyperedge per route-direction, e.g., A|d=0, A|d=1
SPLIT_BY_DIRECTION = False

# If True, route_id is used as the hyperedge key.
# If False, route_short_name is used when present.
USE_ROUTE_ID_AS_LABEL = True

DEFAULT_COLOR = "#666666"


# ----------------------------
# Basic utilities
# ----------------------------
def read_csv_from_zip(zf: zipfile.ZipFile, filename: str):
    """Read a GTFS CSV file from a zip, case-insensitively."""
    for zi in zf.infolist():
        if zi.filename.lower() == filename.lower():
            with zf.open(zi) as f:
                text = f.read().decode("utf-8-sig")
            return list(csv.DictReader(text.splitlines()))
    raise FileNotFoundError(f"{filename} not found in {zf.filename}")


def safe_float(x):
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def normalize_hex_color(c, default=DEFAULT_COLOR):
    if not c:
        return default
    c = str(c).strip()
    if len(c) == 6 and all(ch in "0123456789abcdefABCDEF" for ch in c):
        return "#" + c.upper()
    if c.startswith("#") and len(c) == 7:
        return c.upper()
    return default


def write_python_dict(path, var_name, dct, header=None):
    """Write a small Python file containing one dictionary variable."""
    with open(path, "w", encoding="utf-8") as f:
        if header:
            f.write(header.rstrip() + "\n")
        f.write(f"{var_name} = {{\n")
        for k in sorted(dct):
            f.write(f"    {repr(k)}: {repr(dct[k])},\n")
        f.write("}\n")


# ----------------------------
# GTFS station handling
# ----------------------------
def physical_station_id(stop_id, stop_by_id):
    """
    Collapse platform/direction-level stops to the parent station.

    In many GTFS feeds, child stops represent platforms, entrances, or directions,
    while parent_station represents the physical station. For the hypergraph, use
    the parent station as the node.
    """
    row = stop_by_id.get(stop_id)
    if row is None:
        return stop_id

    parent = (row.get("parent_station") or "").strip()
    if parent and parent in stop_by_id:
        return parent

    return stop_id


def station_name(parent_id, stop_by_id):
    row = stop_by_id.get(parent_id, {})
    return (row.get("stop_name") or parent_id).strip()


def station_lonlat(parent_id, stop_by_id):
    row = stop_by_id.get(parent_id, {})
    lon = safe_float(row.get("stop_lon"))
    lat = safe_float(row.get("stop_lat"))
    return lon, lat


def is_station_like(row):
    """
    Keep station-like rows. Some GTFS feeds use location_type:
        0 or blank = stop/platform
        1 = station
        2 = entrance/exit
        3 = generic node
        4 = boarding area

    For parent stations, location_type may be 1.
    For simple feeds, location_type may be blank.
    """
    loc_type = (row.get("location_type") or "").strip()
    return loc_type in {"", "0", "1"}


# ----------------------------
# Build hypergraph
# ----------------------------
def build_hypergraph_from_gtfs(gtfs_zip_path: str):
    zf = zipfile.ZipFile(gtfs_zip_path)

    stops = read_csv_from_zip(zf, "stops.txt")
    routes = read_csv_from_zip(zf, "routes.txt")
    trips = read_csv_from_zip(zf, "trips.txt")
    stop_times = read_csv_from_zip(zf, "stop_times.txt")

    stop_by_id = {row["stop_id"]: row for row in stops if row.get("stop_id")}
    route_by_id = {row["route_id"]: row for row in routes if row.get("route_id")}

    def route_label(route_id):
        r = route_by_id.get(route_id, {})
        if USE_ROUTE_ID_AS_LABEL:
            return route_id
        return (r.get("route_short_name") or r.get("route_long_name") or route_id).strip()

    # trip_id -> route/direction metadata
    trip_to_route_dir = {}
    for t in trips:
        tid = t.get("trip_id")
        rid = t.get("route_id")
        if not tid or not rid:
            continue
        direction = (t.get("direction_id") or "").strip()
        trip_to_route_dir[tid] = (rid, direction)

    # route/hyperedge -> set(parent_station_id)
    edge_to_parent_stations = defaultdict(set)

    for st in stop_times:
        tid = st.get("trip_id")
        sid = st.get("stop_id")

        if tid not in trip_to_route_dir or sid not in stop_by_id:
            continue

        rid, direction = trip_to_route_dir[tid]
        label = route_label(rid)
        edge_key = f"{label}|d={direction}" if SPLIT_BY_DIRECTION else label

        parent_id = physical_station_id(sid, stop_by_id)
        if parent_id in stop_by_id and is_station_like(stop_by_id[parent_id]):
            edge_to_parent_stations[edge_key].add(parent_id)

    # Remove empty or one-station hyperedges.
    edge_to_parent_stations = {
        edge: sorted(parent_ids)
        for edge, parent_ids in edge_to_parent_stations.items()
        if len(parent_ids) >= 2
    }

    # Assign stable integer ids to physical stations.
    all_parent_ids = sorted(
        set().union(*edge_to_parent_stations.values()),
        key=lambda p: (station_name(p, stop_by_id).lower(), p),
    )

    parent_to_node_id = {parent_id: i + 1 for i, parent_id in enumerate(all_parent_ids)}

    nodes = []
    node_pos = {}
    id_to_name = {}
    id_to_parent = {}

    for parent_id in all_parent_ids:
        node_id = parent_to_node_id[parent_id]
        name = station_name(parent_id, stop_by_id)
        lon, lat = station_lonlat(parent_id, stop_by_id)

        nodes.append({
            "node_id": node_id,
            "parent_station": parent_id,
            "name": name,
            "lon": lon,
            "lat": lat,
        })

        id_to_name[node_id] = name
        id_to_parent[node_id] = parent_id

        if lon is not None and lat is not None:
            node_pos[node_id] = (lon, lat)

    hyperedges = {}
    for edge_key, parent_ids in sorted(edge_to_parent_stations.items()):
        hyperedges[edge_key] = sorted(parent_to_node_id[p] for p in parent_ids)

    # Route colors
    route_colors = {}
    for rid, r in route_by_id.items():
        label = route_label(rid)
        color = normalize_hex_color(r.get("route_color"))
        if SPLIT_BY_DIRECTION:
            continue
        if label in hyperedges:
            route_colors[label] = color

    if SPLIT_BY_DIRECTION:
        base_to_color = {
            route_label(rid): normalize_hex_color(r.get("route_color"))
            for rid, r in route_by_id.items()
        }
        for edge_key in hyperedges:
            base = edge_key.split("|d=")[0]
            route_colors[edge_key] = base_to_color.get(base, DEFAULT_COLOR)

    hg = {
        "metadata": {
            "node_definition": "GTFS parent_station if available, else stop_id",
            "hyperedge_definition": (
                "route_id + direction_id" if SPLIT_BY_DIRECTION else "route_id"
            ),
            "gtfs_zip_path": gtfs_zip_path,
            "split_by_direction": SPLIT_BY_DIRECTION,
        },
        "nodes": nodes,
        "hyperedges": hyperedges,
    }

    return hg, node_pos, id_to_name, id_to_parent, route_colors


# ----------------------------
# Build route shapes for visualization
# ----------------------------
def build_route_shapes_from_gtfs(gtfs_zip_path: str):
    """
    Build route geometry from shapes.txt, if present.

    Output format:
        {
            "A": [
                {"shape_id": "...", "direction_id": "0", "points": [[lat, lon], ...]},
                ...
            ],
            ...
        }

    This is only for visualization. Hyperedges remain unordered sets of station nodes.
    """
    zf = zipfile.ZipFile(gtfs_zip_path)

    try:
        shapes = read_csv_from_zip(zf, "shapes.txt")
    except FileNotFoundError:
        print("[warn] shapes.txt not found. Map will fall back to stop-to-stop segments.")
        return {}

    routes = read_csv_from_zip(zf, "routes.txt")
    trips = read_csv_from_zip(zf, "trips.txt")

    route_by_id = {row["route_id"]: row for row in routes if row.get("route_id")}

    def route_label(route_id):
        r = route_by_id.get(route_id, {})
        if USE_ROUTE_ID_AS_LABEL:
            return route_id
        return (r.get("route_short_name") or r.get("route_long_name") or route_id).strip()

    # shape_id -> ordered points
    shape_points = defaultdict(list)
    for row in shapes:
        sid = row.get("shape_id")
        if not sid:
            continue
        lat = safe_float(row.get("shape_pt_lat"))
        lon = safe_float(row.get("shape_pt_lon"))
        seq_raw = row.get("shape_pt_sequence")
        if lat is None or lon is None or seq_raw is None:
            continue
        try:
            seq = int(seq_raw)
        except Exception:
            continue
        shape_points[sid].append((seq, lat, lon))

    shape_points = {
        sid: [[lat, lon] for seq, lat, lon in sorted(points)]
        for sid, points in shape_points.items()
        if len(points) >= 2
    }

    # route/direction -> most common shape ids.
    route_dir_shape_counts = defaultdict(Counter)
    for t in trips:
        rid = t.get("route_id")
        shape_id = t.get("shape_id")
        if not rid or not shape_id or shape_id not in shape_points:
            continue
        direction = (t.get("direction_id") or "").strip()
        label = route_label(rid)
        route_dir_shape_counts[(label, direction)][shape_id] += 1

    route_shapes = defaultdict(list)

    for (label, direction), counter in sorted(route_dir_shape_counts.items()):
        # Keep the most common shape per route/direction. This avoids drawing
        # many schedule variants, while still keeping both directions.
        for shape_id, count in counter.most_common(1):
            route_shapes[label].append({
                "shape_id": shape_id,
                "direction_id": direction,
                "trip_count": count,
                "points": shape_points[shape_id],
            })

    return dict(route_shapes)


# ----------------------------
# Write outputs
# ----------------------------
def write_outputs(
    hg,
    node_pos,
    id_to_name,
    id_to_parent,
    route_colors,
    route_shapes,
):
    with open(OUT_HYPERGRAPH_JSON, "w", encoding="utf-8") as f:
        json.dump(hg, f, indent=2)

    with open(OUT_ROUTES_TXT, "w", encoding="utf-8") as f:
        for edge_key, node_ids in sorted(hg["hyperedges"].items()):
            f.write(edge_key + " " + " ".join(map(str, node_ids)) + "\n")

    with open(OUT_NODE_POS_PY, "w", encoding="utf-8") as f:
        f.write("# Auto-generated from GTFS parent_station ids.\n")
        f.write("NODE_POS = {\n")
        for node_id in sorted(node_pos):
            lon, lat = node_pos[node_id]
            f.write(f"    {node_id}: ({lon:.8f}, {lat:.8f}),\n")
        f.write("}\n\n")

        f.write("ID_TO_NAME = {\n")
        for node_id in sorted(id_to_name):
            f.write(f"    {node_id}: {repr(id_to_name[node_id])},\n")
        f.write("}\n\n")

        f.write("ID_TO_PARENT = {\n")
        for node_id in sorted(id_to_parent):
            f.write(f"    {node_id}: {repr(id_to_parent[node_id])},\n")
        f.write("}\n")

    write_python_dict(
        OUT_ROUTE_COLORS_PY,
        "ROUTE_COLORS",
        route_colors,
        header="# Auto-generated from GTFS routes.txt",
    )

    with open(OUT_ROUTE_SHAPES_JSON, "w", encoding="utf-8") as f:
        json.dump(route_shapes, f)

    print(f"[ok] wrote {OUT_HYPERGRAPH_JSON}")
    print(f"[ok] wrote {OUT_ROUTES_TXT}")
    print(f"[ok] wrote {OUT_NODE_POS_PY}")
    print(f"[ok] wrote {OUT_ROUTE_COLORS_PY}")
    print(f"[ok] wrote {OUT_ROUTE_SHAPES_JSON}")

    print()
    print("Summary")
    print("-------")
    print(f"nodes:      {len(hg['nodes'])}")
    print(f"hyperedges: {len(hg['hyperedges'])}")
    print(f"positions:  {len(node_pos)}")
    print(f"shapes:     {sum(len(v) for v in route_shapes.values())}")


def main():
    if not Path(GTFS_ZIP_PATH).exists():
        raise FileNotFoundError(
            f"Could not find {GTFS_ZIP_PATH}. Put the GTFS zip in this folder "
            "or edit GTFS_ZIP_PATH at the top of this file."
        )

    hg, node_pos, id_to_name, id_to_parent, route_colors = build_hypergraph_from_gtfs(
        GTFS_ZIP_PATH
    )
    route_shapes = build_route_shapes_from_gtfs(GTFS_ZIP_PATH)

    write_outputs(
        hg=hg,
        node_pos=node_pos,
        id_to_name=id_to_name,
        id_to_parent=id_to_parent,
        route_colors=route_colors,
        route_shapes=route_shapes,
    )


if __name__ == "__main__":
    main()
