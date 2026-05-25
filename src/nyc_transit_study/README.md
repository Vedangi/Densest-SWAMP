# NYC Subway Hypergraph Application

This project builds a hypergraph from an NYC subway GTFS static feed, runs hypergraph algorithms on the resulting network, and visualizes selected solution stations on an interactive Folium map.

The hypergraph is modeled as:

- **Node:** one physical subway station
- **Hyperedge:** one route
- **Solution:** a selected subset of station nodes, saved in `solution_list.py` and highlighted on the map

A station served by multiple routes is called a transfer station.

---
## Input data

The input is a GTFS static feed, which can be downloaded at [https://www.mta.info/developers](https://www.mta.info/developers) and stored as a zip file called "gtfs_subway.zip".

## Main source files

`gen_nyc.py`: Processes GTFS zip file, and creates hypergraph dataset

`nyc_plot_folium.py`: Reads the generated files the html "nyc_solution_map.html " file, loads solution nodes from `solution_list.py`, and creates an interactive map.

The preprocessing script uses the following GTFS files.

### `stops.txt`

Used to define physical station nodes and station coordinates.

Relevant columns:

| Column | Use |
|---|---|
| `stop_id` | GTFS identifier for a stop, platform, or station |
| `stop_name` | Human-readable station name |
| `stop_lat` | Station latitude |
| `stop_lon` | Station longitude |
| `parent_station` | Groups platform-level stops into one physical station |
| `location_type` | Indicates whether the row is a station, stop/platform, entrance, etc. |

We use `parent_station` to define physical stations. If a stop has a valid `parent_station`, we map it to that parent station. Otherwise, we use the stop's own `stop_id`.

Station names are used only as labels, not as unique identifiers. This is important because NYC has different physical stations with the same name, such as different stations named `181 St`, `86 St`, or `Canal St`.

### `routes.txt`

Used to define subway-line hyperedges and line colors.

Relevant columns:

| Column | Use |
|---|---|
| `route_id` | Subway line identifier; used as the hyperedge name |
| `route_short_name` | Display name for the line, when available |
| `route_long_name` | Longer route description, when available |
| `route_color` | Line color used in the map |

Each `route_id` corresponds to one hyperedge in the hypergraph.

### `trips.txt`

Used to connect scheduled trips to subway lines.

Relevant columns:

| Column | Use |
|---|---|
| `trip_id` | Unique scheduled trip identifier |
| `route_id` | Subway line associated with the trip |
| `direction_id` | Direction of travel; not used by default in the hypergraph |
| `shape_id` | Geometry identifier used for map drawing |

By default, directions are not treated as separate hyperedges. Both directions of a route contribute to the same subway-line hyperedge.

### `stop_times.txt`

Used to determine which stations are served by each subway line.

Relevant columns:

| Column | Use |
|---|---|
| `trip_id` | Scheduled trip identifier |
| `stop_id` | Stop/platform visited by the trip |
| `stop_sequence` | Order of the stop within the trip |

For every stop event, the code finds the trip's `route_id`, maps the `stop_id` to its physical station using `parent_station`, and adds that station to the route hyperedge.

### `shapes.txt`

Used only for drawing subway lines on the Folium map.

Relevant columns:

| Column | Use |
|---|---|
| `shape_id` | Route geometry identifier |
| `shape_pt_lat` | Latitude of a geometry point |
| `shape_pt_lon` | Longitude of a geometry point |
| `shape_pt_sequence` | Order of the geometry point |

This file is not used to define the hypergraph. It is only used for visualization so that subway lines appear clean on the map.

---

## Step 1: Generate the NYC hypergraph

Run:

```bash
python gen_nyc.py
```

This script reads the GTFS feed and creates the processed files needed by the algorithm and the map.

The construction is:

```text
physical station node = parent_station if available, otherwise stop_id
route hyperedge = set of physical stations served by a route_id
```

Duplicate station visits on the same route are removed because each hyperedge is represented as a set of nodes.

---

## Generated files

### `nyc_hypergraph_parent_ids.json`

Main processed hypergraph dataset.

Example structure:

```json
{
  "nodes": [
    {
      "node_id": 60,
      "parent_station": "111",
      "name": "181 St",
      "lon": -73.933596,
      "lat": 40.849505
    }
  ],
  "hyperedges": {
    "1": [60, 62, 65, 70],
    "A": [61, 63, 68, 72]
  }
}
```

The `nodes` list stores station metadata. The `hyperedges` dictionary maps each route to the node IDs of the stations served by that route.

### `nyc_routes_hypergraph.txt`

Simple text version of the hyperedges.

Example:

```text
1 60 62 65 70
A 61 63 68 72
C 61 63 69 75
```

Each row contains:

```text
line_id node_1 node_2 ... node_k
```

This file is useful for loading the hypergraph into Julia or another algorithmic pipeline.

Example Julia loader:

```julia
edges_list = [
    parse.(Int, split(strip(line))[2:end])
    for line in eachline("nyc_lines_hypergraph_subway.txt")
    if !isempty(strip(line))
]
```

Then `edges_list[i]` is the set of station nodes in the `i`-th subway-line hyperedge.

### `nyc_node_positions.py`

Stores station coordinates and labels for plotting.

Example:

```python
NODE_POS = {
    60: (-73.933596, 40.849505),
    61: (-73.937969, 40.851695),
}

ID_TO_NAME = {
    60: "181 St",
    61: "181 St",
}

ID_TO_PARENT = {
    60: "111",
    61: "A06",
}
```

`NODE_POS` maps each node ID to `(longitude, latitude)`. The plotting script uses this file to place station markers on the map.

### `nyc_line_colors.py`

Stores route colors from `routes.txt`.

Example:

```python
LINE_COLORS = {
    "1": "#EE352E",
    "A": "#0039A6",
    "C": "#0039A6"
}
```

This file is used only for map visualization.

### `nyc_route_shapes_subway.json`

Stores route geometry extracted from `shapes.txt`.

Example:

```json
{
  "A": [
    {
      "shape_id": "shape_A_north",
      "direction_id": "0",
      "points": [
        [40.851695, -73.937969],
        [40.852300, -73.936800],
        [40.853000, -73.935400]
      ]
    }
  ]
}
```

The plotting script uses this file to draw subway routes as map polylines.

---

## Step 2: Add solution nodes

A demo code that optimally solves submodular maximization on the nyc hypergraph is given in `submod_maximization.jl`
This code is run for different values of (|S|,p) and stored in:

```text
solution_list.py
```

The expected structure is a dictionary where each key is a parameter pair `(|S|, p)` and each value is a list of selected station node IDs.

Example:

```python
SOLUTIONS = {
    (10, 0.0): [60, 61, 75, 102, 118, 203, 244, 310, 400, 455],
    (12, 0.0): [60, 61, 75, 102, 118, 203, 244, 310, 350, 400, 455, 470],
}
```

Here, `|S|` is the desired solution size and `p` is the parameter value used in the experiment.

The node IDs must correspond to the `node_id` values in `nyc_hypergraph_parent_ids.json`.

---

## Step 3: Plot the map

Run:

```bash
python nyc_plot_folium.py
```

This script reads:

```text
nyc_hypergraph_parent_ids.json
nyc_lines_hypergraph.txt
nyc_node_positions.py
nyc_line_colors.py
nyc_route_shapes.json
solution_list.py
```

and writes:

```text
nyc_solution_map_subway.html
```

The output is an interactive Folium map with:

- subway route geometry from `nyc_route_shapes_subway.json`
- all physical station nodes as context markers
- selected solution stations highlighted
- tooltips showing station name, node ID, parent station ID, and lines serving the station
- dropdowns for choosing the solution parameters `(n, p)`

---

## Notes on node IDs

Node IDs are generated during preprocessing. If the GTFS feed or preprocessing logic changes, the node IDs may change. Therefore, solution lists should always be generated using the same processed hypergraph file that is used for plotting.

Do not use station names as unique identifiers. Different physical stations may have the same display name. The unique physical station identifier is the GTFS `parent_station` value, stored in the processed data as `parent_station`.

---

## Plotting the metrics in the paper
To reproduce the metric plots in the `figures` folder, run `plot_metrics.py`.

## Minimal reproducibility checklist

To reproduce the same dataset and map, keep the following fixed:

1. GTFS static feed zip file
2. `gen_nyc.py`
3. generated hypergraph files
4. algorithm output stored in `solution_list.py`
5. `nyc_plot_folium.py`

A recommended project layout is:

```text
project/
├── gtfs_subway.zip
├── gen_nyc.py
├── nyc_plot_folium.py
├── solution_list.py
├── nyc_hypergraph_parent_ids_subway.json
├── nyc_lines_hypergraph_subway.txt
├── nyc_node_positions_subway.py
├── nyc_line_colors_subway.py
├── nyc_route_shapes_subway.json
├── plot_metrics.py
└── nyc_solution_map_subway.html

```
