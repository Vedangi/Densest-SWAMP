# nyc_metric_analysis.py

import json
import math
from collections import defaultdict, Counter
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from solution_list import SOLUTIONS

from nyc_line_colors import LINE_COLORS
from nyc_node_positions import NODE_POS, ID_TO_NAME

LINES_FILE = "nyc_lines_hypergraph.txt"



# ============================================================
# Loading helpers
# ============================================================

def load_line2stations_txt(path):
    """
    Reads a file of the form:
        1 10 20 30 ...
        A 4 7 9 ...
    Returns:
        line2stations: dict line_name -> list[int]
    """
    line2stations = {}

    with open(path, "r", encoding="utf-8") as f:
        for row in f:
            row = row.strip()
            if not row:
                continue

            parts = row.split()
            line = parts[0]
            stations = [int(x) for x in parts[1:]]

            if stations:
                line2stations[line] = stations

    return line2stations


def normalize_node_pos(NODE_POS):
    """
    Ensures NODE_POS has int keys and tuple(float, float) values.
    Input format:
        NODE_POS = {1: (-73.98, 40.73), ...}
    """
    out = {}
    for k, v in NODE_POS.items():
        vid = int(k)
        lon, lat = v
        out[vid] = (float(lon), float(lat))
    return out


def normalize_id_to_name(ID_TO_NAME):
    """
    Ensures ID_TO_NAME has int keys.
    """
    return {int(k): str(v) for k, v in ID_TO_NAME.items()}


# ============================================================
# Basic line coverage metrics
# ============================================================

def station_lines_index(line2stations):
    """
    station id -> list of lines containing that station
    """
    idx = defaultdict(list)

    for line, stations in line2stations.items():
        for v in stations:
            idx[int(v)].append(line)

    return dict(idx)


def line_hit_counts(solution_ids, line2stations):
    """
    line -> |S cap line|
    """
    S = set(int(v) for v in solution_ids)

    hits = {}
    for line, stations in line2stations.items():
        hits[line] = sum(1 for v in stations if int(v) in S)

    return hits


def number_of_lines_touched(solution_ids, line2stations):
    hits = line_hit_counts(solution_ids, line2stations)
    return sum(1 for h in hits.values() if h > 0)


def total_line_volume(solution_ids, line2stations, line_weights=None):
    """
    Volume = sum_e w_e |S cap e|.
    """
    hits = line_hit_counts(solution_ids, line2stations)

    total = 0.0
    for line, cnt in hits.items():
        w = 1.0 if line_weights is None else float(line_weights.get(line, 1.0))
        total += w * cnt

    return total


def reward_value_power(solution_ids, line2stations, p, line_weights=None):
    """
    Reward family:
        r(0)=0
        r(i)=i^p for i>=1

    For p=0, this becomes coverage:
        r(i)=1 for i>=1.
    """
    hits = line_hit_counts(solution_ids, line2stations)

    total = 0.0
    for line, cnt in hits.items():
        if cnt == 0:
            val = 0.0
        else:
            val = float(cnt ** float(p))

        w = 1.0 if line_weights is None else float(line_weights.get(line, 1.0))
        total += w * val

    return total


# ============================================================
# One-node-failure robustness metrics
# ============================================================

def lines_touched_set(solution_ids, line2stations):
    """
    Return set of line names touched by S.
    """
    hits = line_hit_counts(solution_ids, line2stations)
    return {line for line, cnt in hits.items() if cnt > 0}


def coverage_after_removing_node(solution_ids, node_to_remove, line2stations):
    """
    Number of lines still touched after removing node_to_remove from S.
    """
    S_after = [int(v) for v in solution_ids if int(v) != int(node_to_remove)]
    return number_of_lines_touched(S_after, line2stations)


def worst_case_coverage_after_one_failure(solution_ids, line2stations, id_to_name=None):
    """
    Advisor's proposed metric:

        min_{v in S} # lines touched by S \\ {v}

    This measures the worst-case line coverage after any one selected
    station fails.

    Returns JSON-safe dictionary.
    """
    S = [int(v) for v in solution_ids]
    coverage_before = number_of_lines_touched(S, line2stations)

    if not S:
        return {
            "coverage_before_failure": 0,
            "worst_coverage_after_failure": 0,
            "worst_coverage_loss": 0,
            "worst_coverage_retention_ratio": 0.0,
            "worst_failed_nodes": [],
            "worst_failed_node_names": [],
            "worst_lost_lines": [],
        }

    touched_before = lines_touched_set(S, line2stations)

    best_after = math.inf
    worst_nodes = []
    worst_lost_lines = []

    for v in S:
        S_after = [u for u in S if u != v]
        touched_after = lines_touched_set(S_after, line2stations)
        coverage_after = len(touched_after)

        if coverage_after < best_after:
            best_after = coverage_after
            worst_nodes = [v]
            worst_lost_lines = sorted(touched_before - touched_after)
        elif coverage_after == best_after:
            worst_nodes.append(v)

    coverage_loss = coverage_before - int(best_after)
    retention = int(best_after) / coverage_before if coverage_before > 0 else 0.0

    if id_to_name is None:
        worst_names = [str(v) for v in worst_nodes]
    else:
        worst_names = [id_to_name.get(v, str(v)) for v in worst_nodes]

    return {
        "coverage_before_failure": int(coverage_before),
        "worst_coverage_after_failure": int(best_after),
        "worst_coverage_loss": int(coverage_loss),
        "worst_coverage_retention_ratio": float(retention),
        "worst_failed_nodes": [int(v) for v in worst_nodes],
        "worst_failed_node_names": worst_names,
        "worst_lost_lines": worst_lost_lines,
    }


def line_reinforcement_metrics(solution_ids, line2stations):
    """
    Counts how many touched lines are covered at least 2 or 3 times.

    This is a direct backup/reinforcement metric.
    """
    hits = line_hit_counts(solution_ids, line2stations)
    positive_hits = [h for h in hits.values() if h > 0]

    touched = len(positive_hits)
    ge2 = sum(1 for h in positive_hits if h >= 2)
    ge3 = sum(1 for h in positive_hits if h >= 3)

    return {
        "n_lines_touched": touched,
        "n_lines_covered_at_least_2": ge2,
        "n_lines_covered_at_least_3": ge3,
        "frac_touched_lines_covered_at_least_2": ge2 / touched if touched else 0.0,
        "frac_touched_lines_covered_at_least_3": ge3 / touched if touched else 0.0,
    }



# ============================================================
# Connectivity graph metrics
# ============================================================

def build_solution_subgraph(solution_ids, line2stations):
    S = set(int(v) for v in solution_ids)

    B = nx.Graph()

    for line, stations in line2stations.items():
        hits = [int(v) for v in stations if int(v) in S]
        if not hits:
            continue

        line_node = f"line::{line}"
        B.add_node(line_node, kind="line")

        for v in hits:
            station_node = f"station::{v}"
            B.add_node(station_node, kind="station")
            B.add_edge(line_node, station_node)

    return B


def connected_component_metrics(solution_ids, line2stations):
    B = build_solution_subgraph(solution_ids, line2stations)
    comps = list(nx.connected_components(B))

    sizes = sorted([len(c) for c in comps], reverse=True)

    return {
        "connected_components": len(comps),
        "component_sizes": sizes,
        "largest_component_size": sizes[0] if sizes else 0,
    }


# ============================================================
# Master metric function
# ============================================================

def compute_solution_metrics(
    solution_ids, 
    line2stations,
    node_pos,
    p,
    n=None,
    id_to_name=None,
    line_weights=None,
):
    S = sorted(set(int(v) for v in solution_ids))

    metrics = {
        "n": int(n) if n is not None else len(S),
        "p": float(p),
        "solution_ids": S,
        "solution_names": [id_to_name.get(v, str(v)) for v in S] if id_to_name else [str(v) for v in S],
        "n_selected_stations": len(S),
    }

    metrics["hit_count_by_line"] = line_hit_counts(S, line2stations)
    metrics["n_lines_touched"] = number_of_lines_touched(S, line2stations)
    metrics["total_volume"] = total_line_volume(S, line2stations, line_weights=line_weights)
    metrics["reward_value"] = reward_value_power(S, line2stations, p=p, line_weights=line_weights)
    metrics["avg_reward_per_station"] = (
        metrics["reward_value"] / len(S) if len(S) > 0 else 0.0
    )

    metrics.update(line_reinforcement_metrics(S, line2stations))
    metrics.update(worst_case_coverage_after_one_failure(S, line2stations, id_to_name=id_to_name))

    metrics.update(connected_component_metrics(S, line2stations))

    return metrics

# ===========================================
# Convert tuple-keyed solution to nested dict
# ===========================================

def convert_tuplekey_solutions(SOLUTIONS):
    """
    Converts:
        SOLUTIONS = {
            (20, 1.0): [402, 326, ...],
            (20, 0.95): [78, 402, ...],
            (16, 1.0): [...],
        }

    into:
        {
            20: {
                1.0: [...],
                0.95: [...],
            },
            16: {
                1.0: [...],
            }
        }
    """
    out = defaultdict(dict)

    for key, sol in SOLUTIONS.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError(f"Expected key of form (n, p), got {key}")

        n, p = key
        n = int(n)
        p = float(p)

        out[n][p] = [int(v) for v in sol]

    return {n: dict(pdict) for n, pdict in out.items()}

# ============================================================
# Build metrics grid
# ============================================================
def build_metrics_grid(
    solutions_by_n_p,
    line2stations,
    node_pos,
    id_to_name=None,
    line_weights=None,
    dataset_name="nyc_subway",
):
    """
    solutions_by_n_p format:
        {
            8: {
                0.0: [station ids],
                0.2: [station ids],
                ...
            },
            12: {
                0.0: [station ids],
                ...
            }
        }

    Returns a JSON-safe dictionary.
    """
    out = {
        "meta": {
            "dataset": dataset_name,
            "description": "NYC subway CARD-SWAMP metrics over selected-set size n and reward parameter p.",
            "n_values": sorted([int(n) for n in solutions_by_n_p.keys()]),
            "p_values": sorted(
                list({
                    float(p)
                    for n in solutions_by_n_p
                    for p in solutions_by_n_p[n].keys()
                })
            ),
        },
        "results": {},
    }

    for n in sorted(solutions_by_n_p.keys(), key=lambda x: int(x)):
        n_key = str(int(n))
        out["results"][n_key] = {}

        p_dict = solutions_by_n_p[n]
        for p in sorted(p_dict.keys(), key=lambda x: float(x)):
            p_key = str(float(p))
            sol = p_dict[p]

            out["results"][n_key][p_key] = compute_solution_metrics(
                solution_ids=sol,
                line2stations=line2stations,
                node_pos=node_pos,
                p=float(p),
                n=int(n),
                id_to_name=id_to_name,
                line_weights=line_weights,
            )

    return out


def save_metrics_grid(metrics_grid, out_json):
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics_grid, f, indent=2)

    print(f"[ok] wrote {out_json}")


def load_metrics_grid(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    

# =============================================
#  Plotting function
#  ============================================
def get_metric_series(metrics_grid, metric_name):
    """
    Returns:
        series[n] = (p_values, y_values)
    """
    results = metrics_grid["results"]

    series = {}

    for n_key in sorted(results.keys(), key=lambda x: int(x)):
        p_keys = sorted(results[n_key].keys(), key=lambda x: float(x))

        xs = []
        ys = []

        for p_key in p_keys:
            metrics = results[n_key][p_key]
            xs.append(float(p_key))
            ys.append(metrics.get(metric_name, np.nan))

        series[int(n_key)] = (xs, ys)

    return series


def plot_metric_by_n_over_p(
    metrics_grid,
    metric_name,
    outfile=None,
    title=None,
    ylabel=None,
):
    """
    One figure.
    x-axis: p
    y-axis: metric
    one colored line per n
    """
    series = get_metric_series(metrics_grid, metric_name)

    fig, ax = plt.subplots(figsize=(8, 5))

    for n, (xs, ys) in series.items():
        ax.plot(xs, ys, marker="o", linewidth=2, label=f"n={n}")

    ax.set_xlabel("reward parameter p")
    ax.set_ylabel(ylabel or metric_name.replace("_", " "))
    ax.set_title(title or f"{metric_name} vs p")
    ax.grid(False)
    # ax.grid(True, alpha=0.3)
    ax.legend(title="selected stations")

    plt.tight_layout()

    if outfile is not None:
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
        print(f"[ok] wrote {outfile}")

    plt.show()

import numpy as np
import matplotlib.pyplot as plt


def sort_p_key(p):
    """
    Sort p-values numerically even when stored as JSON strings.
    """
    return float(p)


def format_p_label(p):
    """
    Nice labels for p-values on categorical x-axis.
    """
    x = float(p)

    if x == 0:
        return "0"
    if x < 1e-3:
        return f"{x:.0e}"
    if x < 0.01:
        return f"{x:g}"
    return f"{x:g}"


def get_all_p_values(metrics_grid):
    """
    Collect all p-values appearing for any n.
    Returns sorted list of p-keys as strings.
    """
    all_p = set()

    for n_key, pdict in metrics_grid["results"].items():
        for p_key in pdict.keys():
            all_p.add(p_key)

    return sorted(all_p, key=lambda p: float(p))


def plot_metric_by_n_over_p_categorical(
    metrics_grid,
    metric_name,
    outfile=None,
    title=None,
    ylabel=None,
    legend_flag = True,
    rotate_labels=35,
):
    """
    Plot one metric over p.
    Each p-value gets its own equally spaced x-position.

    This prevents tiny values like 1e-9, 1e-7, 1e-5 from collapsing near 0.
    """
    grid_flag = False
    results = metrics_grid["results"]


    p_keys = get_all_p_values(metrics_grid)
    x = np.arange(len(p_keys))
    xlabels = [format_p_label(p) for p in p_keys]

    fig, ax = plt.subplots(figsize=(8, 5)) #10,5.5

    for n_key in sorted(results.keys(), key=lambda z: int(z)):
        y = []

        for p in p_keys:
            if p in results[n_key]:
                y.append(results[n_key][p].get(metric_name, np.nan))
            else:
                y.append(np.nan)

        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2,
            label="$\ell=$" + str(n_key)
        )

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=rotate_labels, ha="right", fontsize=18)

    ax.set_xlabel("reward parameter $p$", fontsize=20)

    ax.tick_params(axis="y", labelsize=18)
    ax.set_ylabel(ylabel or metric_name.replace("_", " "), fontsize=20)
    
    # ax.set_title(title or f"{metric_name} vs p", fontsize=20)
    ax.grid(False) 
    # ax.grid(True, alpha=0.3)
    
    if legend_flag:
        ax.legend(fontsize=12,title_fontsize=12)

    plt.tight_layout()

    if outfile is not None:
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
        print(f"[ok] wrote {outfile}")

    plt.show()

    
# ============================================================
# Main execution
# ============================================================

if __name__ == "__main__":
    line2stations = load_line2stations_txt(LINES_FILE)
    node_pos = normalize_node_pos(NODE_POS)
    id_to_name = normalize_id_to_name(ID_TO_NAME)

    #Run this once to build the metrics grid and save to JSON. Then we can load from JSON and plot without recomputing all metrics.

    solutions_by_n_p = convert_tuplekey_solutions(SOLUTIONS)

    metrics_grid = build_metrics_grid(
        solutions_by_n_p=solutions_by_n_p,
        line2stations=line2stations,
        node_pos=node_pos,
        id_to_name=id_to_name,
        line_weights=None,  
        dataset_name="nyc_subway",
    )

    save_metrics_grid(metrics_grid, "nyc_solution_metrics_grid.json")


    # Now load the precomputed metrics grid and plot some metrics.
    metrics_grid = load_metrics_grid("nyc_solution_metrics_grid.json")

    

    plot_metric_by_n_over_p_categorical(
        metrics_grid,
        "n_lines_touched",
        outfile="nyc_lines_touched_by_n_categorical.pdf",
        title= "Number of routes covered as $p$ changes",
        ylabel=" #routes covered",
        legend_flag=True,
    )


    plot_metric_by_n_over_p_categorical(
        metrics_grid,
        "worst_coverage_retention_ratio",
        outfile="nyc_coverage_retention_ratio_by_n_categorical.pdf",
        title="Worst-case coverage retention ratio",
        ylabel="retained coverage ratio",
        legend_flag=False,
    )

  
    plot_metric_by_n_over_p_categorical(
        metrics_grid,
        "total_volume",
        outfile="nyc_volume_by_n_categorical.pdf",
        title="Total line-station volume",
        ylabel="total volume",
        legend_flag=False,
    )

    

   

    plot_metric_by_n_over_p_categorical(
        metrics_grid,
        "connected_components",
        outfile="nyc_components_by_n_categorical.pdf",
        title="Number of connected components ",
        ylabel="#connected components",
        legend_flag=True,
    )
