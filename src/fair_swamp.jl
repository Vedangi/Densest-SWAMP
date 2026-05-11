include("header.jl")
include("card_swamp.jl")


"""
FAIR densest SWAMP problem: find a subset of vertices S that maximizes density while satisfying fairness constraints on node labels.

Assumptions / conventions:
- Node ids are 1..n (at least up to max node id in edges).
- node_label_data is n×q one-hot (sparse is fine): node_label_data[v,j] != 0 iff node v has label j.
- Each node has exactly ONE label.
- l[j] is the required minimum count for label j.
"""

function fair_densest_subgraph_edge_list(
    edges_list::Vector{Vector{Int64}},
    vertex2edges::Dict{Int64,Vector{Int64}},
    l::Vector{Int64},
    r::Vector{Vector{Float64}},
    node_label_data::SparseMatrixCSC,
    label_names::Vector{Any}
)
    #Total nodes from hyperedges (no isolated nodes as per assumption)
    n = max_node_id(edges_list)
    V = Set(unique(vcat(edges_list...)))
    q = length(label_names)
    @assert size(node_label_data, 1) == n "node_label_data must have n rows"
    @assert size(node_label_data, 2) == q "node_label_data must have q columns"

    edges_list_copy = deepcopy(edges_list)
    vertex2edges_copy = deepcopy(vertex2edges)
    reward_d = deepcopy(r)

    # Build nodes_by_label[j] and label_of[v] using the sparse one-hot matrix
    nodes_by_label = Vector{Vector{Int}}(undef, q)
    total_counts_by_label = Vector{Int}(undef, q)
    label_of = zeros(Int, n)   # label_of[v] = j

    for i in 1:q
        col = node_label_data[:, i]
        nodes = findnz(col)[1]
        nodes_by_label[i] = nodes
        total_counts_by_label[i] = length(nodes)
        @inbounds for v in nodes
            if v <= n
                if label_of[v] != 0
                    error("Node $v appears in multiple labels (not allowed).")
                end
                label_of[v] = i
            end
        end
    end
    @assert all(label_of .!= 0) "Some nodes have no label"

    if any(l[j] > total_counts_by_label[j] for j in 1:q)
        error("Fairness constraint cannot be satisfied: l[j] exceeds total nodes with label j.")
    end

    counts_by_label = zeros(Int, q)
    unmet = count(j -> counts_by_label[j] < l[j], 1:q)  # initially count(l .> 0) but keep generic


    # Maintain chosen set S via:
    # - inS[v] for O(1) membership
    # - Slist for cheap iteration / building candidates
    inS = falses(n)
    Slist = Int[]

    best_set = Set{Int}()
    best_density = -Inf


    # S is our global set of chosen vertices
    S = Set{Int}()


    # S_i_list = Dict{Int,Set{Int}}()
    # S_i_padded_list = Dict{Int,Set{Int}}()

    iter = 0
    # S_i_list[0] = Set{Int}()
    # We'll do an iterative approach until we have e nodes or hypergraph is empty
    while unmet > 0 && length(Slist) < n

        iter += 1

        if !any(!isempty, edges_list_copy)
            break
        end

        # 1) Densest SWAMP on contracted instance
        S_i_set, _ = densest(edges_list_copy, reward_d)

        if isempty(S_i_set)
            break
        end
        new_nodes = Int[]
        @inbounds for v in S_i_set
            if 1 <= v <= n && inS[v]
                error("Error: node $v already in S, but selected again in S_i_set. This should not happen.")
            end
            inS[v] = true
            push!(Slist, v)
            push!(new_nodes, v)

            lbl = label_of[v]
            # update unmet if this label just became satisfied
            if counts_by_label[lbl] < l[lbl] && counts_by_label[lbl] + 1 >= l[lbl]
                unmet -= 1
            end
            counts_by_label[lbl] += 1

        end
        # 3) Build padded candidate W' (ARBITRARY padding, but done deterministically)
        pad_nodes = Int[]
        for j in 1:q
            need = l[j] - counts_by_label[j]
            need <= 0 && continue

            # choose the first `need` nodes of label j not already in W
            @inbounds for v in nodes_by_label[j]
                if v <= n && !inS[v]
                    push!(pad_nodes, v)
                    need -= 1
                    need == 0 && break
                end
            end
            need == 0 || error("Not enough nodes to satisfy fairness for label $j (should be impossible after feasibility check).")
        end

        # Candidate node list for density evaluation 
        candidate_nodes = isempty(pad_nodes) ? Slist : vcat(Slist, pad_nodes)

        # 4) Evaluate density on ORIGINAL instance (edges_list, r)
        dens = hedensity_non_uniform_edge_list(edges_list, candidate_nodes, r)
        if dens > best_density
            best_density = dens
            best_set = Set(candidate_nodes)   # only allocate when we improve
        end
        # If already satisfied in S (un-padded), you can stop 
        if unmet == 0
            break
        end


        reward_d = remove_nodes!(edges_list_copy, reward_d, vertex2edges_copy, new_nodes)

    end

    return best_set, best_density
end

#Example usage:using SparseArrays

# ----------------------------
# Load senate-committees
# ----------------------------
dataname = "senate-committees"
data = matread("Densest-SWAMP/datafolder/large-datasets/senate-committees.mat")

H = sparse(Float64.(data["H"]))
edges = incidence2elist(H)

v_to_edges = build_vertex_to_edges(edges)

# rewards for each hyperedge (choose one)
order = vec(round.(Int64, sum(H, dims=2)))
reward_d, s_d, issup_list = scaled_power_reward(order, 2.0)  # r_e(t) example

# labels
node_label_data = sparse(data["L"])      # ensure SparseMatrixCSC
label_names = data["label_names"]
q = length(label_names)

@show size(node_label_data), q

# ----------------------------
# Choose fairness requirements l
# l[j] = minimum number of nodes from label j
# ----------------------------
# Example A: uniform requirement (e.g., at least 3 from each label)
l = fill(30, q)

# (Optional) sanity check feasibility using label counts
label_counts = [nnz(node_label_data[:, j]) for j in 1:q]
@show label_counts
@assert all(l .<= label_counts) "Infeasible l: some label doesn't have enough nodes."

# ----------------------------
# Run FAIR densest
# ----------------------------
best_S, best_density = fair_densest_subgraph_edge_list(
    edges,
    v_to_edges,
    l,
    reward_d,
    node_label_data,
    label_names
)

println("Best fair set size = ", length(best_S))
println("Best fair density  = ", best_density)

# ----------------------------
# Verify fairness constraints in the returned set
# ----------------------------
Svec = collect(best_S)
counts_in_S = [sum(node_label_data[Svec, j]) for j in 1:q]  # sums 0/1 entries
println("Counts per label in returned set:")
for j in 1:q
    println("  ", label_names[j], ": ", counts_in_S[j], " (req ", l[j], ")")
end

# (Optional) also print label distribution using your helper (if it expects this format)
ld = label_distribution(node_label_data, Svec, label_names)
println(ld)
