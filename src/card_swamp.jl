include("header.jl")

flowtol = 1e-25

"""
    constrained_densest_subgraph_edge_list(
        edge_list,
        vertex2edges,
        c,
        peeling_method,
        r
    )
Implements Algorithm 2 using an edge-list representation:
"""
function constrained_densest_subgraph_edge_list(
    edges_list::Vector{Vector{Int64}},
    vertex2edges::Dict{Int64,Vector{Int}},
    r::Vector{Vector{Float64}},
    c::Int64,
    issup_list::Vector{Bool};
    peeling_method::Int64=1
)
    #Total nodes in the hypergraph
    n = max_node_id(edges_list)
    @assert c <= n "c cannot exceed number of nodes"

    edges_list_copy = deepcopy(edges_list)
    vertex2edges_copy = deepcopy(vertex2edges)
    reward_d = deepcopy(r)


    # S is our global set of chosen vertices
    inS = falses(n)
    Slist = Int[]
    S = Set{Int}()
    best_density = -Inf
    best_set = Set{Int}()
    iter = 0
    # We'll do an iterative approach until we have e nodes or hypergraph is empty
    while length(Slist) < c
        # Check if the hypergraph is effectively empty
        # i.e. if all edges are empty or if no node is left
        # A quick check is if there's at least one non-empty edge,
        # or we can check if any vertex in vertex2edges has edges.
        has_nonempty_edge = any(!isempty, edges_list_copy)
        if !has_nonempty_edge
            break
        end
        iter += 1

        # 1) Get densest (or approx-densest) solution in current contracted instance
        S_i_set = Set{Int}()

        if !(0 in issup_list)
            S_i_set, _ = densest(edges_list_copy, reward_d)

        elseif peeling_method == 1
            s_d = maximal_si.(reward_d)
            gp_res = peeling_with_se_edge_list(edges_list_copy, reward_d, s_d, vertex2edges_copy)
            S_i_set = gp_res["optsol"]
        elseif peeling_method == 2
            gp_res = peeling_with_se_edge_list(edges_list_copy, reward_d, reward_d, vertex2edges_copy)
            S_i_set = gp_res["optsol"]
        elseif peeling_method == 3
            s_d0 = all_zero_si(order)
            gp_res = peeling_with_se_edge_list(edges_list_copy, reward_d, s_d0, vertex2edges_copy)
            S_i_set = gp_res["optsol"]
        elseif peeling_method == 4
            gp_res = peeling_by_degree_edge_list(edges_list_copy, reward_d, vertex2edges_copy)
            S_i_set = gp_res["optsol"]
        else
            println("Invalid argument for peeling method")
            break
        end


        if isempty(S_i_set)
            println("No vertices selected in this iteration $(iter), stopping.")
            break
        end

        new_nodes = Int[]
        @inbounds for v in S_i_set
            if 1 <= v <= n && inS[v]
                error("Error: node $v already in S, but selected again in S_i_set. This should not happen. Check the reduce method to ensure it properly removes nodes from the hypergraph.")
            end
            inS[v] = true
            push!(Slist, v)
            push!(new_nodes, v)
        end

        # 3) Form padded candidate of size exactly c (arbitrary padding, deterministic scan)

        pad_nodes = Int[]
        if length(Slist) < c
            need = c - length(Slist)
            @inbounds for v in 1:n
                if !inS[v]
                    push!(pad_nodes, v)
                    need -= 1
                    need == 0 && break
                end
            end
        end
        candidate_nodes = isempty(pad_nodes) ? Slist : vcat(Slist, pad_nodes)

        # 4) Evaluate density on ORIGINAL instance (edges_list, r)
        dens = hedensity_non_uniform_edge_list(edges_list, candidate_nodes, r)
        println("Iteration $iter: |S_i| = ", length(S_i_set), " |Candidate| = ", length(candidate_nodes), " Density = ", dens)

        if dens > best_density
            best_density = dens
            best_set = Set(candidate_nodes)  # allocate only on improvement
        end

        # If we already reached c in the union, we're done (no need to contract further)
        if length(Slist) >= c
            println("Reached desired size c in Slist, stopping iterations.")
            println("Final candidate set size $(length(Slist)), with density = ", best_density)
            break
        end

        # 3) Remove these nodes from the hypergraph
        reward_d = remove_nodes!(edges_list_copy, reward_d, vertex2edges_copy, new_nodes)
    end

    return best_set, best_density
end

@inline function remove_one!(e::Vector{Int64}, u::Int64)
    @inbounds for i in eachindex(e)
        if e[i] == u
            e[i] = e[end]
            pop!(e)
            return true
        end
    end
    return false
end


"""
    remove_nodes!(edges_list, vertex2edges, nodes_to_remove)

Remove each node in `nodes_to_remove` from the hypergraph represented by
`edge_list` and `vertex2edges`. Edges that become empty are kept as empty vectors
to avoid re-indexing. This modifies `edge_list` and `vertex2edges` in-place.
"""
function remove_nodes!(
    edges_list::Vector{Vector{Int64}},
    reward_d::Vector{Vector{Float64}},
    vertex2edges::Dict{Int64,Vector{Int64}},
    nodes_to_remove::Vector{Int64}
)

    for u in nodes_to_remove
        # Get all edges that contain u

        edges_with_u = get(vertex2edges, u, nothing)
        if edges_with_u === nothing
            error("Error: node $u not found in vertex2edges. It may have already been removed.")
        end

        # Remove u from each of those edges
        for e_idx in edges_with_u
            # If this edge is already empty, skip
            e = edges_list[e_idx]
            isempty(e) && continue

            # Remove u from edge_list[e_idx]
            old_len = length(e)
            removed = remove_one!(e, u)
            removed || continue

            new_len = length(e)
            j = old_len - new_len   # should be 1 if removed=true

            # contract reward table: r'(t) = r(t+j) - r(j)
            old_rew = reward_d[e_idx]
            base = old_rew[j+1]


            new_edge_reward = Vector{Float64}(undef, new_len + 1)
            @inbounds for t in 0:new_len
                val = old_rew[t+j+1] - base
                @assert val >= 0
                new_edge_reward[t+1] = max(val, 0.0)  # optional clamp if you expect tiny negatives
            end
            reward_d[e_idx] = new_edge_reward

        end
        # Clear vertex2edges[u]
        delete!(vertex2edges[u])

    end

    return reward_d
end

#Example usage

dataname = "senate-committees"
# data = matread("../datafolder/large-datasets/$dataname.mat")
data = matread("Densest-SWAMP/datafolder/large-datasets/senate-committees.mat")

H = sparse(Float64.(data["H"]))
Ht = sparse(H')
edges = incidence2elist(H)

@show size(H), mean(H_order(Ht))
order = vec(round.(Int64, sum(H, dims=2)))
L = round(Int64, sum(H))
M = size(H, 1)
n = size(H, 2)


eweights = ones(M)
v_to_edges = build_vertex_to_edges(edges)

# reward_d, s_d, issup_list = standard_reward(order)
# reward_d, s_d, issup_list = allbutone_reward(order)
# reward_d, s_d, issup_list = atleast_half_reward(order)
# reward_d, s_d, issup_list = scaled_power_reward(order, 1.0)
reward_d, s_d, issup_list = scaled_power_reward(order, 2.0)
# reward_d, s_d, issup_list = power_reward(order, 0.5)
# reward_d, s_d, issup_list = atleast_two_reward(order)

println(0 in issup_list)

c = 30

# Choose method:
# - is_supermodular=true => uses your flow-based `densest(...)`
# - otherwise uses peeling_method
card_S, optval = constrained_densest_subgraph_edge_list(
    edges,
    v_to_edges,
    reward_d,
    c,
    issup_list,
    peeling_method=1
)




println("Selected |S| = ", length(card_S))
println("S = ", card_S)
println("density = ", optval)
#################################################################





