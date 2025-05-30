using SparseArrays, Printf
# Count the total number of actual nodes (assuming edges are given 0-indexed)
function count_nodes(edges::Vector{Vector{Int}})
    return maximum([maximum(e) for e in edges]) + 1
end

"""
    construct_network(n, edges, rewards_edges)

Constructs the network as a sparse capacity matrix.
Mapping of nodes:
  - Source: node 1
  - Sink: last node (total nodes = 2 + n + (sum over edges of length(e)))
  - Actual nodes: nodes 2 to 1+n  (original index x becomes 2+x)
  - Edge-nodes: following the actual nodes

For each gadget (edge and its reward vector), we compute "alpha" values:
  - α₁ = r₂ − r₁
  - for i = 2:L,  αᵢ = (rᵢ₊₁ − rᵢ) − (rᵢ − rᵢ₋₁)
Then, for each piece i = 1:L:
  - For every vertex x in the edge, add an edge from the current edge-node to actual node (2+x) with capacity αᵢ.
  - Also, add an edge from source (node 1) to the edge-node with capacity = (L − (i−1)) · αᵢ.
Finally, for each actual node we add an edge from that node to sink with initial capacity 0 (to be updated later).
"""
function construct_network(n::Int, edges::Vector{Vector{Int}}, rewards_edges::Vector{Vector{Float64}})
    necnt = sum(length(e) for e in edges)
    tot_nodes = 2 + n + necnt  # total nodes: source, sink, actual nodes, edge-nodes
    s = 1                    # source node
    t = tot_nodes            # sink node
    actual_start = 2         # actual nodes: indices 2 to 1+n
    edge_start = actual_start + n  # edge-nodes start at n+2

    # Build lists for the sparse matrix (directed graph)
    row = Int[]
    col = Int[]
    data = Float64[]

    # Add edges from each actual node to sink (capacity will be updated later)
    for i in 0:(n-1)
        actual_node = actual_start + i   # mapping: original node i → (2 + i)
        push!(row, actual_node)
        push!(col, t)
        push!(data, 0.0)
    end

    current_edge_node = edge_start
    for (e, r) in zip(edges, rewards_edges)
        L = length(e)
        if length(r) != L + 1
            error("Rewards list for each edge must be of length len(edge)+1.")
        end
        # Compute alpha values for this gadget
        alpha = Vector{Float64}(undef, L)
        alpha[1] = r[2] - r[1]
        for i in 2:L
            alpha[i] = (r[i+1] - r[i]) - (r[i] - r[i-1])
        end

        # For each piece in the gadget, add the corresponding edges
        for i in 1:L
            # For each vertex in the edge, add an edge from the current edge-node to the actual node.
            for x in e
                # x is 0-indexed; actual node index is actual_start + x.
                push!(row, current_edge_node)
                push!(col, actual_start + x)
                push!(data, alpha[i])
            end
            # Add edge from source to the edge-node.
            cap = (L - (i - 1)) * alpha[i]
            push!(row, s)
            push!(col, current_edge_node)
            push!(data, cap)

            current_edge_node += 1
        end
    end

    return sparse(row, col, data, tot_nodes, tot_nodes)
end

# Set (or update) the capacities on edges from actual nodes to the sink.
function set_lambda!(C::SparseMatrixCSC{Float64,Int}, n::Int, value::Float64)
    tot_nodes = size(C, 1)
    t = tot_nodes
    actual_start = 2
    for i in 0:(n-1)
        node = actual_start + i
        C[node, t] = value
    end
end

# Compute the minimum cut using the HLPP maxflow solver.
# After setting λ (value) on edges from actual nodes to sink, we compute the flow,
# then extract S (the source side) and filter out the actual nodes.
# We subtract 2 from the node index to recover the original 0-indexed label.
function cut(C::SparseMatrixCSC{Float64,Int}, n::Int, value::Float64)
    set_lambda!(C, n, value)
    flow = hlpp.maxflow(C, 1e-6)
    S_nodes = hlpp.source_nodes(flow)  # Returns nodes on the source side.
    S = Set{Int}()
    # Only keep nodes corresponding to actual nodes (indices 2 to 1+n)
    for x in S_nodes
        if x >= 2 && x <= 1 + n
            push!(S, x - 2)
        end
    end
    return S
end

# Compute the density for a given subgraph S.
# For each edge and its rewards vector, count how many vertices in the edge lie in S,
# then add the corresponding reward.
function density(edges::Vector{Vector{Int}}, S::Set{Int}, rewards_edges::Vector{Vector{Float64}})
    total_reward = 0.0
    for (e, r) in zip(edges, rewards_edges)
        count = sum(x -> x in S, e)
        total_reward += r[count+1]  # adjust for 1-indexing
    end
    return isempty(S) ? 0.0 : total_reward / length(S)
end

# Iteratively compute cuts with increasing λ (density) until no improvement is found.
# Returns the optimal subgraph (as a set of original node indices) and the time taken.
function densest(edges::Vector{Vector{Int}}, rewards_edges::Vector{Vector{Float64}})
    start_time = time()
    n = count_nodes(edges)
    C = construct_network(n, edges, rewards_edges)
    dens = 0.0
    optS = Set{Int}()
    while true
        S = cut(C, n, dens)
        println(length(S))
        cand = density(edges, S, rewards_edges)
        if cand > dens
            dens = cand
            optS = S
            println(dens)
        else
            break
        end
    end
    time_taken = time() - start_time
    println("Time taken: ", time_taken)
    return optS, time_taken
end

# ===== Example Usage =====

# edges = [[0, 1], [1, 2, 3]]
# rewards_edges = [[0.0, 1.0, 2.0], [0.0, 0.5, 1.0, 1.5]]

# optS, t_taken = densest(edges, rewards_edges)
# println("Optimal S: ", optS)

#Example usage over a real graph network

# data = matread("datafolder/large-datasets/contact-high-school.mat")
# H = sparse(Float64.(data["H"]))
# Ht = sparse(H')
# edge_list = incidence2elist(H)
# n = size(H,2)
# m = size(H,1)
# println("n = ", n)
# println("m = ", m)
# order = vec(round.(Int64,sum(H,dims = 2)))
# v_to_edges = build_vertex_to_edges(edge_list)

# reward_d,s_d,issup_list = atleast_two_reward(order)

# optS, t_taken = densest(edge_list, reward_d)
# println("Optimal S: ", optS)