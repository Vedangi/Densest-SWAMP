# g_value_edge: Computes the marginal gain (g-value) for a vertex based on 
# incident hyperedges, using provided reward (r) and penalty (s) vectors.
# Computes g(v; S) for vertex v given the current set S.
# Here, S is a Set{Int} for fast membership tests.
function g_value_edge(
    edge_list::Vector{Vector{Int}},
    v::Int,
    S::Vector{Int},
    r::Vector{Vector{Float64}},
    s::Vector{Vector{Float64}},
    vertex2edges::Dict{Int, Vector{Int}},
)
    total = 0.0
    # Iterate over edges incident to v.
    for e in vertex2edges[v]
        # Count the number of vertices in this edge that are still in S.
        count_in_S = 0
        for u in edge_list[e]
            if u in S
                count_in_S += 1
            end
        end
        # Only contribute if the edge touches S.
        if count_in_S > 0
            # For an edge e, if count_in_S = k then we use:
            #   r[e][k+1]  and  s[e][k]  (since after removing v, the count becomes k-1)
            total += r[e][count_in_S+1] - s[e][count_in_S]
        end
    end
    return total
end

# greedy_peeling_non_sup: Iteratively removes vertices with the smallest 
# g-value using a priority queue, updating neighbors' g-values, and tracks 
# the best objective value (via hedensity_non_uniform).
function greedy_peeling_with_se(
    # edge_list::Vector{Vector{Int64}},
    H :: SparseMatrixCSC{Tf, Ti},
    r::Vector{Vector{Float64}},
    s::Vector{Vector{Float64}},
)where {Ti <: Integer, Tf <: AbstractFloat}
   


    #S is a vector from 1:N
    total_dt = @elapsed begin
        n = size(H,2)
        m = size(H,1)
        
        # Start with S as a vector of all vertices.
        # S = copy(vertices)
        S = Vector(1:n)
        best_ans = hedensity_non_uniform(H, S, r)
        best_ans = round(best_ans , digits=5)
        best_S = copy(S)
        peeling_ord = Int[]  # will record the order of removals

        edge_list = incidence2elist(H)
        # Build mapping: vertex -> list of incident edges.
        vertex2edges = build_vertex_to_edges(edge_list)
        
        # Priority queue: key = vertex, priority = current g_value (lower means more likely to peel).
        pq = PriorityQueue{Int, Float64}()
        # We'll also keep a dictionary of current g values for convenience.
        curr_g = Dict{Int, Float64}()
        for v in S
            gv = g_value_edge(edge_list, v, S, r, s, vertex2edges)
            curr_g[v] = gv
            enqueue!(pq, v => gv)
        end

        # Greedy peeling: while S is not empty, remove the vertex with smallest g value.
        while !isempty(S)
            # Remove vertex with smallest g(v; S).
            # println("Current pq is: ", pq)
            v = dequeue!(pq)
            push!(peeling_ord, v)
            S = setdiff(S, [v])

            # Update neighbors that share an edge with v.
            for e in vertex2edges[v]
                for u in edge_list[e]
                    if u in S
                        # Recompute g(u; S) with the updated set S.
                        new_g = g_value_edge(edge_list, u, S, r, s, vertex2edges)
                        curr_g[u] = new_g
                        # Update the priority queue.
                        pq[u] = new_g
                    end
                end
            end

            if !isempty(S)
                curr_obj = hedensity_non_uniform(H, S, r)
                if curr_obj > best_ans
                    best_ans = curr_obj
                    best_S = copy(S)
                end
            end
        end
    end

    return Dict(
        "optval" => best_ans,
        "optsol" => best_S,
        "peeling_ord" => peeling_ord,
        "total_dt" => total_dt
    )
end


"""
    greedy_peeling_with_se_edge_list(
        edge_list,
        r,
        s,
        vertex2edges
    )

Perform greedy peeling on a hypergraph represented by an edge list.
- `edge_list` is a Vector of edges (each a Vector of Int vertex IDs).
- `r` and `s` are vectors of reward vectors for each edge.
- `vertex2edges` is a mapping from vertex (Int) to a Vector of incident edge indices.
The function returns a dictionary containing:
    - "optval": The best objective value (reward-based density) found.
    - "optsol": The corresponding set of vertices.
    - "peeling_ord": The order in which vertices were removed.
    - "total_dt": Total computation time.
"""
function greedy_peeling_with_se_edge_list(
    edge_list::Vector{Vector{Int}},
    r::Vector{Vector{Float64}},
    s::Vector{Vector{Float64}},
    vertex2edges::Dict{Int, Vector{Int}}
)
    total_dt = @elapsed begin
        # Determine all vertices from vertex2edges (assumed to cover all vertices)
        vertices = sort(collect(keys(vertex2edges)))
        # Start with S as the full vertex set.
        S = copy(vertices)
        best_ans = hedensity_non_uniform_edge_list(edge_list, S, r)
        best_ans = round(best_ans, digits=5)
        best_S = copy(S)
        peeling_ord = Int[]

        # Priority queue: key = vertex, priority = current g_value.
        # curr_g keeps track of the current g value for each vertex.
        pq = PriorityQueue{Int, Float64}()
        curr_g = Dict{Int, Float64}()
        for v in S
            gv = g_value_edge(edge_list, v, S, r, s, vertex2edges)
            curr_g[v] = gv
            enqueue!(pq, v => gv)
        end

        # Greedy peeling: remove the vertex with smallest g value iteratively.
        while !isempty(S)
            v = dequeue!(pq)
            push!(peeling_ord, v)
            S = setdiff(S, [v])
            # Update neighbors: for every edge incident to v, update g(u; S)
            for e in vertex2edges[v]
                for u in edge_list[e]
                    if u in S
                        new_g = g_value_edge(edge_list, u, S, r, s, vertex2edges)
                        curr_g[u] = new_g
                        pq[u] = new_g
                    end
                end
            end

            if !isempty(S)
                curr_obj = hedensity_non_uniform_edge_list(edge_list, S, r)
                if curr_obj > best_ans
                    best_ans = curr_obj
                    best_S = copy(S)
                end
            end
        end
    end

    return Dict(
        "optval" => best_ans,
        "optsol" => best_S,
        "peeling_ord" => peeling_ord,
        "total_dt" => total_dt
    )
end