#This file contains a tweaked implementation of a degree-based peeling version of
# the Greedy++ algorithm, inspired by a naive C++ implementation available at:
# https://www.dropbox.com/s/jzouo9fjoytyqg3/code-greedy%2B%2B.zip?dl=0&file_subpath=%2Fcode-greedy%2B%2B

# Key Features:
# - Computes initial vertex degrees from the input sparse matrix.
# - Uses a priority queue to efficiently peel off vertices with the smallest degree.
# - Updates the density objective iteratively as vertices are removed.
# - Returns the optimal density value, the corresponding vertex subset (solution),
#   the order of vertex removal, and the total execution time.
#
# Inputs:
#   - H  : SparseMatrixCSC representing the graph (edge-vertex relationships).
#   - Ht : Transpose of H, for efficient neighbor lookup.
#   - r  : A vector of reward weights used in the non-uniform density function.
#
# Outputs (returned as a dictionary):
#   - "optval"    : Best objective value found.
#   - "optsol"    : Subset of vertices corresponding to the best objective.
#   - "peeling_ord": Order in which vertices were peeled off.
#   - "total_dt"  : Total execution time of the algorithm.
#
using DataStructures
function peeling_by_degree(
    H::SparseMatrixCSC{Tf,Ti},
    Ht::SparseMatrixCSC{Tf,Ti},
    r::Vector{Vector{Float64}},
) where {Ti<:Integer,Tf<:AbstractFloat}
    total_dt = @elapsed begin

        m, n = size(H) # number of edges, vertices

        # compute the degrees of vertices
        deg = H_deg(H)

        total_wts = hedensity_non_uniform(H, Vector(1:n), r) * n
        best_ans = total_wts / n
        best_size = n
        best_S = Vector(1:n)
        S = Vector(1:n)
        peeling_ord = zeros(Ti, n)

        # indicate  
        # whether one node has been peeled off or not
        # whether one edge has already not been fully contained
        exists_v = ones(Bool, n)
        contained_e = ones(Bool, m)

        # keep vertices sorted by degree
        pq = PriorityQueue{Ti,Tf}()
        curr_deg = zeros(n)


        curr_wts = total_wts
        # println("Initial wts is: ", curr_wts)

        for u in 1:n

            curr_deg[u] = deg[u]

            exists_v[u] = true
            enqueue!(pq, u => curr_deg[u])
        end
        for e in 1:m
            contained_e[e] = true
        end

        for i in 1:n
            # delete the vertex with the smallest degree
            # println("Current pq is: ", pq)
            u = dequeue!(pq)

            exists_v[u] = false
            peeling_ord[i] = u
            # curr_wts -= curr_deg[u]
            S = setdiff(S, [u])
            curr_wts = hedensity_non_uniform(H, S, r) * length(S)
            # println("curr_wts using hedensity_non_uniform function is ", curr_wts)

            for nzi in H.colptr[u]:(H.colptr[u+1]-1)
                e = H.rowval[nzi]
                # if this edge has already been not fully contained
                # then we skip
                if !contained_e[e]
                    continue
                end
                contained_e[e] = false
                for nzj in Ht.colptr[e]:(Ht.colptr[e+1]-1)
                    v = Ht.rowval[nzj]
                    # we only process those vertices that
                    # haven't been peeled off
                    if !exists_v[v]
                        continue
                    end

                    curr_deg[v] -= Ht.nzval[nzj]
                    pq[v] = curr_deg[v]
                end
            end
            if i < n
                curr_density = curr_wts / (n - i)
                if curr_density > best_ans
                    best_ans = curr_density
                    best_size = n - i
                    # println("New best size in this if statement is ",best_size)
                end
            end
        end
        best_S = peeling_ord[n-best_size+1:n]


        # println("best_S: in this step is ", best_S, " with best_ans: ", best_ans, "lngth(best)S)", length(best_S))
    end
    return Dict(
        "optval" => best_ans,
        "optsol" => best_S,
        "peeling_ord" => peeling_ord,
        "total_dt" => total_dt,
    )
end



"""
    peeling_by_degree_edge_list(
         edge_list,
         r,
         vertex2edges
    )

Perform peeling based on vertex degree on a hypergraph represented
by an edge list. The vertex set is determined by extracting all unique vertices
from `edge_list`.

Arguments:
- `edge_list::Vector{Vector{Int}}`: List of edges; each edge is a vector of vertex IDs.
- `r::Vector{Vector{Float64}}`: Reward vectors for each edge.
- `s::Vector{Vector{Float64}}`: Auxiliary reward vectors for each edge.
- `vertex2edges::Dict{Int, Vector{Int}}`: Mapping from vertex to a vector of incident edge indices.

Returns a dictionary with:
  - `"optval"`: Best objective value (density) found.
  - `"optsol"`: The corresponding set of vertices.
  - `"peeling_ord"`: The order in which vertices were removed.
  - `"total_dt"`: Total elapsed time.
"""
function peeling_by_degree_edge_list(
    edge_list::Vector{Vector{Int}},
    r::Vector{Vector{Float64}},
    vertex2edges::Dict{Int,Vector{Int}}
)
    total_dt = @elapsed begin
        # 1. Determine all vertices from the edge_list.
        vertices = sort(unique(vcat(edge_list...)))
        n = length(vertices)

        # 2. Compute the degree of each vertex (here, the degree is the number of incident edges).
        deg = Dict{Int,Float64}()
        for v in vertices
            deg[v] = length(get(vertex2edges, v, []))
        end

        # 3. Initialize S as the set of all vertices from the edge_list.
        S = copy(vertices)
        total_wts = hedensity_non_uniform_edge_list(edge_list, S, r) * n
        best_ans = total_wts / n
        best_S = copy(S)
        peeling_ord = Int[]

        # 4. Keep track of vertex existence using a dictionary.
        exists_v = Dict(v => true for v in vertices)
        # For each edge, whether it is still "fully contained" in S.
        contained_e = trues(length(edge_list))

        # 5. Initialize a priority queue with the vertices and their current degrees.
        pq = PriorityQueue{Int,Float64}()
        curr_deg = Dict{Int,Float64}()
        for v in vertices
            curr_deg[v] = deg[v]
            enqueue!(pq, v => curr_deg[v])
        end

        curr_wts = total_wts

        # 6. peeling loop.
        for i in 1:n
            # Remove the vertex with smallest degree.
            v = dequeue!(pq)
            exists_v[v] = false
            push!(peeling_ord, v)
            # Remove v from S.
            S = setdiff(S, [v])
            # Recompute total reward for the remaining set S.
            curr_wts = hedensity_non_uniform_edge_list(edge_list, S, r) * length(S)

            # Update the degrees of neighbors in all edges incident to v.
            for e in get(vertex2edges, v, Int[])
                # If this edge has already been processed, skip it.
                if !contained_e[e]
                    continue
                end
                contained_e[e] = false
                for u in edge_list[e]
                    # Only update vertices that still exist.
                    if !haskey(exists_v, u) || !exists_v[u]
                        continue
                    end
                    # In the matrix version, the update was weighted by Ht.nzval.
                    # For an unweighted edge list, subtract 1.
                    curr_deg[u] -= 1.0
                    # Update the priority queue.
                    pq[u] = curr_deg[u]
                end
            end

            # Update the best objective if the current density improves.
            if !isempty(S) && i < n
                curr_density = curr_wts / length(S)
                if curr_density > best_ans
                    best_ans = curr_density
                    best_S = copy(S)
                end
            end
        end
    end

    return Dict(
        "optval" => best_ans,
        "optsol" => best_S,
        "peeling_ord" => peeling_ord,
        "total_dt" => total_dt,
    )
end
