using HTTP
using ZipFile
using CSV
using DataFrames
using JSON
using Dates
using CodecZlib
using Random
using Statistics
include("../header.jl")
include("../card_swamp.jl")
include("submod_maximization.jl")


#define function to read the edges list from a file
function load_nyc_edges_list(path::String)
    edges_list = Vector{Vector{Int}}()
    line_names = String[]

    open(path, "r") do io
        for raw_line in eachline(io)
            line = strip(raw_line)
            isempty(line) && continue

            parts = split(line)

            # First entry is the subway line / route name
            line_name = parts[1]

            # Remaining entries are the node IDs in that hyperedge
            nodes = parse.(Int, parts[2:end])

            push!(line_names, line_name)
            push!(edges_list, nodes)
        end
    end

    return edges_list, line_names
end


function low_power_reward(order::Vector{Int64}, p::T) where {T<:Real}

    reward_d = Vector{Vector{Float64}}()
    s_d = Vector{Vector{Float64}}()
    issup_list = Vector{Bool}()

    for e = 1:length(order)
        k = order[e]
        g = zeros(k + 1)
        g[2] = 1.0
        for i = 2:k
            g[i+1] = (i)^p
        end
        s = maximal_si(g)
        push!(reward_d, g)
        push!(s_d, s)
        issup = isconvex(g, 10)
        push!(issup_list, issup)
    end
    return reward_d, s_d, issup_list
end





# ---- Example usage ----
edges_list, line_names = load_nyc_edges_list("nyc_lines_hypergraph.txt")
#find number of unique nodes in edges_list
node_names = unique(vcat(edges_list...))

order_n = [length(e) for e in edges_list]
reward_d, s_d, issup_list = scaled_power_reward(order_n, 1.0)

edges_weights = ones(length(edges_list))
vertex2edges = build_vertex_to_edges(edges_list)


# Run the ILP for exact maximization (for different values of p)
solution_summary = Dict{Tuple{Float64,Float64},Vector{Int64}}()
for p in [1.0, 0.95, 0.9, 0.7, 0.5, 0.3, 1e-2, 1e-5, 1e-9, 1e-10, 0.0]
    reward_d, s_d, issup_list = low_power_reward(order_n, p)
    ilp_S, ilp_val = ilp_submodular_maximization_gurobi(
        edges_list,
        reward_d,
        20; 
        edge_weights=edges_weights,
        mip_gap=nothing, # 1% gap
        verbose=false
    )
    # card_S, optval = constrained_densest_subgraph_edge_list(
    #     edges_list,
    #     vertex2edges,
    #     reward_d,
    #     6,
    #     issup_list,
    #     peeling_method=1
    # )
    solution_summary[(20, p)] = collect(ilp_S)
end
print(solution_summary)
