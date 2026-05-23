###############################################################################
# Monotone submodular maximization on a hypergraph with per-edge count rewards
# Inputs:
#   edges_list::Vector{Vector{Int}}         # hyperedges as node-id lists
#   vertex2edges::Dict{Int,Vector{Int}}     # node -> incident hyperedge indices (1..m)
#   reward_d::Vector{Vector{Float64}}       # reward_d[e][t+1] = r_e(t), t=0..|e|
# Optional:
#   edge_weights::Vector{Float64}           # w_e (default all 1.0)
#
# Algorithms:
#   1) Greedy (Nemhauser): max_{|S|=k} f(S)
#   2) ILP exact via JuMP + Gurobi using threshold variables y[e,t]
#
# NOTES:
# - Greedy gives (1-1/e) approx for monotone submodular f under |S|=k.
# - ILP is exact (NP-hard worst-case; often fine for moderate sizes).
###############################################################################
# ---------------------------- Utility: f(S) ----------------------------
"""
    f_value(
    edges_list::Vector{Vector{Int}},
    S,
    reward_d::Vector{Vector{Float64}};
    edge_weights::Union{Nothing,Vector{Float64}}=nothing
)::Float64

Compute f(S)= sum_e w_e * r_e(|e ∩ S|).
S can be any iterable of node ids.
"""
function f_value(
    edges_list::Vector{Vector{Int}},
    S,
    reward_d::Vector{Vector{Float64}};
    edge_weights::Union{Nothing,Vector{Float64}}=nothing
)::Float64
    Sset = S isa Set ? S : Set(S)
    m = length(edges_list)
    w = edge_weights === nothing ? ones(Float64, m) : edge_weights
    @assert length(w) == m "edge_weights length must match #edges"

    total = 0.0
    for e in 1:m
        he = edges_list[e]
        t = 0
        @inbounds for v in he
            t += (v in Sset) ? 1 : 0
        end
        # reward_d[e] indexed by t+1
        @assert 0 ≤ t ≤ length(he)
        total += w[e] * reward_d[e][t+1]
    end
    return total
end


# ---------------------------- Greedy algorithm ----------------------------

"""
    greedy_submodular_maximization(edges_list, vertex2edges, reward_d, k;
                                  edge_weights=nothing, rng=Random.default_rng())

Greedy (Nemhauser) for max_{|S|=k} f(S) where f is monotone submodular.

Returns:
  (S::Set{Int}, fS::Float64)
"""
function greedy_submodular_maximization(
    edges_list::Vector{Vector{Int}},
    vertex2edges::Dict{Int,Vector{Int}},
    reward_d::Vector{Vector{Float64}},
    k::Int;
    edge_weights::Union{Nothing,Vector{Float64}}=nothing,
    rng::AbstractRNG=Random.default_rng()
)
    m = length(edges_list)
    w = edge_weights === nothing ? ones(Float64, m) : edge_weights
    @assert length(w) == m "edge_weights length must match #edges"

    # Universe of nodes: keys of vertex2edges is usually enough
    V = collect(keys(vertex2edges))
    n = length(V)
    @assert k ≤ n "k cannot exceed number of nodes in V"

    # Maintain current counts c[e] = |e ∩ S|
    c = zeros(Int, m)

    S = Set{Int}()
    fS = 0.0

    # Precompute marginal increments per edge per count: inc[e][t] = r(t)-r(t-1) for t=1..|e|
    # Stored as Vector{Vector{Float64}} with length |e| (index t)
    inc = Vector{Vector{Float64}}(undef, m)
    for e in 1:m
        re = reward_d[e]
        he = edges_list[e]
        @assert length(re) == length(he) + 1 "reward_d[e] must have length |e|+1"
        ie = zeros(Float64, length(he))
        @inbounds for t in 1:length(he)
            ie[t] = re[t+1] - re[t]  # r(t)-r(t-1)
        end
        inc[e] = ie
    end

    # Greedy selection
    for step in 1:k
        best_v = 0
        best_gain = -Inf

        # Optionally, you can keep a candidate set of not-yet-chosen nodes
        for v in V
            (v in S) && continue

            # Δ(v|S) = sum_{e ∋ v} w_e * (r(c_e+1)-r(c_e))
            gain = 0.0
            for e in get(vertex2edges, v, Int[])
                ce = c[e]
                # If ce == |e|, adding v can't increase this edge (but v shouldn't be in a full edge unless duplicates exist)
                if ce < length(edges_list[e])
                    # increment for going from ce -> ce+1 is inc[e][ce+1]
                    gain += w[e] * inc[e][ce+1]
                end
            end

            # tie-break randomly for stability
            if gain > best_gain + 1e-12
                best_gain = gain
                best_v = v
            elseif abs(gain - best_gain) ≤ 1e-12 && gain > -Inf
                if rand(rng) < 0.5
                    best_v = v
                end
            end
        end

        if best_v == 0
            # no remaining node gives any gain (possible if rewards saturate and all incident edges are full)
            break
        end

        # Add best_v and update counts + objective
        push!(S, best_v)
        fS += max(best_gain, 0.0)  # monotone should be ≥0

        for e in get(vertex2edges, best_v, Int[])
            c[e] += 1
        end
    end

    return S, fS
end


# ---------------------------- Exact ILP (JuMP + Gurobi) ----------------------------
"""
    ilp_submodular_maximization_gurobi(edges_list, reward_d, k;
                                       edge_weights=nothing, time_limit_sec=nothing, mip_gap=nothing,
                                       verbose=true)

Exact MIP:
  max sum_e w_e * sum_{t=1..|e|} (r_e(t)-r_e(t-1)) * y[e,t]
  s.t. sum_v x[v] = k
       sum_{v in e} x[v] >= t * y[e,t]
       x[v] ∈ {0,1}, y[e,t] ∈ {0,1}

Returns:
  (S::Set{Int}, obj::Float64, status)
"""
function ilp_submodular_maximization_gurobi(
    edges_list::Vector{Vector{Int}},
    reward_d::Vector{Vector{Float64}},
    k::Int;
    edge_weights::Union{Nothing,Vector{Float64}}=nothing,
    time_limit_sec::Union{Nothing,Real}=nothing,
    mip_gap::Union{Nothing,Real}=nothing,
    verbose::Bool=true
)


    m = length(edges_list)
    w = edge_weights === nothing ? ones(Float64, m) : edge_weights
    @assert length(w) == m "edge_weights length must match #edges"

    # Build node universe from edges_list (safe even if vertex2edges missing some)
    V = sort!(collect(Set(vcat(edges_list...))))
    n = length(V)
    @assert k ≤ n "k cannot exceed number of nodes in hypergraph"

    # Map node id -> index 1..n for JuMP array vars
    vidx = Dict{Int,Int}(v => i for (i, v) in enumerate(V))

    # Precompute deltas δ[e,t] = r_e(t)-r_e(t-1)
    delta = Vector{Vector{Float64}}(undef, m)
    for e in 1:m
        he = edges_list[e]
        re = reward_d[e]
        @assert length(re) == length(he) + 1 "reward_d[e] must have length |e|+1"
        de = zeros(Float64, length(he))
        @inbounds for t in 1:length(he)
            de[t] = re[t+1] - re[t]
        end
        delta[e] = de
    end

    model = Model(Gurobi.Optimizer)
    if !verbose
        set_silent(model)
    end
    if time_limit_sec !== nothing
        set_optimizer_attribute(model, "TimeLimit", float(time_limit_sec))
    end
    if mip_gap !== nothing
        set_optimizer_attribute(model, "MIPGap", float(mip_gap))
    end

    @variable(model, x[1:n], Bin)

    # y[e,t] only defined for valid t=1..|e|
    # We'll store as a Vector of Vector of variables.
    y = Vector{Vector{JuMP.VariableRef}}(undef, m)
    for e in 1:m
        y[e] = Vector{JuMP.VariableRef}(undef, length(edges_list[e]))
        for t in 1:length(edges_list[e])
            y[e][t] = @variable(model, base_name = "y_$(e)_$(t)", binary = true)
        end
    end

    # Cardinality constraint
    @constraint(model, sum(x) == k)

    # Threshold constraints: sum_{v in e} x[v] >= t * y[e,t]
    for e in 1:m
        he = edges_list[e]
        idxs = [vidx[v] for v in he]  # indices in x
        for t in 1:length(he)
            @constraint(model, sum(x[i] for i in idxs) >= t * y[e][t])
        end
    end

    # Objective
    @objective(model, Max, sum(w[e] * sum(delta[e][t] * y[e][t] for t in 1:length(delta[e])) for e in 1:m))

    optimize!(model)

    status = termination_status(model)
    if status != MOI.OPTIMAL && status != MOI.TIME_LIMIT && status != MOI.SUBOPTIMAL
        @warn "Solver terminated with status $status"
    end

    xval = value.(x)
    S = Set{Int}()
    for i in 1:n
        if xval[i] >= 0.5
            push!(S, V[i])
        end
    end

    obj = objective_value(model)
    return S, obj, status
end


# ---------------------------- Convenience: run both ----------------------------

"""
    run_greedy_and_ilp(edges_list, vertex2edges, reward_d, k;
                       edge_weights=nothing, seed=1)

Runs Greedy and ILP and prints a small comparison.
"""
function run_greedy_and_ilp(
    edges_list::Vector{Vector{Int}},
    vertex2edges::Dict{Int,Vector{Int}},
    reward_d::Vector{Vector{Float64}},
    k::Int;
    edge_weights::Union{Nothing,Vector{Float64}}=nothing,
    seed::Int=1
)
    rng = MersenneTwister(seed)

    Sg, fg = greedy_submodular_maximization(edges_list, vertex2edges, reward_d, k;
        edge_weights=edge_weights, rng=rng)
    println("Greedy |S|=$(length(Sg))  f(S)=$fg")

    Si, fi, st = ilp_submodular_maximization_gurobi(edges_list, reward_d, k;
        edge_weights=edge_weights, verbose=true)
    println("ILP    |S|=$(length(Si))  f(S)=$fi   status=$st")

    # sanity: recompute f(S) by direct evaluation
    fg_check = f_value(edges_list, Sg, reward_d; edge_weights=edge_weights)
    fi_check = f_value(edges_list, Si, reward_d; edge_weights=edge_weights)
    println("Check  greedy f=$fg_check   ilp f=$fi_check")

    if fi_check > 0
        println("Greedy/ILP ratio = ", fg_check / fi_check)
    end
    return (Sg=Sg, fg=fg_check, Si=Si, fi=fi_check, status=st)
end
