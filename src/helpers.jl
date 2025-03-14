
"""
    hedensity_non_uniform(H, S, r)

Given:
  - H: (m x n) incidence matrix of a hypergraph (SparseMatrixCSC).
  - S: vector of vertices in the set S (1-based indices).
  - r: Vector of vector of reward vectors [r0, r1, r2, r3, ...]. 

Compute:
  - The "reward-based" edge density, i.e. 
        sum_of_edge_rewards / |S|.
    For each edge that touches S, we add r[k+1] where k is the 
    number of vertices of that edge in S.
"""
function hedensity_non_uniform(
    H::SparseMatrixCSC{Tf, Ti},
    S::Vector{Ti},
    r::Vector{Vector{T}}
) :: Float64 where {Tf <: AbstractFloat, Ti <: Integer, T<:Real}

    # @assert length(S) > 0 "S cannot be empty"
    if length(S) == 0
        return 0.0
    end
    # m = number of edges, n = number of vertices
    m = size(H, 1)

    total_reward = 0.0
    HS = H[:, S]
    rp_HS = HS.rowval 
    etouchS_list = unique(rp_HS)

    @inbounds for i in etouchS_list
        # Count how many vertices of edge i lie in S
        count_in_S = 0
        for v in S
            # If H[i,v] != 0, that means vertex v is in edge i
            if H[i, v] != 0
                count_in_S += 1
            end
        end
        # If the edge touches S (count_in_S > 0), add the corresponding reward
        if count_in_S > 0
            # r is [ e0, r1, r2, r3, ... ]
            # For k nodes, we use r[k+1].
            total_reward += r[i][count_in_S + 1]
        end
    end

    return total_reward / length(S)
end


"""
    hedensity_non_uniform_edge_list(edge_list, S, r)

Given:
  - `edge_list`: A vector of edges, where each edge is a collection
                 of node IDs (e.g. `Vector{Int}` or `Set{Int}`).
  - `S`: Vector of vertices in the set S (1-based indices).
  - `r`: A vector of vectors of rewards. `r[e]` is `[r0, r1, r2, ...]`
         for the e-th edge in `edge_list`.

Compute:
  - The "reward-based" edge density = (sum_of_edge_rewards) / |S|.
    For each edge that intersects S, we add `r[e][i+1]`, where `i`
    is the number of vertices of that edge in S.
"""
function hedensity_non_uniform_edge_list(
    edge_list::Vector{<:AbstractVector{Int}},  # or Vector{Set{Int}}
    S::Vector{Int},
    r::Vector{Vector{Float64}}
) :: Float64

    # If S is empty, return 0.0 to avoid division by zero
    if isempty(S)
        return 0.0
    end

    # Convert S to a Set for O(1) membership checks
    S_set = Set(S)

    total_reward = 0.0

    # Iterate over each edge i
    for (i, edge) in enumerate(edge_list)
        # Count how many nodes of this edge lie in S
        count_in_S = 0
        for v in edge
            if v in S_set
                count_in_S += 1
            end
        end

        # If at least one node of edge i is in S, add its reward
        if count_in_S > 0
            # For k = count_in_S, we add r[i][k+1]
            # Make sure r[i] has enough entries for k+1
            total_reward += r[i][count_in_S + 1]
        end
    end

    return total_reward / length(S)
end


"""
    hedensity(H, S, r)

Given:
  - H: (m x n) incidence matrix of a hypergraph (SparseMatrixCSC).
  - S: vector of vertices in the set S (1-based indices).
  - r: reward vector [r0, r1, r2, r3, ...]. (uniform across all edges)

Compute:
  - The "reward-based" edge density, i.e. 
        sum_of_edge_rewards / |S|.
    For each edge that touches S, we add r[k+1] where k is the 
    number of vertices of that edge in S.
"""
function hedensity(
    H::SparseMatrixCSC{Tf, Ti},
    S::Vector{Ti},
    r::AbstractVector{T},
) :: Float64 where {Tf <: AbstractFloat, Ti <: Integer, T<:Real}

    # @assert length(S) > 0 "S cannot be empty"
    if length(S) == 0
        return 0.0
    end
    # m = number of edges, n = number of vertices
    m = size(H, 1)

    total_reward = 0.0
    HS = H[:, S]
    rp_HS = HS.rowval 
    etouchS_list = unique(rp_HS)

    @inbounds for i in etouchS_list
        # Count how many vertices of edge i lie in S
        count_in_S = 0
        for v in S
            # If H[i,v] != 0, that means vertex v is in edge i
            if H[i, v] != 0
                count_in_S += 1
            end
        end
        # If the edge touches S (count_in_S > 0), add the corresponding reward
        if count_in_S > 0
            # r is [ e0, r1, r2, r3, ... ]
            # For k nodes, we use r[k+1].
            total_reward += r[count_in_S + 1]
        end
    end

    return total_reward / length(S)
end

"""
    hedensity(A, S, r)

Given:
  - A: (n x n) adjacency matrix of a graph (SparseMatrixCSC or dense).
  - S: vector of vertices in S (1-based indices).
  - r: reward vector [e0, r1, r2, r3, ...].

Compute:
  - The "reward-based" edge density = sum_of_edge_rewards / |S|.
    For each edge {u,v} that touches S, we add r[k+1], where k 
    is how many of the vertices (u or v) are in S.
"""
function hedensity(
    A::AbstractMatrix{Tval},
    S::Vector{Ti},
    r::AbstractVector{T},
) :: Float64 where {Tval<:Real, Ti<:Integer, T<:Real}

    # @assert length(S) > 0 "S cannot be empty"
    if length(S) == 0
        return 0.0
    end
    n = size(A,1)
    @assert n == size(A,2) "Adjacency matrix must be square"

    total_reward = 0.0

    # We'll iterate over all pairs (u,v) with u < v to avoid double counting.
    @inbounds for u in 1:n
        for v in (u+1):n
            # Is there an edge?
            if A[u,v] != 0
                # Count how many are in S
                count_in_S = 0
                count_in_S += (u in S ? 1 : 0)
                count_in_S += (v in S ? 1 : 0)
                if count_in_S > 0
                    total_reward += r[count_in_S + 1]
                end
            end
        end
    end

    return total_reward / length(S)
end

"""
r_to_R(reward_d::Vector{Vector{Float64}}, n::Int64)
Convert the reward values to integers by multiplying by 10^n.
"""
function r_to_R(reward_d::Vector{Vector{Float64}}, n::Int64)
    factor = 10^n
    R = Vector{Vector{Float64}}()
    for r in reward_d
        R_e = Vector{Float64}()
        for i in 1:length(r)
            push!(R_e, r[i]*factor)
        end
        push!(R, R_e)
    end
    return R
end

"""
    all_zero_si(order)
Create a vector of all-zero maximal si vectors for each edge.
"""
function all_zero_si(order)
    s_d = Vector{Vector{Float64}}()
    for e = 1:length(order)
        k = order[e]
        s = zeros(k+1)
        push!(s_d,s)
    end
    return s_d
end

"""
    standard_reward(order)
Create a reward function for the standard all-or-nothing
    densest subhypergraph problem.
"""
function standard_reward(order::Vector{Int64})
    reward_d = Vector{Vector{Float64}}()
    s_d = Vector{Vector{Float64}}()
    issup_list = Vector{Bool}()

    for e = 1:length(order)
        k = order[e]
        g = zeros(k+1)
        g[end] = 1
        s = maximal_si(g)
        push!(reward_d,g)
        push!(s_d,s)
        issup = isconvex(g,3)
        push!(issup_list, issup)
    end
    return reward_d, s_d, issup_list
end

"""
Create a reward function where
    you get 1 point if you have k or k-1
    nodes from a hyperedge of size k. (Here k>2)
"""
function allbutone_reward(order::Vector{Int64})
    reward_d = Vector{Vector{Float64}}()
    s_d = Vector{Vector{Float64}}()
    issup_list = Vector{Bool}()
    for e = 1:length(order)
        k = order[e]
        g = zeros(k+1)
        g[k+1] = 1
        if k > 2
            g[k] = 1
        end
        s = maximal_si(g)
        
        push!(reward_d,g)
        push!(s_d,s)
        issup = isconvex(g,8)
        push!(issup_list, issup)
    end
    
    return reward_d, s_d, issup_list
end

"""
Create a reward function where
    you get 1 point if you have atleast 2
    nodes from a hyperedge of size k (k > 2).
"""
function atleast_two_reward(order::Vector{Int64})
    reward_d = Vector{Vector{Float64}}()
    s_d = Vector{Vector{Float64}}()
    issup_list = Vector{Bool}()
    for e = 1:length(order)
        k = order[e]
        g = zeros(k+1)
        g[k+1] = 1
        if k > 2
            for i in 2:k
                g[i+1] = 1
            end
        end
        s = maximal_si(g)
        
        push!(reward_d,g)
        push!(s_d,s)
        issup = isconvex(g,3)
        push!(issup_list, issup)
    end
    
    return reward_d, s_d, issup_list
end

"""
Create a reward function where
    r(i) = i^p. for i > 1,
    r(1) = 0.
"""
function power_reward(order::Vector{Int64},p::T) where {T<:Real}

    reward_d = Vector{Vector{Float64}}()
    s_d = Vector{Vector{Float64}}()
    issup_list = Vector{Bool}()

    for e = 1:length(order)
        k = order[e]
        g = zeros(k+1)
        for i = 2:k
            g[i+1] = ((i)^p)
        end
        s = maximal_si(g)
        push!(reward_d,g)
        push!(s_d,s)
        issup = isconvex(g,20)
        push!(issup_list, issup)
    end
    return reward_d, s_d, issup_list
end

"""
Create a reward function where
    r(i) = i^p/k.
"""
function scaled_power_reward(order::Vector{Int64}, p::T) where {T<:Real}

    reward_d = Vector{Vector{Float64}}()
    s_d = Vector{Vector{Float64}}()
    issup_list = Vector{Bool}()

    for e = 1:length(order)
        k = order[e]
        g = zeros(k+1)
        for i = 0:k
            g[i+1] = ((i)^p)/k
        end
        s = maximal_si(g)
        push!(reward_d,g)
        push!(s_d,s)
        issup = isconvex(g,10)
        push!(issup_list, issup)
    end
    return reward_d, s_d, issup_list
end

"""
you get 1 point if you have atleast k/2
    nodes from a hyperedge of size k (k > 2).
"""

function atleast_half_reward(order::Vector{Int64})
    reward_d = Vector{Vector{Float64}}()
    s_d = Vector{Vector{Float64}}()
    issup_list = Vector{Bool}()

    for e = 1:length(order)
        k = order[e]
        g = zeros(k+1)
        if k <= 2
            g[k+1] = 1
        else
            half_up = round(k/2, RoundUp)
            for i = 0:k
                g[i+1] = (i >= half_up) ? 1 : 0
            end
        end
        s = maximal_si(g)
        push!(reward_d,g)
        push!(s_d,s)
        issup = isconvex(g,12)
        push!(issup_list, issup)
    end
    return reward_d, s_d, issup_list
end

# Build a mapping from each vertex to a list of edges (indices) in which it appears.
function build_vertex_to_edges(edge_list::Vector{Vector{Int}})
    vertex2edges = Dict{Int, Vector{Int}}()
    for (e_idx, edge) in enumerate(edge_list)
        for v in edge
            push!(get!(vertex2edges, v, Int[]), e_idx)
        end
    end
    return vertex2edges
end

"""
check if the reward vector is monotonic
"""
function ismonotonic( r::AbstractVector{T},) where {T<:Real}
    n = length(r)
    for i in 1:n-1
        if r[i] > r[i+1]
            return false
        end
    end
    return true
end

"""
check if the reward vector is convex
"""
function isconvex( r::AbstractVector{T},round_digit::Int64) where {T<:Real}
    
    n = length(r)-1
    for i in 2:n
        for j in i:n
           
            if round(r[i] + r[j], digits = round_digit) > round(r[i-1] + r[j+1],digits = round_digit)
                return false
            end
        end
    end
    return true
end

"""
Create the maximal se for an edge for a given reward vector.
"""
function maximal_si(r::AbstractVector{T})where {T<:Real}
    n = length(r)
    s = zeros(T, n)  # Initialize s as zeros of type T
    
    # max_diff = -Inf  # Initialize max difference to negative infinity
    
    for i in 1:n-1
        max_diff = -Inf  # Reset max difference to negative infinity
        for j in 1:i
            if r[j+1] - r[j] > max_diff
                max_diff = r[j+1] - r[j]  # Update max difference
            end
        end
        s[i] = r[i+1] - max_diff  # Compute s(i)
    end
    
    return s
end

function label_distribution(L, S, label_names)
    # Extract the relevant rows for nodes in S
    L_S = L[S, :]
    # println("L_S = ", L_S)
    if ndims(L_S) == 1
        L_S = reshape(L_S, 1, :)
    end
    # Count occurrences of each label within subset S
    label_counts = vec(sum(L_S, dims=1))  # Sum along rows to get label frequencies

    print("label_counts = ", label_counts)
    # Convert to dictionary with label names
    label_distribution = Dict(label_names[i] => label_counts[i] for i in 1:length(label_names))

    return label_distribution
end

"""
    times_to_integer(x::Real)

Return the exponent n such that x * 10^n is an integer, 
assuming x is a finite decimal in base 10.
If x is not a finite decimal, throw an error.
"""
function times_to_integer(x::Real)
    r = rationalize(x)
    d = denominator(r)
    
    # Count how many times 2 divides d
    count2 = 0
    while d % 2 == 0
        d ÷= 2
        count2 += 1
    end
    
    # Count how many times 5 divides d
    count5 = 0
    while d % 5 == 0
        d ÷= 5
        count5 += 1
    end
    
    # If d != 1 now, then x has factors other than 2 or 5 in denominator
    # => x is not a finite decimal in base 10
    if d != 1
        error("Number $x is not a finite decimal in base 10.")
    end
    
    return max(count2, count5)
end

"""
    scale_to_integer(x::Real)

Return an integer equal to x * 10^n, where n is the minimal
number of times we need to multiply x by 10 to get an integer.
"""
function scale_to_integer(r::AbstractVector{T}) where {T<:Real}
    round_digits = 4
    r_rounded = [round(x,digits=round_digits) for x in r]
    times_vec = [times_to_integer(x) for x in r_rounded]
    n = maximum(times_vec)
    println("n = ", n)
    println("10^n = "   , 10^n)
    return [round(x * (10^n)) for x in r_rounded]
end

#The next set of functions are used for projecting a non-convex reward vector to a convex reward vector

"""
    project_to_convex(
        r:: Vector{Vector{Float64}})
Project an edge reward to a convex sequence.
"""
function project_to_convex(
    r:: Vector{Vector{Float64}},
    issup_list::Vector{Bool}
)
    m = length(r)
    r_proj = Vector{Vector{Float64}}(undef, m)
    for e = 1:m
        if !issup_list[e]
            breakpoints = 1:length(r[e])-1
            # println("breakpoints = ", breakpoints)
            values = r[e][2:end]
            new_slopes = draw_nearest_convex_lower_bound(breakpoints, values)

            r_proj[e] = [0.0]
            for i in 1:length(new_slopes)
                new_y = r_proj[e][end] + new_slopes[i] * 1.0 #(breakpoints[i] - (i == 1 ? 0 : breakpoints[i-1]))
                push!(r_proj[e], new_y)
            end
            
        else
            r_proj[e] = r[e]
        end
        
    end
    return r_proj
end

#The functions are used in the function project_to_convex--------------------------------
function find_steepest_slope(breakpoints, values)
    @assert length(breakpoints) == length(values) > 0
    slopes = values ./ breakpoints
    idx    = argmin(slopes)
    return slopes[idx], idx
end

function find_next_steepest_slope_from_current_point(breakpoints, values, origin)
    # return least_slope, least_slope_breakpoint
    n = length(values)
    if origin == n
        return 0.0, origin
    end

    # Slicing from origin+1 to end
    idx_range = origin+1:n
    sub_slopes = (values[idx_range] .- values[origin]) ./ (breakpoints[idx_range] .- breakpoints[origin])

    # Find minimum slope in the sub-range
    sub_idx_offset = argmin(sub_slopes)
    min_slope      = sub_slopes[sub_idx_offset]
    # Convert offset into the actual index in the original array
    min_idx        = idx_range[sub_idx_offset]

    return min_slope, min_idx
end

function find_next_nearest_supmodular_slope(breakpoints, origin,nslopes)
    n = length(nslopes)
    if origin == n
        return 0.0  # or return 0.0, depending on your logic  # Nan to 0.0 change
    end

    next_idx = origin + 1
    return nslopes[next_idx] < 0 ? 0.0 : nslopes[next_idx]
end

function draw_nearest_convex_lower_bound(breakpoints, values)
    n = length(breakpoints)

    @assert n == length(values)

    slopes = diff(values) ./ diff(breakpoints)
    nslopes = [values[1], slopes...] # Add the first slope
    # new_values = Float64[values...]  # Ensure new_values is a float array
    new_slopes = fill(NaN, n)  # Initialize new_slopes with float values

    least_slope_1, least_slope_breakpoint_1 = find_steepest_slope(breakpoints, values)
    for i in 1:least_slope_breakpoint_1
        new_slopes[i] = least_slope_1
        # new_values[i] = max_slope_1 * breakpoints[i]
    end

    if least_slope_breakpoint_1 == n
        return  new_slopes #new_values, new_slopes
    end

    least_slope_breakpoint = least_slope_breakpoint_1
    least_slope_next, least_slope_breakpoint_next = find_next_steepest_slope_from_current_point(breakpoints, values, least_slope_breakpoint)

    while least_slope_breakpoint_next != least_slope_breakpoint
        # intercept = values[max_slope_breakpoint] - max_slope_next * breakpoints[max_slope_breakpoint]
        for i in (least_slope_breakpoint+1):least_slope_breakpoint_next
            new_slopes[i] = least_slope_next
            # new_values[i] = max_slope_next * breakpoints[i] + intercept
        end
        least_slope_breakpoint = least_slope_breakpoint_next
        if least_slope_breakpoint != n
            least_slope_next, least_slope_breakpoint_next = find_next_steepest_slope_from_current_point(breakpoints, values, least_slope_breakpoint)
        else
            break
        end
    end

    if least_slope_breakpoint == n
        return new_slopes #new_values, new_slopes
    end

    nearest_slope_breakpoint = least_slope_breakpoint
    new_slp = find_next_nearest_supmodular_slope(breakpoints, nearest_slope_breakpoint, nslopes)
    # new_slopes[nearest_slope_breakpoint+1] = (new_values[nearest_slope_breakpoint+1] - new_values[nearest_slope_breakpoint]) / (breakpoints[nearest_slope_breakpoint+1] - breakpoints[nearest_slope_breakpoint])
    new_slopes[nearest_slope_breakpoint+1] = new_slp

    nearest_slope_breakpoint += 1
    while nearest_slope_breakpoint != n
        new_slp = find_next_nearest_supmodular_slope(breakpoints, nearest_slope_breakpoint, nslopes)
        # new_slopes[nearest_slope_breakpoint+1] = (new_values[nearest_slope_breakpoint+1] - new_values[nearest_slope_breakpoint]) / (breakpoints[nearest_slope_breakpoint+1] - breakpoints[nearest_slope_breakpoint])
        new_slopes[nearest_slope_breakpoint+1] = new_slp
        nearest_slope_breakpoint += 1
    end

    if any(isnan, new_slopes)
        println("Error: Slopes not updated properly")
    end

    return  new_slopes #new_values
end
# ------------------------------------------------------------

"""
    cumulative_to_incremental(cr_vec)

Given a cumulative reward vector `cr_vec`, return a new vector
`inc_vec` such that inc_vec[i] = cr_vec[i] - cr_vec[i-1].
"""
function cumulative_to_incremental(cr_vec::Vector{Vector{Float64}})
    # We assume cum_vec has at least 2 elements (e.g. cum_vec[0], cum_vec[1], ...).
    inc_vec = Vector{Vector{Float64}}()
    for i in 1:length(cr_vec)
        inc_vec_i = Vector{Float64}()
        push!(inc_vec_i, cr_vec[i][1])
        for j in 2:length(cr_vec[i])
            push!(inc_vec_i, round(cr_vec[i][j] - cr_vec[i][j-1],digits = 3))
        end
        push!(inc_vec, inc_vec_i)
    end
    return inc_vec
end
