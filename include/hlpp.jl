module hlpp
using SparseArrays
using Printf

# Push Relabel solver for maximum s-t flow, minimum s-t cut problems
# Throughout this code, we always assume source node s is node 1, 
# and sink node t is the last one
# as in this project we won't consider integral flow, we always set flowtol = 1e-6

mutable struct stFlow{Tf, Ti}
    flowvalue::Tf # gives you the max-flow value
    cutvalue::Tf # gives min-cut value, which should equal flowvalue, # but may differ by a small amount.
    source_nodes::Vector{Ti} # give the indices of the nodes attached to the source
    height::Vector{Ti} # gives the final height of each node
    C::SparseMatrixCSC{Tf, Ti} # gives the original capacity matrix
    F::SparseMatrixCSC{Tf, Ti} # gives the values of the flows on each edge
end


"""
maxflow implementation using the highest label preflow push method.

Given a sparse matrix A representing a weighted and possibly directed graph, 
return the maximum s-t flow.

flowtol = tolerance parameter for whether there is still capacity available on
            an edge. Helps avoid rounding errors. Default is 1e-6.

Returns F, which is of type stFlow.
"""
function maxflow(
    B::SparseMatrixCSC{Tf, Ti},
    flowtol::Tf = zero(Tf),
) where {Ti <: Integer, Tf}

    n = size(B, 1)
    # println("n is $n")

    S, FlowMat, height, flowvalue = HLPP(B, flowtol)
    inS = zeros(Bool, n)
    inS[S] .= true
    cutvalue = zero(Tf) 
    I, J, V = findnz(B)
    for i = eachindex(I)
        if inS[I[i]] && !inS[J[i]]
            cutvalue += V[i]
        end
    end
    return stFlow{Tf, Ti}(flowvalue, cutvalue, S, height, B, FlowMat)
end


"""
This maxflow code assumes that A represents the adjacencies between
non-terminal nodes. Edges adjecent to source node s and sink node t
are given by vectors svec and tvec.

This code sets s as the first node, and t as the last node.
"""
function maxflow(
    A::SparseMatrixCSC{Tf, Ti},
    svec::Vector{Tf},
    tvec::Vector{Tf},
    flowtol::Tf = zero(Tf),
) where {Ti <: Integer, Tf}

    n = size(A, 1)

    # Directly set up the flow matrix
    C = [spzeros(Tf, 1,1) sparse(svec') spzeros(Tf, 1,1);
         spzeros(Tf, n, 1) A sparse(tvec);
         spzeros(Tf, 1,1) spzeros(Tf, 1, n) spzeros(Tf, 1,1)]

    return maxflow(C, flowtol)
end


"""
Given a flow, stored in an stFlow object, return the set of nodes attached to
the source
"""
function source_nodes(F::stFlow)
    # Run a bfs from the sink node. Anything with distance
    # n is disconnected from the sink. Thus it's part of the minimium cut set
    n = size(F.C,2)
    S = Vector{Int64}()
    for i = 1:n
        if F.height[i] >= n
            push!(S,i)
        end
    end

    # Sanity checks: source node is on source side, sink node is on sink side
    @assert(~in(n,S))
    @assert(in(1,S))

    return S
end


"""
Given a flow, stored in an stFlow object, return the set of nodes attached to
the sink
"""
function sink_nodes(F::stFlow)
    # Run a bfs from the sink node. Anything with distance < n is sink-attached.
    n = size(F.C,2)
    T = Vector{Int64}()
    for i = 2:n
        if F.height[i] < n
            push!(T,i)
        end
    end

    # Sanity checks
    @assert(in(n,T))
    @assert(~in(1,T))

    return T
end

"""
Gives the cut as a list of edges.
"""
function cut_edges(F::stFlow)
    # Run a bfs from the sink node to get source and sink sets
    n = size(F.C,2)
    T = Vector{Int64}()
    S = Vector{Int64}()
    for i = 1:n
        if F.height[i] < n
            push!(T,i)
        else
            push!(S,i)
        end
    end

    I,J,V = findnz(F.C[S,T])
    return [S[I] T[J]]
end


"""
Gives the non-terminal cut edges.
"""
function cut_edges_nonterminal(F::stFlow)
    # Run a bfs from the sink node to get source and sink sets
    Edges = cut_edges(F)
    T = Vector{Int64}()
    S = Vector{Int64}()
    for i = 1:size(Edges,1)
        I = Edges[i,1]
        J = Edges[i,1]
        if I != n && I!= 1 && J != n && J != 1 
            push!(S,I)
            push!(T,J)
        end
    end
    return [S T]
end


"""
Main function for Highest Label Preflow Push Method
"""
function HLPP(
    C::SparseMatrixCSC{Tf, Ti},
    flowtol::Tf;
) where {Ti <: Integer, Tf}
    n = size(C, 1) # number of vertices in the graph
    m = nnz(C)

    # height(level) of nodes
    height = zeros(Ti, n)

    # auxiliary array for creating adjacency lists
    cursor = zeros(Ti, n)

    # edges with id from m_starts[i] to m_starts[i+1]-1 are from node i
    m_starts = zeros(Ti, n + 1)

    # number of edges from node i (self-loops excluded)
    d = zeros(Ti, n)

    # the node that edge i points to
    to = zeros(Ti, 2*m)

    # rev[i] stores the id of the reverse edge of edge i
    rev = zeros(Ti, 2*m)

    # rescap[i] = capacity of edge i - flow on edge i
    rescap = zeros(Tf, 2*m) # capacity in the residual graph


    function ConstructAdjlist()
        I, J, V = findnz(C)
        # compute d
        for k = eachindex(I)
            u = I[k]; v = J[k];
            if u != v # remove self-loops
                d[u] += 1 
                d[v] += 1 
            end
        end
        # cursor is the prefix sum of d 
        for i = 1:n
            cursor[i] = (i == 1) ? 1 : cursor[i - 1] + d[i - 1]
            m_starts[i] = cursor[i]
        end
        m_starts[n + 1] = cursor[n] + d[n]
        for k = eachindex(I)
            u = I[k]; v = J[k]; c = V[k]
            if u != v # remove self-loops
                # for each directed edge, create two edges, with capacity c and 0
                curu = cursor[u]; curv = cursor[v] 
                to[curu] = v; rev[curu] = curv; rescap[curu] = c
                to[curv] = u; rev[curv] = curu; rescap[curv] = zero(Tf)
                cursor[u] += 1
                cursor[v] += 1
            end
        end
    end
    ConstructAdjlist()

    # active nodes
    # for each level, use a cyclic linked list to store active nodes
    excess = zeros(Tf, n)
    # in case we add a node to the active list twice
    # this will happen when there are roundoff errors
    inexcess = zeros(Bool, n) 
    excess_next = zeros(Ti, n * 2 + 1)
    # maximum height of active nodes
    excess_height = zero(Ti) 

    # Infinite capacity
    if Tf <: AbstractFloat
        infinite_cap = 1e15
    elseif Tf <: Integer
        infinite_cap = Tf(round(typemax(Tf) / 2))
    else
        error("Type of Flow not supported")
    end
    infinite_height = Ti(round(typemax(Ti) / 2))

    function excess_insert(v::Ti, h::Ti) 
        # insert v into the cyclic linked list for level h
        excess_next[v] = excess_next[n + 1 + h]
        excess_next[n + 1 + h] = v
        inexcess[v] = true
        if h > excess_height
            excess_height = h
        end
    end

    function excess_add(v::Ti, f::Tf)
        excess[v] += f
        if excess[v] <= f + flowtol && !inexcess[v]
            excess_insert(v, height[v])
        end
    end

    function excess_remove(v::Ti, f::Tf)
        excess[v] -= f
    end

    # gap heuristic
    # once one gap is observed, then those nodes with label 
    # larger than this level is disconnected from sink, so
    # it is useless to push flow from them, we can relabel 
    # them to infinite height or n + 1
    # we use cyclic double linked lists to implement gap heuristics
    gap_prev = zeros(Ti, n * 2 + 1)
    gap_next = zeros(Ti, n * 2 + 1)
    gap_highest = zero(Ti)

    function gap_insert(v::Ti, h::Ti)
        # insert v into the cyclic linked list for level h
        gap_prev[v] = n + 1 + h 
        gap_next[v] = gap_next[n + 1 + h]
        gap_prev[gap_next[v]] = v
        gap_next[gap_prev[v]] = v
        if h > gap_highest
            gap_highest = h
        end
    end


    function gap_erase(v::Ti)
        # erase v from the cyclic linked list for level h
        gap_next[gap_prev[v]] = gap_next[v]
        gap_prev[gap_next[v]] = gap_prev[v]
    end

    
    function update_height(v::Ti, h::Ti)
        if height[v] != infinite_height 
            gap_erase(v)
        end
        height[v] = h
        if h != infinite_height 
            gap_insert(v, h)
            if excess[v] > flowtol && !inexcess[v] 
                excess_insert(v, h)
            end
        end
    end

    discharge_count::Ti = 0
    function global_relabel()
        discharge_count = 0 

        #initialize head of linked lists
        for i = n + 1: 2 * n + 1
            excess_next[i] = i
            gap_prev[i] = i
            gap_next[i] = i
        end
        for i = 1:n
            inexcess[i] = false
        end
        fill!(height, infinite_height)
        height[n] = 0
        queue = zeros(Ti, n)
        head = 1
        tail = 1
        queue[tail] = n
        # perform a reverse bfs from the sink node
        # using edges in the residual graph 
        while head <= tail
            u = queue[head]
            head += 1
            for i in m_starts[u]:m_starts[u + 1] - 1 
                v = to[i]
                rev_edge = rev[i] 
                # if v has an edge to u with positive residual capacity
                # then we update the height of v and add it to the queue
                if rescap[rev_edge] > flowtol && height[v] > height[u] + 1 
                    update_height(v, height[u] + 1)
                    tail += 1
                    queue[tail] = v
                end
            end
        end
    end

    function push(u::Ti, v::Ti, eid::Ti, f::Tf)
        excess_remove(u, f)
        excess_add(v, f)
        rescap[eid] -= f
        rescap[rev[eid]] += f
    end

    # pointers for current arc heuristic
    cur_arc = deepcopy(m_starts) 

    function discharge(u::Ti)
        h = n 
        pos = cur_arc[u] 
        m_end = m_starts[u + 1] - 1
        while cur_arc[u] <= m_end 
            e = cur_arc[u]
            v = to[e] 
            if rescap[e] > flowtol 
                if height[u] == height[v] + 1
                    push(u, v, cur_arc[u], min(excess[u], rescap[e]))
                    if excess[u] <= flowtol 
                        return
                    end
                else
                    if height[v] < h 
                        h = height[v]
                    end
                end
            end
            cur_arc[u] += 1
        end
        cur_arc[u] = m_starts[u] 
        while cur_arc[u] < pos
            e = cur_arc[u]
            v = to[e] 
            if rescap[e] > flowtol 
                if height[u] == height[v] + 1
                    push(u, v, cur_arc[u], min(excess[u], rescap[e]))
                    if excess[u] <= flowtol 
                        return
                    end
                else
                    if height[v] < h 
                        h = height[v]
                    end
                end
            end
            cur_arc[u] += 1
        end
        discharge_count += 1
        if gap_next[gap_next[n + 1 + height[u]]] <= n
            update_height(u, h == n ? infinite_height : h + 1)
        else
            # if one gap is observed, then those nodes with label
            # larger than this gap will be relabeled
            oldh = height[u]
            for h = height[u]:gap_highest
                while gap_next[n + 1 + h] <= n
                    j = gap_next[n + 1 + h]
                    height[j] = infinite_height
                    gap_erase(j)
                end
            end
            gap_highest = oldh - 1
        end
    end

    function print_key_variables()
        for i = 1:n
            print("$(height[i]) ")
        end
        println("")
        for i = 1:n
            print("$(excess[i]) ")
        end
        println("")
        for i = 1:n
            for j = m_starts[i]:m_starts[i+1]-1
                println("$i $(to[j]) $(rescap[j])")
            end
        end
    end

    global_relabel()

    # if source can reach sink
    if height[1] < infinite_height
        excess_add(1, infinite_cap)
        excess_remove(n, infinite_cap)
        while excess_height > 0 
            while true
                # pick the active node with the highest height
                v = excess_next[n + 1 + excess_height]
                if v > n
                    break
                end
                excess_next[n + 1 + excess_height] = excess_next[v]
                inexcess[v] = false
                if height[v] != excess_height
                    continue
                end
                if excess[v] > flowtol
                    discharge(v)
                end
                #print_key_variables()

                # perform the global relabeling heuristic every 4n discharges
                if discharge_count >= 4 * n
                    global_relabel()
                end
            end
            excess_height -= 1
        end
    end
    S = Vector{Ti}()
    global_relabel()
    #print_key_variables()
    for i = 1:n
        if height[i] == infinite_height
            push!(S, i)
        end
    end
    I_resC = Vector{Ti}()
    J_resC = Vector{Ti}()
    V_resC = Vector{Tf}()
    for i = 1:n
        for j = m_starts[i]:m_starts[i+1]-1
            push!(I_resC, i)
            push!(J_resC, to[j])
            push!(V_resC, rescap[j])
        end
    end
    resC = sparse(I_resC, J_resC, V_resC, n, n)
    F = C .- resC
    return S, F, height, excess[n] + infinite_cap
end

end


