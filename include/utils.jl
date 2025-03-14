"""
    deduplicate(H)

Remove duplicate nodes in one hyperedge.
"""
function deduplicate(
    H::SparseMatrixCSC{Tf, Ti},
) where {Tf, Ti <: Integer}
    # deal with the corner case that one vertex 
    # appears in one hyperedge multiple times 
    I, J, _ = findnz(H)
    return sparse(I, J, ones(Tf, length(I)), size(H)...)
end


"""
    adj_del_selfloops(A)

Delete all self-loops in an adjacency matrix A.  
"""
function adj_del_selfloops(
    A::SparseMatrixCSC{Tf, Ti}
) where {Ti <: Integer, Tf}
    A[diagind(A)] .= zero(Tf) 
    return dropzeros(A)
end


"""
    inc_del_selfloops(H, Ht)

Delete all self-loops in an incidence matrix H. 
"""
function inc_del_selfloops(
    H::SparseMatrixCSC{Tf, Ti},
    Ht::SparseMatrixCSC{Tf, Ti},
) where {Ti <: Integer, Tf} 
    m, _ = size(H)
    order = H_order(Ht)
    eids = findall(x->order[x]>1, 1:m)
    return H[eids, :], Ht[:, eids] 
end


"""
    adj2elist(A, [undirected=true])

Convert an adjacency matrix of an (un)directed graph to a list of edges.
"""
function adj2elist(
    A::SparseMatrixCSC{Tf, Ti},
    undirected=true,
) where {Tf, Ti <: Integer}
    n = size(A, 1)
    rowval = A.rowval    
    colptr = A.colptr
    nzval = A.nzval
    Hyperedges = Vector{Vector{Ti}}()
    for u = 1:n
        for nzi in colptr[u]:colptr[u+1]-1
            v = rowval[nzi]
            edge = Vector{Ti}()
            if u <= v  
                push!(edge, u)
                push!(edge, v)
                push!(Hyperedges, edge)
            end
            if (u > v && !undirected)
                push!(edge, u)
                push!(edge, v)
                push!(Hyperedges, edge)
            end
        end
    end
    return Hyperedges
end


"""
    elist2inc(Hyperedges, n)

Convert a list of hyperedges into an incidence matrix of a hypergraph with n nodes.
"""
function elist2inc(
    Hyperedges::Vector{Vector{Ti}},
    n::Ti,
) where {Ti <: Integer}
    eid = 0
    I = Ti[]
    J = Ti[]
    V = Float64[] 
    for i = eachindex(Hyperedges)
        eid += 1
        for v in Hyperedges[i]
            push!(I, eid)
            push!(J, v)
            push!(V, 1.0)
        end
    end
    return sparse(I, J, V, length(Hyperedges), n)
end


"""
    inc2adj(H, Ht)

Turn an incidence matrix into an adjacency matrix.
Only works for normal graphs without self-loops.
"""
function inc2adj(
    H::SparseMatrixCSC{Tf, Ti},
    Ht::SparseMatrixCSC{Tf, Ti},
) where {Tf, Ti <: Integer} 
    m, n = size(H)
    order = H_order(Ht) 
    # some sanity check
    # make sure this is a normal graph
    @assert maximum(order) == 2
    @assert minimum(order) == 2
    colptr = Ht.colptr
    rowval = Ht.rowval
    I = Ti[]; J = Ti[]
    for e = 1:m
        u = colptr[e]; v = colptr[e]+1
        u = rowval[u]; v = rowval[v]
        push!(I, u); push!(J, v)
        push!(I, v); push!(J, u)
    end
    return sparse(I, J, ones(Float64, length(I)), n, n)
end


"""
    inc2elist(Ht)

Turn the transpose of one incidence matrix into a list of hyperedges.
"""
function inc2elist(
    Ht::SparseMatrixCSC{Tf, Ti},
) where {Tf, Ti <: Integer}
    n, m = size(Ht)
    hyperedges = Vector{Vector{Ti}}()
    for e = 1:m
        st = Ht.colptr[e]
        ed = Ht.colptr[e+1]-1
        push!(hyperedges, Ht.rowval[st:ed])
    end
    return hyperedges
end


"""
    H_largest_component(H, Ht)

Compute the largest component of the hypergraph. Here we assume the hypergraph is undirected.
""" 
function H_largest_component(
    H::SparseMatrixCSC{Tf, Ti},
    Ht::SparseMatrixCSC{Tf, Ti},
) where {Ti <: Integer, Tf} 
    m, n = size(H)
    # connected component id of each vertex
    cc_id = zeros(Ti, n) 
    # connected component count
    cc_cnt = 0
    for i = 1:n
        if cc_id[i] != 0 # already visited
            continue
        end
        cc_cnt += 1
        cc_id[i] = cc_cnt
        q = Queue{Ti}()
        enqueue!(q, i)
        while !isempty(q)
            u = dequeue!(q)
            for j = H.colptr[u]:(H.colptr[u+1]-1)
                e = H.rowval[j]
                for k = Ht.colptr[e]:(Ht.colptr[e+1]-1)
                    v = Ht.rowval[k]
                    if cc_id[v] == 0
                        cc_id[v] = cc_cnt
                        enqueue!(q, v)
                    end
                end
            end
        end
    end
    cc_sz = zeros(Ti, cc_cnt)
    for i = 1:n
        cc_sz[cc_id[i]] += 1
    end
    # id of the largest connected component
    lcc_id = argmax(cc_sz) 
    # vertices in the largest connected component
    lcc = findall(x->cc_id[x]==lcc_id, 1:n) 
    _, lcc_eid = etouchS(H, lcc)
    return H[lcc_eid, lcc], Ht[lcc, lcc_eid], lcc,lcc_eid
end


"""
    inc_sortbysize(H, Ht)

Sort rows(hyperedges) of an incidence matrix according to their sizes.
"""
function inc_sortbysize(
    H::SparseMatrixCSC{Tf, Ti},
    Ht::SparseMatrixCSC{Tf, Ti},
) where {Ti <: Integer, Tf}
    order = H_order(Ht)  
    e_ord = sortperm(order)
    return H[e_ord, :], Ht[:, e_ord],e_ord
end


"""
    precision(S, Target)

Computing the precision of S with Target as the ground truth.
"""
function precision(
    S::Vector{Ti},
    Target::Vector{Ti},
)where {Ti}
    return length(intersect(S, Target)) / length(S)
end


"""
    recall(S, Target)

Computing the recall of S with Target as the ground truth.
"""
function recall(
    S::Vector{Ti},
    Target::Vector{Ti},
) where {Ti}
    return length(intersect(S, Target)) / length(Target)
end


"""
    F1score(S, Target)

Computing the F1 score of S with Target as the ground truth.
"""
function F1score(
    S::Vector{Ti},
    Target::Vector{Ti},
) where {Ti}
    return 2 * length(intersect(S, Target)) / (length(S) + length(Target))
end

"""
    Jaccard(S, Target)

Computing the Jaccard index of S with Target as the ground truth.
"""
function Jaccard(
    S::Vector{Ti},
    Target::Vector{Ti},
) where {Ti}
    return length(intersect(S, Target)) / length(union(S, Target))
end


"""
    etouchS(H, S)

Count and list those edges incident to a set S.
"""
function etouchS(
    H::SparseMatrixCSC{Tf, Ti},
    S::Vector{Ti},
) where {Tf <: AbstractFloat, Ti <: Integer}
    HS = H[:, S] 
    rp_S = HS.rowval 
    Sedges = unique(rp_S) 
    return length(Sedges), Sedges
end


"""
    einS(H, order, S)

Given the incidence matrix of a hypergraph and hyperedge orders, and a vertex set S,
compute the number of hyperedges fully contained in S and list those hyperedges.
"""
function einS(
    H::SparseMatrixCSC{Tf, Ti},
    order::Vector{Ti},
    S::Vector{Ti},
) where {Tf, Ti <: Integer}
    HS = H[:, S]
    rp_HS = HS.rowval 
    etouchS_list = unique(rp_HS) 
    @inbounds for i in rp_HS
        order[i] -= 1 
    end
    einS_list = findall(x->x==0, order[etouchS_list])
    einS_list = etouchS_list[einS_list]
    einS_count = length(einS_list)
    @inbounds for i in rp_HS
        order[i] += 1
    end
    return einS_count, einS_list 
end


"""
    H_deg(H)

Degree of each vertex in H
"""
function H_deg(
    H::SparseMatrixCSC{Tf, Ti},
) where {Tf, Ti <: Integer}
    return Ti.(sum(H, dims=1)[1, :])
end


"""
    H_order(Ht)

Size of each hyperedge in H
"""
function H_order(Ht::SparseMatrixCSC{Tf, Ti},
) where {Tf, Ti <: Integer}
    return Ti.(sum(Ht, dims=1)[1, :])
end



"""
    diameter(H, Ht)

Compute the diameter of a hypergraph, i.e. the longest shortest path.
We assume the hypergraph is connected.
"""
function diameter(
    H::SparseMatrixCSC{Tf, Ti},
    Ht::SparseMatrixCSC{Tf, Ti},
) where {Tf <: AbstractFloat, Ti <: Integer}
    dis = bfs(H, Ht, 1)
    s = argmax(dis) 
    dis = bfs(H, Ht, s)
    return maximum(dis)
end


