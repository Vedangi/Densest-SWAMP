
"""
    preprocess(H, [delmultiedge=false, sortbysize=true])

Perform all preprocessing steps for hypergraphs. 
By default, we don't remove multi-edges and sort hyperedges by size.
"""
function preprocess(
    H::SparseMatrixCSC{Tf, Ti};
    delmultiedge::Bool = false,
) where {Ti <: Integer, Tf}
    H = deduplicate(H)
    Ht = sparse(H')
    if delmultiedge
        m, n = size(H)
        elist = inc2elist(Ht)  
        elist = unique(elist)
        H = elist2inc(elist, n)
        Ht = sparse(H')
    end
    H, Ht = inc_del_selfloops(H, Ht)
    H, Ht, p,edgeidx = H_largest_component(H, Ht)
    H, Ht,e_order = inc_sortbysize(H, Ht)
    return H, Ht, p,edgeidx,e_order
end


"""
    read_nverts_simplices(datafolder, dataset)

Read raw data from a file. The file should contain two files:
    filepath-nverts.txt
    filepath-simplices.txt
"""
function read_nverts_simplices(datafolder, dataset)
    nverts_file = datafolder*"/"*dataset*"/"*dataset*"-nverts.txt" 
    simplices_file = datafolder*"/"*dataset*"/"*dataset*"-simplices.txt"  
    order = Int64[] 
    open(nverts_file) do order_file
        for l in eachline(order_file)
            esize = parse(Int64, l)
            push!(order, esize)
        end
    end
    Hyperedges = Vector{Vector{Int64}}()
    n = 0
    open(simplices_file) do hyperedges_file 
        for i in eachindex(order) 
            e = Vector{Int64}()
            for j in 1:order[i]
                l = readline(hyperedges_file)
                v = parse(Int64, l)
                n = max(n, v)
                push!(e, v)
            end
            push!(Hyperedges, e)
        end
    end
    return elist2inc(Hyperedges, n)
end


"""
    read_hyperedges_txt(datafolder, dataset)

Read raw data from a file. The file should contain three files:
    hyperedges-dataset.txt
    label-names-dataset.txt
    node-labels-dataset.txt
"""
function read_hyperedges_txt(datafolder, dataset)
    hyperedges_file = datafolder*"/"*dataset*"/hyperedges-"*dataset*".txt"
    label_names_file = datafolder*"/"*dataset*"/label-names-"*dataset*".txt"
    node_labels_file = datafolder*"/"*dataset*"/node-labels-"*dataset*".txt"

    ## read label names
    n_labels = 0
    label_names = Vector{String}()
    open(label_names_file) do file
        for l in eachline(file)
            n_labels += 1
            push!(label_names, l)
        end
    end

    ## read node labels
    I = Vector{Int64}()
    J = Vector{Int64}()
    n_nodes = 0

    open(node_labels_file) do file
        for l in eachline(file)
            n_nodes += 1
            labellist = split(l, ',')
            for label in labellist
                push!(I, n_nodes)
                push!(J, parse(Int64, label))
            end
        end
    end
    L = sparse(I, J, ones(Float64, length(I)), n_nodes, n_labels)

    ## read hyperedges
    m = 0
    U = Vector{Int64}()
    V = Vector{Int64}()
    open(hyperedges_file) do file 
        for l in eachline(file)
            m += 1
            vlist = split(l, ',')
            for v in vlist
                push!(U, m)
                push!(V, parse(Int64, v))
            end
        end
    end
    H = sparse(U, V, ones(Float64, length(U)), m, n_nodes)
    res = Dict(
        "H" => H,
        "L" => L,
        "label_names" => label_names,
    )
    return res
end

"""
    read_hyperedges_edgelabeled_txt(datafolder, dataset)

Read raw data from a file. The file should contain three files:
    hyperedges-dataset.txt
    edgelabel-names-dataset.txt
    edge-labels-dataset.txt
"""
function read_hyperedges_edgelabeled_txt(datafolder, dataset)
    hyperedges_file = datafolder*"/"*dataset*"/hyperedges-"*dataset*".txt"
    label_names_file = datafolder*"/"*dataset*"/edgelabel-names-"*dataset*".txt"
    edge_labels_file = datafolder*"/"*dataset*"/edge-labels-"*dataset*".txt"

    ## read label names
    n_labels = 0
    label_names = Vector{String}()
    open(label_names_file) do file
        for l in eachline(file)
            n_labels += 1
            push!(label_names, l)
        end
    end

    ## read edge labels
    I = Vector{Int64}()
    J = Vector{Int64}()
    n_edges = 0

    open(edge_labels_file) do file
        for l in eachline(file)
            n_edges += 1
            labellist = split(l, ',')
            for label in labellist
                push!(I, n_edges)
                push!(J, parse(Int64, label))
            end
        end
    end
    # EL now is an (n_edges x n_labels) sparse matrix.
    EL = sparse(I, J, ones(Float64, length(I)), n_edges, n_labels)

    ## read hyperedges
    m = 0
    U = Vector{Int64}()
    V = Vector{Int64}()
    n_nodes = 0 # number of nodes
    open(hyperedges_file) do file 
        for l in eachline(file)
            m += 1
            vlist = split(l, ',')
            for v in vlist
                vertex = parse(Int64, v)
                push!(U, m)
                push!(V, parse(Int64, v))
                n_nodes = max(n_nodes, vertex)
            end
        end
    end
    H = sparse(U, V, ones(Float64, length(U)), m, n_nodes)
    res = Dict(
        "H" => H,
        "EL" => EL,
        "label_names" => label_names,
    )
    return res
end


"""
    read_txt(filepath)

Read an undirected graph from a txt file, return the adjacency matrix.

Corresponding Datasets: email-Enron ca-AstroPh ca-HepPh com-amazon com-youtube
"""
function read_txt(filepath)
    I = Array{Int}(undef, 0) 
    J = Array{Int}(undef, 0)
    linecounter = 0
    m = 0
    open(filepath) do file
        for l in eachline(file)  
            linecounter += 1
            if linecounter >= 5                 
                u, v = split(l, '\t')
                u = parse(Int, u)
                v = parse(Int, v)
                if u != v
                    push!(I, u); push!(J, v)
                    push!(I, v); push!(J, u)
                end
            end
        end
    end
    I .+= 1
    J .+= 1
    n = maximum(J)
    A = sparse(I, J, ones(Float64, length(I)), n, n)
    A, _ = largest_component(A)
    return A 
end


"""
    preprocess_graph(dataset, [delmultiedge=false, filefolder=homedir()*"/local-DHSG/data/"])

Preprocess a hypergraph dataset and store it in a .mat file. 

# Arguments
- dataset: name of the dataset
- delmultiedge: whether to delete multi-edges
- filefolder: folder to store the .mat file
"""
function preprocess_graph(
    dataset,
    delmultiedge=false,
    # filefolder = homedir()*"/local-DHSG/data/",
    filefolder = "datafolder/",
)
    if dataset in [
        "threads-math-sx",
        "threads-ask-ubuntu",]
        H = read_nverts_simplices(filefolder, dataset)
        H, Ht, p = preprocess(H, delmultiedge=delmultiedge)
        newres = Dict(
            "H"=>H,
        )
    elseif dataset in [
        "amazon-reviews",
        "trivago-clicks",
        "walmart-trips",
        "contact-high-school",
        "contact-primary-school",
        "house-bills",
        "house-committees",
        "mathoverflow-answers",
        "senate-bills",
        "senate-committees",
    ]
        res = read_hyperedges_txt(filefolder, dataset)
        H = res["H"]
        L = res["L"]
        label_names = res["label_names"]
        H, Ht, p = preprocess(H, delmultiedge=delmultiedge)
        L = L[p, :]
        newres = Dict(
            "H"=>H,
            "L"=>L,
            "label_names"=>label_names,
        )
    elseif dataset in [
        "ca-AstroPh",
        "ca-HepPh",
        "email-Enron",
        "com-amazon",
        "com-youtube",
    ]
        if dataset in ["ca-AstroPh", "ca-HepPh", "email-Enron"]
            A = read_txt(filefolder*dataset*".txt")
        else
            A = read_txt(filefolder*dataset*".ungraph.txt")
        end
        newres = Dict(
            "A"=>A,
        )
    elseif dataset in [
        "cooking-dataset"   
    ]
        res = read_hyperedges_edgelabeled_txt(filefolder, dataset)
        H = res["H"]
        EL = res["EL"]
        label_names = res["label_names"]
        H, Ht, p, edgeidx,e_order = preprocess(H, delmultiedge=delmultiedge)
        EL = EL[edgeidx, :]
        EL = EL[e_order, :]
        newres = Dict(
            "H"=>H,
            "EL"=>EL,
            "label_names"=>label_names,
        )
    else
        error("Dataset not supported")
    end
    if delmultiedge 
        matwrite(filefolder*dataset*"-unique.mat", newres)
    else
        matwrite(filefolder*dataset*".mat", newres)
    end
end


"""
    load_graph(dataset, [delmultiedge=false, filefolder=homedir()*"/local-DHSG/data/"])

Once preprocessed, load the hypergraph from a .mat file.
"""
function load_graph(
    dataset,
    delmultiedge=false,
    # filefolder=homedir()*"/local-DHSG/data/",
    filefolder="datafolder/",
)
    if delmultiedge 
        data = matread(filefolder*dataset*"-unique.mat")
    else
        data = matread(filefolder*dataset*".mat")
    end
    return data
end
