## Load a hypergraph
include("hyper_format.jl")
include("gendensubhyper-ILP.jl")

using MAT

filename = "contact-high-school"

Mt = matread("../datafolder/large-datasets/$filename.mat")
H = Mt["H"]
H = sparse(Float64.(Mt["H"]))
n = size(H,2)
println("n = ", n)
println("m = ", size(H,1))

EdgeList = incidence2elist(H)
# H = elist2incidence(EdgeList,n)

order = vec(round.(Int64,sum(H,dims = 2)))
L = round(Int64,sum(H))
M = size(H,1)
eweights = ones(M)

## Create the reward vector for each hyperedge, and solve the problem

reward_d = allbutone_reward(order)
# reward_d = standard_dshg_reward(order)

eS , alpS = genden_ILP_full(EdgeList,reward_d,order,eweights,n,false);
S = findall(x->x==1,eS)
