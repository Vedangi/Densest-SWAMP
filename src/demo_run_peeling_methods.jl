include("header.jl")

flowtol = 1e-8
dataname = "contact-high-school"

data = load_graph(dataname, true)

H = sparse(Float64.(data["H"]))
Ht = sparse(H')
edge_list = incidence2elist(H)
vertex2edges = build_vertex_to_edges(edge_list)
@show size(H), mean(H_order(Ht))
order = vec(round.(Int64, sum(H, dims=2)))
L = round(Int64, sum(H))
M = size(H, 1)
n = size(H, 2)

eweights = ones(M)

reward_d, s_d, issup_list = standard_reward(order)

#--------------------------- PeelMax: Peeling with maximal s_e for every edge e ------------------------------
gp_res_max_se = peeling_with_se(H, reward_d, s_d)

# Using edge_list instead of H as input
gp_res_max_se = peeling_with_se_edge_list(edge_list, reward_d, s_d, vertex2edges)

#--------------------------- Greedy: peeling with s_e = r_e for every edge e ---------------------------------
gp_res_same_se = peeling_with_se(H, reward_d, reward_d)

#--------------------------- DegPeel: peeling by maximal degree------------------------------------")
gp_res_deg = peeling_by_degree(H, Ht, reward_d)
@show gp_res_deg["optval"] #, gp_res["optsol"]# gp_res["peeling_ord"]

# using edge list
gp_res_deg = peeling_by_degree_edge_list(edge_list, reward_d, vertex2edges)

# check if length of opt solution is the same
@show length(gp_res_same_se["optsol"]), length(gp_res_deg["optsol"])

#check if the peeling order is the same 
@show gp_res_same_se["peeling_ord"] == gp_res_deg["peeling_ord"]

#check where does the peeling differ
@show findall(x -> x == false, gp_res_samesi["peeling_ord"] .== gp_res_deg["peeling_ord"])