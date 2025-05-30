include("header.jl")
# -----------------------------------------------------------------------------
flowtol = 1e-6
function run_expts(method::Int64, obj::Int64, datafolder::String)

    #methods = 1: PeelMax 
    #2: Greedy
    #3: PeelZero
    #4: DegPeel
    #5: Project+PeelMax
    #6: Project+Greedy
    #7: Project+PeelZero
    #8: Project+DegPeel
    #9: ILP
    #10: Project+ILP
    #11: Flow
    #12: ProjFlow


    #obj = 1: standard(all_or_nothing)
    #2: all_but_one 
    #3: atleast_half
    #4: scaled_power_reward(power = 1)
    #5: scaled_power_reward(power = 2)
    #6: power_reward(power = 0.5)

    large_datasets = ["contact-high-school", "contact-primary-school", "house-committees", "senate-committees"]

    large_trivago = ["trivago_379_lcc", "trivago_125_lcc", "trivago_860_lcc", "trivago_119_lcc", "trivago_284_lcc", "trivago_121_lcc",
        "trivago_651_lcc", "trivago_1030_lcc", "trivago_614_lcc", "trivago_1365_lcc"]

    dataname_list = Vector{String}()
    mean_Horder_list = Vector{Float64}()
    datasize_list = Vector{Tuple{Int64,Int64}}()

    runtime_list = Vector{Float64}()
    solutions_list = Vector{Vector{Int64}}()
    obj_values_list = Vector{Float64}()
    einS_list = Vector{Vector{Int64}}()
    etouchS_list = Vector{Vector{Int64}}()


    flowtol = 1e-8
    obj_name = ""
    method_name = ""

    if obj == 1
        obj_name = "standard"
    elseif obj == 2
        obj_name = "all_but_one"
    elseif obj == 3
        obj_name = "atleast_half"
    elseif obj == 4
        obj_name = "scaled_power1"
    elseif obj == 5
        obj_name = "scaled_power2"
    elseif obj == 6
        obj_name = "power_0pt5"
    elseif obj == 7
        obj_name = "atleast_two"
    end



    if method == 1
        method_name = "PeelMax"
    elseif method == 2
        method_name = "Greedy"
    elseif method == 3
        method_name = "PeelZero"
    elseif method == 4
        method_name = "DegPeel"
    elseif method == 5
        method_name = "PeelMaxProj"
    elseif method == 6
        method_name = "GreedyProj"
    elseif method == 7
        method_name = "PeelZeroProj"
    elseif method == 8
        method_name = "DegPeelProj"
    elseif method == 9
        method_name = "ILP"
    elseif method == 10
        method_name = "ProjILP"
    elseif method == 11
        method_name = "Flow"
    elseif method == 12
        method_name = "ProjFlow"
    end




    for dataset in ["trivago_125_lcc"]

        println("dataset = ", dataset)
        push!(dataname_list, dataset)

        #data = matread("$datafolder/large-datasets/$dataset.mat") #use for large datasets
        data = matread("$datafolder/small_benchmarks/trivago-small/$dataset.mat")

        H = sparse(Float64.(data["H"]))
        Ht = sparse(H')
        edge_list = incidence2elist(H)

        vertex2edges = build_vertex_to_edges(edge_list)
        push!(mean_Horder_list, mean(H_order(Ht)))

        order = vec(round.(Int64, sum(H, dims=2))) #size of hyperedges
        L = round(Int64, sum(H))
        M = size(H, 1)
        n = size(H, 2)
        push!(datasize_list, (M, n))
        # println("N = ", N)
        eweights = ones(M)

        reward_d = Vector{Vector{Float64}}() #list of reward vectors for each edge
        s_d = Vector{Vector{Float64}}() # list of s_e for each edge
        issup_list = Vector{Int64}() #list of indicators of supermodularity for each edge

        if obj == 1
            reward_d, s_d, issup_list = standard_reward(order)
        elseif obj == 2
            reward_d, s_d, issup_list = allbutone_reward(order)
        elseif obj == 3
            reward_d, s_d, issup_list = atleast_half_reward(order)
        elseif obj == 4
            reward_d, s_d, issup_list = scaled_power_reward(order, 1.0)
        elseif obj == 5
            reward_d, s_d, issup_list = scaled_power_reward(order, 2.0)
        elseif obj == 6
            reward_d, s_d, issup_list = power_reward(order, 0.5)
        elseif obj == 7
            reward_d, s_d, issup_list = atleast_two_reward(order)
        end


        if method == 9
            inc_reward_d = cumulative_to_incremental(reward_d)
            eS, alpS, time_req = genden_ILP_full(edge_list, inc_reward_d, order, eweights, n, false)
            S_opt = findall(x -> x == 1, eS)
            _, etouchS_edges = etouchS(H, S_opt)
            _, einS_edges = einS(H, order, S_opt)

            push!(solutions_list, S_opt)
            push!(obj_values_list, alpS)
            push!(einS_list, einS_edges)
            push!(etouchS_list, etouchS_edges)
            push!(runtime_list, time_req)
            println("ILP obj val is ", obj_values_list)
            println("ILP runtime is ", runtime_list)

        elseif method == 1
            # gp_res = peeling_with_se(H,reward_d,s_d)
            gp_res = peeling_with_se_edge_list(edge_list, reward_d, s_d, vertex2edges)
            optsol = gp_res["optsol"]
            time_req = gp_res["total_dt"]
            _, etouchS_edges = etouchS(H, optsol)
            _, einS_edges = einS(H, order, optsol)
            push!(solutions_list, optsol)
            push!(obj_values_list, gp_res["optval"])
            push!(einS_list, einS_edges)
            push!(etouchS_list, etouchS_edges)
            push!(runtime_list, time_req)
            println("Peel max obj is", obj_values_list)
            println("Peel max runtime is", runtime_list)

        elseif method == 2
            # gp_res = peeling_with_se(H,reward_d,reward_d)
            gp_res = peeling_with_se_edge_list(edge_list, reward_d, reward_d, vertex2edges)
            optsol = gp_res["optsol"]
            time_req = gp_res["total_dt"]
            _, etouchS_edges = etouchS(H, optsol)
            _, einS_edges = einS(H, order, optsol)
            push!(solutions_list, optsol)
            push!(obj_values_list, gp_res["optval"])
            push!(einS_list, einS_edges)
            push!(etouchS_list, etouchS_edges)
            push!(runtime_list, time_req)
            println("Greedy obj val is ", obj_values_list)
            println("Greedy runtime is ", runtime_list)

        elseif method == 3
            s_d0 = all_zero_si(order)
            # gp_res = peeling_with_se(H,reward_d,s_d0)
            gp_res = peeling_with_se_edge_list(edge_list, reward_d, s_d0, vertex2edges)
            optsol = gp_res["optsol"]
            time_req = gp_res["total_dt"]
            _, etouchS_edges = etouchS(H, optsol)
            _, einS_edges = einS(H, order, optsol)
            push!(solutions_list, optsol)
            push!(obj_values_list, gp_res["optval"])
            push!(einS_list, einS_edges)
            push!(etouchS_list, etouchS_edges)
            push!(runtime_list, time_req)
            println("PeelZero obj val is", obj_values_list)
            println("PeelZero runtime is", runtime_list)

        elseif method == 4
            # gp_res = peeling_by_degree(H, Ht,reward_d)
            gp_res = peeling_by_degree_edge_list(edge_list, reward_d, vertex2edges)
            optsol = gp_res["optsol"]
            time_req = gp_res["total_dt"]
            _, etouchS_edges = etouchS(H, optsol)
            _, einS_edges = einS(H, order, optsol)
            push!(solutions_list, optsol)
            push!(obj_values_list, gp_res["optval"])
            push!(einS_list, einS_edges)
            push!(etouchS_list, etouchS_edges)
            push!(runtime_list, time_req)
            println("DegPeel obj vals is ", obj_values_list)
            println("DegPeel runtime is ", runtime_list)

        elseif method == 11
            S_flow_opt_set, time_taken = densest(edge_list, reward_d)
            println("Length of optsol is ,$(length(S_flow_opt_set))")
            S_flow_opt = collect(S_flow_opt_set)
            _, etouchS_edges = etouchS(H, S_flow_opt)
            _, einS_edges = einS(H, order, S_flow_opt)
            push!(solutions_list, S_flow_opt)
            push!(obj_values_list, hedensity_non_uniform(H, S_flow_opt, reward_d))
            println("Flow Obj_val_list is", obj_values_list)
            push!(einS_list, einS_edges)
            push!(etouchS_list, etouchS_edges)
            push!(runtime_list, time_taken)
            println("Flow Runtime is ", runtime_list)

        elseif method == 10 && 0 in issup_list
            # method_name = "ILP_proj"
            reward_d_proj = project_to_convex(reward_d, issup_list)
            inc_reward_d_proj = cumulative_to_incremental(reward_d_proj)
            eS_proj, alpS_proj, time_req = genden_ILP_full(edge_list, inc_reward_d_proj, order, eweights, n, false)
            S_opt_proj = findall(x -> x == 1, eS_proj)
            _, etouchS_edges = etouchS(H, S_opt_proj)
            _, einS_edges = einS(H, order, S_opt_proj)
            push!(solutions_list, S_opt_proj)
            push!(obj_values_list, hedensity_non_uniform(H, S_opt_proj, reward_d))
            push!(einS_list, einS_edges)
            push!(etouchS_list, etouchS_edges)
            push!(runtime_list, time_req)
            println("ProjILP obj is ", obj_values_list)
            println("ProjILP runtime is ", runtime_list)

        elseif method == 12 && 0 in issup_list

            reward_d_proj = project_to_convex(reward_d, issup_list)
            S_flow_proj_set, time_taken = densest(edge_list, reward_d_proj)
            println("Length of solution is ", length(S_flow_proj_set))
            S_flow_proj = collect(S_flow_proj_set)
            _, etouchS_edges = etouchS(H, S_flow_proj)
            _, einS_edges = einS(H, order, S_flow_proj)
            push!(solutions_list, S_flow_proj)
            push!(obj_values_list, hedensity_non_uniform(H, S_flow_proj, reward_d))
            println("ProjFlow Obj_list is", obj_values_list)
            push!(einS_list, einS_edges)
            push!(etouchS_list, etouchS_edges)
            push!(runtime_list, time_taken)
            println("ProjFlow Runtime is ", runtime_list)

        elseif method == 5 && 0 in issup_list
            reward_d_proj = project_to_convex(reward_d, issup_list)
            # gp_res = peeling_with_se(H,reward_d_proj,s_d)
            s_d_proj = Vector{Vector{Float64}}()
            for i in 1:length(reward_d_proj)
                push!(s_d_proj, maximal_si(reward_d_proj[i]))
            end
            gp_res = peeling_with_se_edge_list(edge_list, reward_d_proj, s_d_proj, vertex2edges)
            optsol = gp_res["optsol"]
            time_req = gp_res["total_dt"]
            _, etouchS_edges = etouchS(H, optsol)
            _, einS_edges = einS(H, order, optsol)
            push!(solutions_list, optsol)
            push!(obj_values_list, hedensity_non_uniform(H, optsol, reward_d))
            push!(einS_list, einS_edges)
            push!(etouchS_list, etouchS_edges)
            push!(runtime_list, time_req)
            println("Project+PeelMax obj is ", obj_values_list)
            println("Project+PeelMax runtime is ", runtime_list)

        elseif method == 6 && 0 in issup_list
            reward_d_proj = project_to_convex(reward_d, issup_list)
            # gp_res = peeling_with_se(H,reward_d_proj,reward_d_proj)
            gp_res = peeling_with_se_edge_list(edge_list, reward_d_proj, reward_d_proj, vertex2edges)
            optsol = gp_res["optsol"]
            time_req = gp_res["total_dt"]
            _, etouchS_edges = etouchS(H, optsol)
            _, einS_edges = einS(H, order, optsol)
            push!(solutions_list, optsol)
            push!(obj_values_list, hedensity_non_uniform(H, optsol, reward_d))
            push!(einS_list, einS_edges)
            push!(etouchS_list, etouchS_edges)
            push!(runtime_list, time_req)
            println("Project+Greedy obj is ", obj_values_list)
            println("Project+Greedy runtime is", runtime_list)


        elseif method == 7 && 0 in issup_list
            reward_d_proj = project_to_convex(reward_d, issup_list)
            s_d0 = all_zero_si(order)
            # gp_res = peeling_with_se(H,reward_d_proj,s_d0)
            gp_res = peeling_with_se_edge_list(edge_list, reward_d_proj, s_d0, vertex2edges)
            optsol = gp_res["optsol"]
            time_req = gp_res["total_dt"]
            _, etouchS_edges = etouchS(H, optsol)
            _, einS_edges = einS(H, order, optsol)
            push!(solutions_list, optsol)
            push!(obj_values_list, hedensity_non_uniform(H, optsol, reward_d))
            push!(einS_list, einS_edges)
            push!(etouchS_list, etouchS_edges)
            push!(runtime_list, time_req)
            println("Project+PeelZero obj is ", obj_values_list)
            println("Project+PeelZero runtime is ", runtime_list)

        elseif method == 8 && 0 in issup_list
            # method_name = "greedy_deg_proj"
            reward_d_proj = project_to_convex(reward_d, issup_list)
            # gp_res = peeling_by_degree(H, Ht,reward_d_proj)
            gp_res = peeling_by_degree_edge_list(edge_list, reward_d_proj, vertex2edges)
            optsol = gp_res["optsol"]
            time_req = gp_res["total_dt"]
            _, etouchS_edges = etouchS(H, optsol)
            _, einS_edges = einS(H, order, optsol)
            push!(solutions_list, optsol)
            push!(obj_values_list, hedensity_non_uniform(H, optsol, reward_d))
            push!(einS_list, einS_edges)
            push!(etouchS_list, etouchS_edges)
            push!(runtime_list, time_req)
            println("Project+DegPeel obj is ", obj_values_list)
            println("Project+DegPeel runtime is ", runtime_list)
        else
            println("Method-obj combination  not found")
            println("Given method name is ", method_name)
            println("Input objective is ", obj_name)
            println("Is there a 0 in issup_list", (0 in issup_list))
        end


    end
    #collect final result for all datasets and store in a json file with method and obj name in the filename

    #create a final dictionary of results
    final_res = Dict(
        "dataname_list" => dataname_list,
        "mean_Horder_list" => mean_Horder_list,
        "datasize_list" => datasize_list,
        "solutions_list" => solutions_list,
        "obj_values_list" => obj_values_list,
        "einS_list" => einS_list,
        "etouchS_list" => etouchS_list,
        "runtime_list" => runtime_list,
    )

    #example save the final results in a json file
    fs = open("results/$obj_name-$method_name-Triv.json", "w")
    JSON.print(fs, final_res)
    close(fs)

end

datafolder = "datafolder"

#Example usage of runing experiments
run_expts(1, 1, datafolder)
run_expts(1, 2, datafolder)
run_expts(1, 3, datafolder)
run_expts(1, 4, datafolder)
run_expts(1, 5, datafolder)
run_expts(1, 7, datafolder)
run_expts(1, 6, datafolder)

# run_expts(2,1,datafolder)
# run_expts(2,2,datafolder)
# run_expts(2,3,datafolder)
# run_expts(2,5,datafolder)
# run_expts(2,6,datafolder)
# run_expts(2,7,datafolder)



