using Gurobi
using SparseArrays
using LinearAlgebra
using JuMP
# using MathOptInterface
# const MOI = MathOptInterface
const gurobi_env = Gurobi.Env()
# Set OutputFlag to 0 (suppress all Gurobi output)



"""
For a given value alp and a hypergraph H, this
checks whether it it possible to find a subhypergraph
with (generalized) densest subhypergraph value
strictly greater than alp.

    EdgeList is the edge list for the hypergraph: EdgeList[i] gives node indices for ith edge

    reward_d: vector of density rewards. reward_d[e][i] is the (extra) reward for including i nodes of 
        hyperedge e in output set S, in addition to the rewards already received for including j < i nodes (they sum).
        Thus, the total reward for including i nodes of e is r_i = sum(reward_d[e][j] for i = 1:j)

    order: hyperedge size vector. order[e] = number of nodes in e

    eweights: hyperedge weights. eweights[e] is a positive weight for scaling rewards at hyperedge e

    n: number of nodes in the hypergraph

    outputflag: whether or not to display the Gurobi output

"""
function genden_decision(alp::Float64,EdgeList::Vector{Vector{Int64}},reward_d::Vector{Vector{Float64}},order::Vector{Int64},eweights::Vector{Float64},n::Int64,pair2lin::Dict,outflag::Bool = false)
   
    # Size of hypergraph; equal number of y variables in the ILP
    L = sum(order)

    # Set up Gurobi model
    m = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(gurobi_env), "OutputFlag" => outflag))

    set_optimizer_attribute(m, "OutputFlag", 0)

      # Set strict tolerances to ensure opt - 0.0 <= 1e-10
    # set_optimizer_attribute(m, "MIPGap", 1e-10)        # Ensure very precise optimality gap
    # set_optimizer_attribute(m, "IntFeasTol", 1e-9)    # Ensure integer feasibility within 1e-10
    # set_optimizer_attribute(m, "FeasibilityTol", 1e-9) # Ensure constraints are satisfied within 1e-10
    # set_optimizer_attribute(m, "OptimalityTol", 1e-9)  # Ensure solver stops only if it's optimal within 1e-10


    @variable(m, y[1:L],Bin)
    @variable(m, x[1:n],Bin)

    M = length(EdgeList)

    # maximize  \left( \sum_{e \in E} c_e  \sum_{i = 1}^{|e|} \delta_{e,i} \cdot y_{e,i} \right) + \alpha \sum_{v \in V} x_v
    @objective(m, Max, -alp*sum(x[i] for i = 1:n) + sum( eweights[e]*sum(reward_d[e][i+1]*y[pair2lin[(e,i)]] for i = 1:order[e])  for e=1:M))

    for e = 1:M
        edge = EdgeList[e]
        for i = 1:order[e]
            # y_{e,i} = y_l \leq 1/i \sum_{v \in e} x_v
            # This ensures that y_{e,i} = 1 if and only if |e \cap S| \geq i (otherwise it's zero)
            # Does so by putting a bound y_{e,i} < 1 if |e \cap S| < i and y_{e,i} >= 1 if |e \cap S| >= i
            # The objective encourages y_{e,i} to be at large as possible
            l = pair2lin[(e,i)]
            @constraint(m, y[l] <= 1/i*sum(x[v] for v in edge))
        end
    end

    JuMP.optimize!(m)
    X = JuMP.value.(x)
    Y = JuMP.value.(y)
    OPT = JuMP.objective_value(m)

    return X, Y, OPT
end

"""
    Compute the objective value for the genearlized densest subhypergraph problem
        for a set S. 

    See the function genden_decision for an explanation of the inputs

"""
function check_gdsh_objective(eS::Vector{Int64},EdgeList::Vector{Vector{Int64}},reward_d::Vector{Vector{Float64}},order::Vector{Int64},eweights::Vector{Float64},n::Int64)
    if length(eS) < n
        S = eS
        Snum = length(S)
        eS = zeros(Int64,n)
        eS[S] .= 1
    else
        Snum = sum(eS)
    end
    if Snum == 0
        return 0
    end
    M = length(EdgeList)
    reward = 0
    for e = 1:M
        edge = EdgeList[e]

        # se is the number of nodes from edge e that are in the set S
        se = round(Int64,sum(eS[edge]))
        reward += eweights[e]*sum(reward_d[e][2:se+1]) #Added 2 to the index to account for the fact that the first element of reward_d is for 0 nodes
    end

    return reward/Snum
end


"""
Repeatedly solve the decision problem in order to get the optimal solution to 
the Generalized Densest Subhypergraph Problem with general weights

We assume that the entries of reward_d are all nonnegative, or else this doesn't work.
"""

function genden_ILP_full(EdgeList::Vector{Vector{Int64}},reward_d::Vector{Vector{Float64}},order::Vector{Int64},eweights::Vector{Float64},n::Int64,outflag::Bool = false)

    M = length(EdgeList)

    # Start with the density of the entire hypergraph
    eS = ones(Int64,n)
    alp = check_gdsh_objective(eS,EdgeList,reward_d,order,eweights,n)
    Sbest = eS
    alpbest = alp
    
    iter = 1
    println("Full hypergraph density is $alp")
    
    ## Set up dictionary for subproblems 
    pair2lin = Dict() # map from pair (edge, i) to linear index indicating the (edge,number) pair
    lin2pair = Vector{Tuple{Int64,Int64}}() # map in the other direction
    next = 0
    for e = 1:M
        re = length(EdgeList[e])
        for i = 1:re
            next += 1
            pair2lin[(e,i)] = next  # y_{e,i} = y_{next} = y_{pair2lin[(e,i)]}, for i = 1,2, ... , |e| = re
            push!(lin2pair,(e,i))
        end
    end

    # Search for better and better sets until you can't find anymore
    
    while true
        print("Running step $iter: ")
        X,Y,OPT = genden_decision(alp,EdgeList,reward_d,order,eweights,n,pair2lin,outflag)
        X = round.(Int64,X)
        # print("Shape of X is ",size(X))
        #TRY ROUNDING Y
        Y = round.(Int64,Y)
        # print("Shape of Y is ",size(Y))
        
        #CHECK IF CONSTRAINTS ARE SATISFIED OR NOT NOW THAT Y IS ROUNDED
        # for e = 1:M
        #     edge = EdgeList[e]
        #     for i = 1:order[e]
        #         l = pair2lin[(e,i)]
        #         # println("i is $i")
        #         # println("Y[l] = ",Y[l])
        #         # println("sum(X[v] for v in edge) = ",sum(X[v] for v in edge))
        #         # println("1/i*sum(X[v] for v in edge) = ",1/i*sum(X[v] for v in edge))
        #         if Y[l] > 1/i*sum(X[v] for v in edge)
        #             println("Constraint not satisfied for edge $e and i = $i")
        #             println("Y[l] = ",Y[l])
        #             println("1/i*sum(X[v] for v in edge) = ",1/i*sum(X[v] for v in edge))
        #         end
        #         @assert(Y[l] <= 1/i*sum(X[v] for v in edge))
        #     end
        # end

        alpnew = check_gdsh_objective(X,EdgeList,reward_d,order,eweights,n)

        if alpnew > alp
            r2 = sum(eweights[e]*sum(reward_d[e][i+1]*Y[pair2lin[(e,i)]] for i = 1:order[e])  for e=1:M) #added 1 to the index to account for the fact that the first element of reward_d is for 0 nodes
            if round(r2/sum(X),digits = 5) != round(alpnew,digits = 5)
                println("In the r2/sum(X) != alpnew case")
                println("r2/sum(X) = ",r2/sum(X))
                println("alpnew = ",alpnew)

            end

            if r2/sum(X) != alpnew
                println("The objective value is not equal to the density of the set")
                println("r2/sum(X) = ",r2/sum(X))
                println("alpnew = ",alpnew)
                println(" r2/sum(X) - alpnew = ",r2/sum(X) - alpnew)

            end
            # @assert(round(abs(r2/sum(X)-alpnew),digits = 5) < 1e-4)
            # @assert (r2/sum(X) == alpnew)
            @assert(abs(r2/sum(X) - alpnew) < 1e-10)

            # if you found a better objective, keep it
            Sbest = X
            alpbest = alpnew
            alp = alpnew
            nums = sum(X)
            println(" found subset with $nums nodes and density $alpnew")
            iter += 1
        else
            # if not, you're done, stop
            if (round(norm(OPT - 0.0),digits = 8) > 1e-6)
                println(" found no improvement, but the objective value is not zero")
                println("OPT = $OPT")
                println("alp = $alp")
                println("alpnew = $alpnew")
                println("The norm thing is",norm(OPT - 0.0))
                println("The round thing is",round(norm(OPT - 0.0),digits = 8))
                println("Using abs gives",abs(OPT))
                # break
            end
            
            if (OPT - 0.0 >= 1e-10)
                println(" found no improvement, but the objective value is not zero")
                println("OPT = $OPT")
                println("alp = $alp")
                println("alpnew = $alpnew")
                println("The norm thing is",norm(OPT - 0.0))
                println("The round thing is",round(norm(OPT - 0.0),digits = 8))
                println("Using abs gives",abs(OPT))
                
            end
            # @assert(round(norm(OPT - 0.0),digits = 5) <= 1e-4)

            @assert(OPT-0.0 < 1e-10)
           
            println(" found no improvement")
            break
        end
    end

    return Sbest, alpbest

end

# """
# Create a reward function for the standard all-or-nothing
#     densest subhypergraph problem.
# """


# """
# Create a reward function for GDSH where
#     you get 1 point if you have k or k-1
#     nodes from a hyperedge of size k
# """
