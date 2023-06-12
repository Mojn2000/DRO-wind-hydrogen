#************************************************************************
# DRO
# El. pric: "Normal" DRO 
# Wind power: Chance constraint
#
# Uses convex trick for bin variables
#
# Power-to-X Modelling and Optimization 
# Three States Model with Storage 
#************************************************************************
# Packages
using JuMP
using Gurobi
using DataFrames
using Statistics


## Optimal for return
theta = 20
rho = 0
epsilon = 0.5

#param_list = [0 30 60]
#param_list = [0.0 0.2 0.4 0.8 1.6]
#param_list = [0.01 0.05 0.15 0.3 0.5]
param_list = [30]
#param_list = [0 5 10 15 20 25 30 40 60 80 100 200 400]
#param_list = [0 10 20 30 40]
#param_list = [1, 1000]
#param_list = [1, 60, 100, 200, 400, 1000]
#param_list = [0.5]

# dependencies
include("../innerRealization.jl")
include("input-1stage.jl")


# init data frames
blocks = 1:length(TT_daily)
df_results = DataFrame()
df_results[!,:block]        = blocks
df_results[!,:obj]          = blocks .* 0.0
df_results[!,:obj_passive]  = blocks .* 0.0
df_results[!,:obj_active]   = blocks .* 0.0
suprise = round.(blocks .* 0.0)

it = 0
while it < length(param_list)
    it = it + 1
    theta = param_list[it]

    # print hyper parameters
    println("param_list: ", param_list)
    println("Iteration: ", it)
    println("theta: ", theta)
    println("rho: ", rho)
    println("epsilon: ", epsilon)

    ## in sample T=721 to T=4368  2022-06-12 00:00:00
    t1 = 42*3+1
    #t2 = 182
    #t1 = 183
    t2 = length(TT_daily)
    #t1 = 223
    #t2 = 223
    #t1 = 91 
    #t2 = 546
    
    ## out of sample
    #t1 = 547
    #t2 = length(TT_daily)
    #block = 657
    #block = 500

    mean_dro = []

    for block=t1:t2 

        # print parameters
        println("epsilon: ", epsilon)
        println("rho: ", rho)
        println("theta: ", theta)

        T   = TT_daily[block]
        #m3s = Model(CPLEX.Optimizer)
        m3s = Model(Gurobi.Optimizer)
        #m3s = Model(Mosek.Optimizer)
        set_silent(m3s)
        #************************************************************************
        # Variables
        @variables m3s begin
            p_w[T] >= 0     # wind power (possibly curtailed)

            p_DA[T]         # electrictiy from day ahead market
            p_DAp[T] >= 0   # electrictiy from day ahead market (positive component)
            p_DAn[T] >= 0   # electrictiy from day ahead market (negative component)
            
            p_B[T]          # electrictiy from balance market
            p_Bp[T] >= 0    # electrictiy from balance market (positive component)
            p_Bn[T] >= 0    # electrictiy from balance market (negative component)
            
            p_E[T]  >= 0    # electricity used for electrolyzer
            p_C[T]  >= 0    # electricity used for compressor
            
            e[T]      >= 0 # electrolyzer consumption for each segment
            h[T]        >= 0 # hydrogen production
            h_d[T]      >= 0 # hydrogen production directly to demand
            d[T]        >= 0 # hydrogen sold
            s_in[T]     >= 0 # hydrogen stored
            s_out[T]    >= 0 # hydrogen used from storage
            soc[T]      >= 0 # state of charge of storage (kg)
            #z_h[T,S],   Bin  # specific hydrogen production 
            z_on[T],    Bin  # on electrolyzer
            #z_off[T],   Bin  # off electrolyzer
            z_sb[T],    Bin  # standby electrolyzer
            z_start[T]  >= 0  # start electrolyzer 
        end

        ## DRO variables (pricing)
        psi = Array{Any,1}(undef, length(T))
        normVarPsi = Array{Any,1}(undef, length(T))
        sigma = Array{Any,1}(undef, length(T))
        gamma = Array{Any,1}(undef, length(T))
        for t in 1:length(T)
            tt = T[t]
            psi[t]          = @variable(m3s, lower_bound=0)
            normVarPsi[t]   = @variable(m3s, [1:length_error[tt],1:2], lower_bound=0) # aux var

            sigma[t]    = @variable(m3s, [1:length_error[tt]])
            gamma[t]    = @variable(m3s, [1:4,1:length_error[tt]], lower_bound=0)
        end

        ## DRO variables (wind power)
        tau = Array{Any,1}(undef, length(T))
        psiCVaR = Array{Any,1}(undef, length(T))
        sigmaCVaR = Array{Any,1}(undef, length(T))
        gammaW1 = Array{Any,1}(undef, length(T))
        gammaW2 = Array{Any,1}(undef, length(T))
        for t in 1:length(T)
            tt = T[t]
            tau[t] = @variable(m3s) 
            psiCVaR[t] = @variable(m3s, lower_bound=0)
            sigmaCVaR[t] = @variable(m3s, [1:length_error[tt]])

            gammaW1[t] = @variable(m3s, [1:2,1:length_error[tt]], lower_bound=0)
            gammaW2[t] = @variable(m3s, [1:2,1:length_error[tt]], lower_bound=0)
        end


        ## objective function
        @objective(m3s, Min, 
            -sum((lambda_DA[t])*p_DA[t] - lambda_TSO*p_DAn[t]
                    + (lambda_B[t])*p_B[t]  - lambda_TSO*p_Bn[t]
                    + lambda_H*d[t]
                    - lambda_start*z_start[t]
                for t=T) 
                + sum(psi[t]*theta + 1/length_error[T[t]]*sum(sigma[t]) for t=1:length(T)) )



        #************************************************************************
        ## Price DRO
        for t in 1:length(T)    
            tt = T[t]
            # Wasserstein budget
            @constraint(m3s, [i=1:length_error[tt]],
                -transpose([p_DA[tt], p_B[tt]])*[error_DA[tt][i], error_B[tt][i]]
                + transpose(gamma[t][:,i])*(h_rhs[t]  - Q[t]*[error_DA[tt][i], error_B[tt][i]]) <= sigma[t][i])
                #h_rhs[tt*4-3:tt*4]
            # Wasserstein radius limit 
            #@constraint(m3s, [i=1:length_error[tt]], transpose(Q[t])*gamma[t][:,i] + [p_DA[tt], p_B[tt]] .>= -psi[t])
            #@constraint(m3s, [i=1:length_error[tt]], transpose(Q[t])*gamma[t][:,i] + [p_DA[tt], p_B[tt]] .<=  psi[t])

            @constraint(m3s, [i=1:length_error[tt]], sum(normVarPsi[t][i,:]) <= psi[t])
            @constraint(m3s, [i=1:length_error[tt]], transpose(Q[t])*gamma[t][:,i] + [p_DA[tt], p_B[tt]] .>= -normVarPsi[t][i,:])
            @constraint(m3s, [i=1:length_error[tt]], transpose(Q[t])*gamma[t][:,i] + [p_DA[tt], p_B[tt]] .<=  normVarPsi[t][i,:])
            
            # dual feasibility
            @constraint(m3s, gamma[t] .>= 0)
        end

        #****************************   DRO chance constraint (on wind)    *************************************
        # wind power must be less than available
        #@constraint(m3s, [t=T], p_w[t] <= P_W[t])
        for t in 1:length(T)
            tt = T[t]
            @constraint(m3s, tau[t] + (1/epsilon)*(psiCVaR[t]*rho + (1/length_error[tt])*sum(sigmaCVaR[t])) <= 0 )
            
            @constraint(m3s, [i=1:length_error[tt]],
                -error_windPower[tt][i]-(P_W[tt] - p_w[tt]) - tau[t] 
                + gammaW1[t][:,i]'*([C_W-P_W[tt]; P_W[tt]] - [1; -1]*error_windPower[tt][i]) <= sigmaCVaR[t][i])
            
            @constraint(m3s, [i=1:length_error[tt]],
                gammaW2[t][:,i]'*([C_W-P_W[tt]; P_W[tt]] - [1; -1]*error_windPower[tt][i]) <= sigmaCVaR[t][i]) 

            @constraint(m3s, [i=1:length_error[tt]], [1; -1]'*gammaW1[t][:,i] + 1 .<=  psiCVaR[t])
            @constraint(m3s, [i=1:length_error[tt]], [1; -1]'*gammaW1[t][:,i] + 1 .>= -psiCVaR[t])
            @constraint(m3s, [i=1:length_error[tt]], [1; -1]'*gammaW2[t][:,i] .<=  psiCVaR[t])
            @constraint(m3s, [i=1:length_error[tt]], [1; -1]'*gammaW2[t][:,i] .>= -psiCVaR[t])
        end


        #**************************     aux constraints     ************************************
        # sanity caps on power sold in DA and balance market
        #@constraint(m3s, [t=T],   -10 <= p_B[t])
        #@constraint(m3s, [t=T],    10 >= p_B[t])
        #@constraint(m3s, [t=T],   -C_W <= p_DA[t])
        @constraint(m3s, [t=T],    -p_E[t]-p_C[t]  <= p_DA[t])
        @constraint(m3s, [t=T],    -p_w[t]  <= p_DA[t])
        @constraint(m3s, [t=T],    C_W >= p_DA[t])

        #@constraint(m3s, [t=T],    -p_E[t]-p_C[t]  <= p_B[t])

        # energy in must be equal to energy out
        @constraint(m3s, [t=T],    p_w[t] - p_DA[t] - p_B[t] - p_E[t] - p_C[t] == 0)

        # positive and negative parts of DA and balance market
        @constraint(m3s, [t=T],    p_DA[t] == p_DAp[t] - p_DAn[t])
        @constraint(m3s, [t=T],    p_B[t] == p_Bp[t] - p_Bn[t])

        #******************* Deterministic constraints (from original problem) *************************
        # Standby power from market
        #@constraint(m3s, [t=T],    p_DAn[t] <= P_sb * z_sb[t])
        # Total electricity consumption
        @constraint(m3s, [t=T],    p_E[t] == e[t] + P_sb * z_sb[t])
        # Hydrogen production
        @constraint(m3s, [t=T,s=S],    h[t] <= a[s]*e[t] + b[s]*z_on[t] )
        # Hydrogen balance
        @constraint(m3s, [t=T],    h[t] == h_d[t] + s_in[t])
        # Demand balance 
        @constraint(m3s, [t=T],    d[t] == h_d[t] + s_out[t])  
        # Maximum storage output
        @constraint(m3s, [t=T],    s_out[t] <= C_E * eta_full_load)
        # Hydrogen min demand (DAILY)
        @constraint(m3s, sum(d) >= C_D)
        # Maximum electrolyzer power
        @constraint(m3s, [t=T],    p_E[t] <= C_E * z_on[t] + P_sb * z_sb[t])
        # Minimum electrolyzer power
        @constraint(m3s, [t=T],    p_E[t] >= P_min * z_on[t] + P_sb * z_sb[t])
        # States
        @constraint(m3s, [t=T],    1 >= z_on[t] + z_sb[t])
        # Not from Off to Standby
        @constraint(m3s, [t=T[2:end]],    z_sb[t] <= z_on[t-1] + z_sb[t-1])
        # Startup cost
        @constraint(m3s, [t=T[2:end]],    z_start[t] >= z_on[t] - z_on[t-1] - z_sb[t-1])
        # Compressor consumption
        @constraint(m3s, [t=T],    p_C[t] == s_in[t] * P_C)
        # Max storage fill
        @constraint(m3s, [t=T],    soc[t] <= C_S)
        # SOC
        @constraint(m3s, [t=T],    soc[t] == (t > T[1] ? soc[t-1] : 0) - s_out[t] + s_in[t])
        #************************************************************************


        # Solve
        optimize!(m3s)

        df_results[block, :obj]          = -objective_value(m3s)
        df_results[block, :obj_passive]  = innerRealisation_passive([value.(p_DA)[t] for t=T], [value.(p_E)[t] for t=T], [value.(p_C)[t] for t=T], T)
        df_results[block, :obj_active]   = innerRealisation_active([value.(p_DA)[t] for t=T], T)
        
        suprise[block] = df_results[block, :obj_active]+objective_value(m3s)
        #println([round(maximum(error_DA[tt]./error_B[tt])) for tt in T])
        #println([round(minimum(error_DA[tt]./error_B[tt])) for tt in T])
        println("Block: ", block, " - ")
        println("Objective value estimate: ", -objective_value(m3s))
        println("Objective value actual: ", df_results[block, :obj_active])
        println([round(value.(p_DA)[t],digits=2) for t=T])
        println(suprise[block])
        println("##############################################")


        #append!(mean_dro, mean(df_results[df_results[:, :obj] .!= 0, :obj_active]))
    end
    CSV.write("Data/Output/1-stage/DRO/theta-8h-"*string(theta)*".csv", df_results)
end


sum(df_results[:,:obj_active] .!= 0)
sum(df_results[1:182, :obj_passive])
sum(df_results[1:182, :obj_active])
sum(df_results[183:end, :obj_passive])
sum(df_results[183:end, :obj_active])



## 0%, max BA
# θ = 22 train passive  => 478822.7
# θ = 22 train active   => 541565.0
# θ = 22 test passive   => 333373.0
# θ = 22 test active    => 353523.5
# θ =  0 train passive  => 460940.2
# θ =  0 train active   => 528806.7
# θ =  0 test passive   => 315958.5
# θ =  0 test active    => 334216.5

## 1%, max BA
# θ = 22 train passive  => 484985.3
# θ = 22 train active   => 546940.4
# θ = 22 test passive   => 334304.3
# θ = 22 test active    => 354187.5
# θ =  0 train passive  => 495822.6
# θ =  0 train active   => 561950.5
# θ =  0 test passive   => 319425.1
# θ =  0 test active    => 337334.4

## 2%, max BA
# θ = 22 train passive  => 493287.7
# θ = 22 train active   => 554415.0
# θ = 22 test passive   => 335610.3
# θ = 22 test active    => 355270.1
# θ = 0  train passive  => 506978.1
# θ = 0  train active   => 571925.2
# θ = 0  test passive   => 322213.5
# θ = 0  test active    => 339785.0

## 3%, max BA
# θ = 22 train passive  => 501180.0
# θ = 22 train active   => 561041.4
# θ = 22 test passive   => 337206.6
# θ = 22 test active    => 356573.0
# θ = 0  train passive  => 522381.6
# θ = 0  train active   => 585631.6
# θ = 0  test passive   => 325472.1
# θ = 0  test active    => 342638.6

## 4%, max BA
# θ = 22 train passive  => 518662.7
# θ = 22 train active   => 577193.2
# θ = 22 test passive   => 338230.6
# θ = 22 test active    => 357396.8
# θ = 0  train passive  => 534964.7
# θ = 0  train active   => 596500.6
# θ = 0  test passive   => 328121.8
# θ = 0  test active    => 344834.0

## 5%, max BA
# θ = 22 train passive  => 535079.4
# θ = 22 train active   => 592846.9
# θ = 22 test passive   => 339226.5
# θ = 22 test active    => 358247.5
# θ = 0  train passive  => 546971.2
# θ = 0  train active   => 607387.1
# θ = 0  test passive   => 331301.6
# θ = 0  test active    => 347571.5

## 6%, max BA
# θ = 22 train passive  => 547355.9
# θ = 22 train active   => 603115.2
# θ = 22 test passive   => 339831.0
# θ = 22 test active    => 358539.3
# θ = 0  train passive  => 569233.0
# θ = 0  train active   => 628105.9
# θ = 0  test passive   => 334470.1
# θ = 0  test active    => 350368.7

## 7%, max BA
# θ = 22 train passive  => 
# θ = 22 train active   => 
# θ = 22 test passive   => 
# θ = 22 test active    => 
# θ = 0  train passive  => 
# θ = 0  train active   => 
# θ = 0  test passive   => 
# θ = 0  test active    => 

###########################################################################################

## Deterministic 24h, 6 weeks of burn in
# train passive  => 479388.2
# train active   => 536004.2
# test passive   => 333,176.9
# test active    => 348,102.1

#
