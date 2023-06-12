#************************************************************************
# DRO
# El. pric: DRO with linear depedence between DA and B price
# Wind power: Chance constraint
#
#
# Power-to-X Modelling and Optimization 
# Three States Model with Storage 
#************************************************************************
# Packages
using JuMP
using Gurobi
using Mosek
using MosekTools
using DataFrames
using Pajarito
using Statistics

#************************************************************************
# dependencies
include("dependence-1stage-input.jl")

## Optimal for sharp 
theta = 25
rho = 0 # theta^CVaR
epsilon = 0.5
eta = 1.5

t1 = 100
t2 = length(TT_daily)

k = 0
for block=t1:t2
    k = k + 1
    T   = TT_daily[block]
    oa_solver = optimizer_with_attributes(Gurobi.Optimizer, 
                    MOI.Silent()=>true,
                    "FeasibilityTol"=>1e-3,
                    "MIPGap"=>1e-2)

    conic_solver = optimizer_with_attributes(Mosek.Optimizer,
                        MOI.Silent() => true,
                        "INTPNT_CO_TOL_REL_GAP"=>1.0e-3)
    opt = optimizer_with_attributes(Pajarito.Optimizer, 
                    "oa_solver"=>oa_solver,
                    "conic_solver"=>conic_solver,
                    "time_limit"=>120)

    m3s = Model(opt)

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
    beta = Array{Any,1}(undef, length(T))
    Psi = Array{Any,1}(undef, length(T))
    sigma = Array{Any,1}(undef, length(T))
    gamma = Array{Any,1}(undef, length(T))
    for t in 1:length(T)
        tt = T[t]
        psi[t]          = @variable(m3s, lower_bound=0) 
        normVarPsi[t]   = @variable(m3s, [1:length_error[tt],1:2], lower_bound=0) # aux var
        Psi[t]          = @variable(m3s, [1:2,1:2], PSD) # aux var
        sigma[t]        = @variable(m3s, [1:length_error[tt]])
        gamma[t]        = @variable(m3s, [1:2,1:length_error[tt]])
        beta[t]         = @variable(m3s, [1:length_error[tt]], lower_bound=0)
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

    # frobenius inner product between two matrices
    function frobeniusInner(A,B)
        return sum( A .* B)
    end

    ## objective function
    @objective(m3s, Min, 
        -sum(lambda_DA[t]*p_DA[t] - lambda_TSO*p_DAn[t]
                + lambda_B[t]*p_B[t]  - lambda_TSO*p_Bn[t]
                + lambda_H*d[t]
                - lambda_start*z_start[t]
            for t=T) 
            + sum(psi[t] * theta for t=1:length(T)) 
            + sum( mean(sigma[t]) for t=1:length(T) )
            + sum( frobeniusInner(Psi[t], eta.*cov_DA_B[T[t]]) for t=1:length(T)) )
            

            
    #************************************************************************
    # F function
    function F(t,i)
        return [Psi[t] (-1/2*[p_DA[T[t]], p_B[T[t]]]+1/2*gamma[t][:,i]-Psi[t]*mean_P[T[t]]');
                (-1/2*[p_DA[T[t]], p_B[T[t]]]+1/2*gamma[t][:,i]-Psi[t]*mean_P[T[t]]')' sigma[t][i]-(gamma[t][:,i]'*[error_DA[T[t]][i] error_B[T[t]][i]]')[1]+(mean_P[T[t]]*Psi[t]*mean_P[T[t]]')[1]]
    end

    M = [1000        0   0;
            0     1000   0;
            0        0  -1] # ellipsoidal support

    ## Price DRO
    for t in 1:length(T)    
        tt = T[t]
        # Wasserstein budget
        @constraint(m3s, [i=1:length_error[tt]], F(t,i) >= 0, PSDCone())

        # Wasserstein radius limit 
        @constraint(m3s, [i=1:length_error[tt]], sum(normVarPsi[t][i,:]) <= psi[t])
        @constraint(m3s, [i=1:length_error[tt]], gamma[t][:,i] .>= -normVarPsi[t][i,:])
        @constraint(m3s, [i=1:length_error[tt]], gamma[t][:,i] .<=  normVarPsi[t][i,:])
    end

    #****************************   DRO chance constraint (on wind)    *************************************
    for t in 1:length(T)   
        tt = T[t]
        @constraint(m3s, 
            tau[t] + (1/epsilon)*(psiCVaR[t]*rho + (1/length_error[tt])*sum(sigmaCVaR[t])) <= 0 )
        
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
    @constraint(m3s, [t=T],    -p_E[t]-p_C[t]  <= p_DA[t])
    @constraint(m3s, [t=T],    -p_w[t]  <= p_DA[t])
    @constraint(m3s, [t=T],    C_W >= p_DA[t])

    # energy in must be equal to energy out
    @constraint(m3s, [t=T],    p_w[t] - p_DA[t] - p_B[t] - p_E[t] - p_C[t] == 0)

    # positive and negative parts of DA and balance market
    @constraint(m3s, [t=T],    p_DA[t] == p_DAp[t] - p_DAn[t])
    @constraint(m3s, [t=T],    p_B[t] == p_Bp[t] - p_Bn[t])

    #******************* Deterministic constraints (from original problem) *************************
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

    if termination_status(m3s) == MOI.OPTIMAL || termination_status(m3s) == MOI.LOCALLY_SOLVED
        df_results[block, :obj]          = -objective_value(m3s)
        df_results[block, :obj_passive]  = innerRealisation_passive([value.(p_DA)[t] for t=T], [value.(p_E)[t] for t=T], [value.(p_C)[t] for t=T], T)
        df_results[block, :obj_active]   = innerRealisation_active([value.(p_DA)[t] for t=T], T)
    else
        df_results[block, :obj]             = -9999
        df_results[block, :obj_passive]     = innerRealisation_passive(zeros(length(T)), zeros(length(T)), zeros(length(T)), T)
        df_results[block, :obj_active]      = innerRealisation_active(zeros(length(T)), T)
    end

    println("Block: ", block, " - ")
    println("Objective value estimate: ", df_results[block, :obj])
    println("Objective value actual: ",   df_results[block, :obj_active])
end

CSV.write("Data/Output/DRO-cor.csv", df_results)


sum(df_results[100:end, :obj_active])
sum(df_results[100:end, :obj_passive])
std(df_results[100:end, :obj_active])
std(df_results[100:end, :obj_passive])


