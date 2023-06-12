#************************************************************************
# Fully Deterministic model
# 
#
#
# Power-to-X Modelling and Optimization 
# Three States Model with Storage 
#************************************************************************
# Packages
using JuMP
using Gurobi
using DataFrames
using CSV
using Statistics

#************************************************************************
# Model
include("../innerRealization.jl")
include("input-1stage.jl")

## prepare data DataFrame
df.p_DA = df.spotMeas*0
df.p_DAn= df.spotMeas*0
df.p_B  = df.spotMeas*0
df.p_Bn = df.spotMeas*0
df.p_E  = df.spotMeas*0
df.p_C  = df.spotMeas*0
df.d    = df.spotMeas*0
df.start= df.spotMeas*0

blocks = 1:length(TT_daily)
df_results = DataFrame()
df_results[!,:block]        = blocks
df_results[!,:obj]          = blocks .* 0.0
df_results[!,:obj_passive]  = blocks .* 0.0
df_results[!,:obj_active]   = blocks .* 0.0

t1 = 100
t2 = length(TT_daily)

for block=t1:t2
    T   = TT_daily[block]

    #m3s = Model(CPLEX.Optimizer)
    m3s = Model(Gurobi.Optimizer)
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
        
        e[T]      >= 0 # electrolyzer consumption
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
        z_start[T]  >= 0 # start electrolyzer 
    end

    ## objective function
    @objective(m3s, Min, 
        -sum(p_DA[t]*lambda_DA[t]
            + p_B[t]*(lambda_B[t])*0.9999
            - (p_DAn[t]+p_Bn[t])*lambda_TSO
            + lambda_H*d[t]
            - lambda_start*z_start[t] for t=T))


    #****************************   Curtailed power    ************************************* 
    @constraint(m3s, p_w .<= P_W[T] )


    #**************************     aux constraints     ************************************
    # sanity caps on power sold in DA and balance market
    @constraint(m3s, [t=T],    -p_E[t]-p_C[t]  <= p_DA[t])
    @constraint(m3s, [t=T],    -p_w[t]  <= p_DA[t])
    @constraint(m3s, [t=T],    C_W >= p_DA[t])
    
    # energy in must be equal to energy out
    @constraint(m3s, [t=T], p_w[t] - p_DA[t] - p_B[t] - p_E[t] - p_C[t] == 0)

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


    df_results[block, :obj]          = -objective_value(m3s)
    df_results[block, :obj_passive]  = innerRealisation_passive([value.(p_DA)[t] for t=T], [value.(p_E)[t] for t=T], [value.(p_C)[t] for t=T], T)
    df_results[block, :obj_active]   = innerRealisation_active([value.(p_DA)[t] for t=T], T)


    println("Block: ", block, " - ")
    println("Objective value estimate: ", objective_value(m3s))
    println("Objective value actual: ", df_results[block, :obj_active])

end

CSV.write("Output/1-stage/DET.csv", df_results)

sum(df_results[100:end, :obj_passive])
sum(df_results[100:end, :obj_active])
mean(df_results[100:end, :obj_passive])
mean(df_results[100:end, :obj_active])



