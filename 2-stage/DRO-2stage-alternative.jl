using JuMP
using Gurobi
using Statistics
using CSV

# Model
include("input-2stage.jl")
include("../innerRealization.jl")

# read Wasserstein distance based on wind power only
df_Wdist = CSV.read("2-stage/Wdists_wind.csv", DataFrame, header=true)
df_Wdist = Matrix(df_Wdist)

# read synthetic samples
df_samples = CSV.read("2-stage/X_sample.csv", DataFrame, header=false) # DA, B, pow
df_samples = Matrix(df_samples)

# read Wasserstein distance based on all variables
dist = CSV.read("2-stage/dist_all.csv", DataFrame, header=true)
dist = Matrix(dist)

# DRO parameters
theta = 20             # Wasserstein radius
nS = 500 #size(df_samples)[1]  # number of decretization points
#nS = 0  # number of decretization points
nN = 10                   # number of samples

## init data frame for results
blocks = 1:length(TT_daily)
df_results = DataFrame()
df_results[!,:block]        = blocks
df_results[!,:obj]          = blocks .* 0.0
df_results[!,:obj_active]   = blocks .* 0.0

block = 300

param_list = [8 16]


it = 0
while it < length(param_list)
    it = it + 1
    theta = param_list[it]

    # print hyper parameters
    println("param_list: ", param_list)
    println("Iteration: ", it)
    println("theta: ", theta)

    ## in sample
    t1 = 42*3+1
    #t2 = 546
    #t2 = 300
    ## out of sample
    #t1 = 547
    t2 = length(TT_daily)
    block = t1

    for block=t1:t2
        # find the 10 closest empirical sample to current based on wind power only
        T = TT_daily[block] # time index
        SN = [] # empirical set
        aux = sort(df_Wdist[block,1:(block-1)])[1:nN] 
        for i in 1:length(aux)
            for j in 1:length(df_Wdist[block,1:(block-1)])
                if aux[i] == df_Wdist[block,j]
                    push!(SN,j)
                end
            end
        end

        # def problem
        m = Model(Gurobi.Optimizer)

        # 1-stage variables
        @variables m begin
            p_DA[T]         # electrictiy from day ahead market
            p_DAp[T] >= 0   # electrictiy from day ahead market (positive component)
            p_DAn[T] >= 0   # electrictiy from day ahead market (negative component)

            alphaVar >= 0   # Wasserstein variable
        end

        gamma = @variable(m, [SN])

        # 1-stage objective
        @objective(m, Min,  -sum(
            lambda_DA[t]*p_DA[t]
            - lambda_TSO*p_DAn[t]
            for t=T)
            + theta*alphaVar
            + 1/length(gamma)*sum(gamma))

        # 1-stage constraints
        #@constraint(m, [t=T],   -sum(p_E[s][t]-p_C[s][t] for s=1:(nN+nS))/(nN+nS)  <= p_DA[t])
        #@constraint(m, [t=T],   -sum(p_w[s][t] for s=1:(nN+nS))/(nN+nS) <= p_DA[t])    # limit speculation
        @constraint(m, [t=T],      0 <= p_DA[t])    # limit speculation
        @constraint(m, [t=T],    C_W >= p_DA[t])    # limit speculation
        @constraint(m, [t=T],    p_DA[t] == p_DAp[t] - p_DAn[t])


        ## Inner program
        # 2-stage variables
        p_B     = Array{Any,1}(undef, nN+nS)
        p_Bp    = Array{Any,1}(undef, nN+nS)
        p_Bn    = Array{Any,1}(undef, nN+nS)
        p_w     = Array{Any,1}(undef, nN+nS)
        for s in 1:(nN+nS)
            p_B[s]    = @variable(m, [T])
            p_Bp[s]   = @variable(m, [T], lower_bound = 0)
            p_Bn[s]   = @variable(m, [T], lower_bound = 0)

            p_w[s]    = @variable(m, [T], lower_bound = 0)
        end

        ## Deterministic variables
        p_E     = Array{Any,1}(undef, nN+nS)
        p_C     = Array{Any,1}(undef, nN+nS)
        e       = Array{Any,1}(undef, nN+nS)
        h       = Array{Any,1}(undef, nN+nS)
        h_d     = Array{Any,1}(undef, nN+nS)
        d       = Array{Any,1}(undef, nN+nS)
        s_in    = Array{Any,1}(undef, nN+nS)
        s_out   = Array{Any,1}(undef, nN+nS)
        soc     = Array{Any,1}(undef, nN+nS)
        z_h     = Array{Any,1}(undef, nN+nS)
        z_on    = Array{Any,1}(undef, nN+nS)
        z_off   = Array{Any,1}(undef, nN+nS)
        z_sb    = Array{Any,1}(undef, nN+nS)
        z_start = Array{Any,1}(undef, nN+nS)

        for s in 1:(nN+nS)
            p_E[s]  = @variable(m, [T],     lower_bound = 0)
            p_C[s]  = @variable(m, [T],     lower_bound = 0)
            e[s]    = @variable(m, [T],   lower_bound = 0)
            h[s]    = @variable(m, [T],     lower_bound = 0)
            h_d[s]  = @variable(m, [T],     lower_bound = 0)
            d[s]    = @variable(m, [T],     lower_bound = 0)
            s_in[s] = @variable(m, [T],     lower_bound = 0)
            s_out[s]= @variable(m, [T],     lower_bound = 0)
            soc[s]  = @variable(m, [T],     lower_bound = 0)
            #z_h[s]  = @variable(m, [T,S],   Bin)
            z_on[s] = @variable(m, [T],     Bin)
            #z_off[s]= @variable(m, [T],     Bin)
            z_sb[s] = @variable(m, [T],     Bin)
            z_start[s] = @variable(m, [T],  lower_bound = 0)
        end

        # 2-stage constraints
        for s in 1:(nN+nS), n in 1:nN
            if s <= nN  # empirical sample
                lda_s = df[TT_daily[SN[s]], :nominal]       # lambda_DA for sample
                lb_s  = df[TT_daily[SN[s]], :bal_price]     # lambda_B for sample
                pw_s  = df[TT_daily[SN[s]], :power_actual]  # wind power for sample  
            else        # synthetic sample
                lda_s = df_samples[s-nN,1:8]  # lambda_DA for sample
                lb_s  = df_samples[s-nN,9:16] # lambda_B for sample
                pw_s  = df_samples[s-nN,17:24]# wind power for sample  
            end

            ## epigraph (inner objective)
            Q = (lda_s - lambda_DA[T])'*p_DA + lb_s'*p_B[s] - sum(lambda_TSO.*p_Bn[s]) + sum(lambda_H.*d[s]) - sum(z_start[s]).*lambda_start
            if s <= nN
                @constraint(m, gamma[SN[n]] >= -Q - alphaVar*dist[SN[n], SN[s]])
            else
                @constraint(m, gamma[SN[n]] >= -Q - alphaVar*dist[SN[n], length(TT_daily)+s-nN])
            end

            ## inner constraints
            @constraint(m, p_w[s] .<= pw_s)
            @constraint(m, p_w[s] .- p_DA .- p_B[s] .- p_E[s] .- p_C[s] .== 0)
            @constraint(m, p_B[s] .== p_Bp[s] .- p_Bn[s])


            #******************* Deterministic constraints (from original problem) *************************
            # Total electricity consumption
            @constraint(m,[t=T], p_E[s][t] == e[s][t] + P_sb * z_sb[s][t])
            # Hydrogen production
            @constraint(m,[t=T,seg=S], h[s][t] <= a[seg] * e[s][t] + b[seg]*z_on[s][t])
            # Segment min power boundary
            #@constraint(m,[t=T,seg=S], e[s][t,seg] >= P_segments[segments][seg] * C_E * z_h[s][t,seg])
            # Segment max power boundary
            #@constraint(m,[t=T,seg=1:length(S)-1], e[s][t,seg] <= P_segments[segments][seg+1] * C_E * z_h[s][t,seg])
            # Hydrogen balance
            @constraint(m, [t=T],  h[s][t] == h_d[s][t] + s_in[s][t])
            # Demand balance
            @constraint(m, [t=T],        d[s][t] == h_d[s][t] + s_out[s][t])
            # Maximum storage output
            @constraint(m, [t=T],        s_out[s][t] <= C_E * eta_full_load)
            # Hydrogen min demand (DAILY)
            @constraint(m, sum(d[s]) >= C_D)
            # Maximum electrolyzer power
            @constraint(m, [t=T],        p_E[s][t] <= C_E * z_on[s][t] + P_sb * z_sb[s][t])
            # Minimum electrolyzer power
            @constraint(m, [t=T],        p_E[s][t] >= P_min * z_on[s][t] + P_sb * z_sb[s][t])
            # only one efficiency if on or standby
            #@constraint(m, [t=T],        z_on[s][t] == sum(z_h[s][t,seg] for seg=S))
            # States
            @constraint(m, [t=T],    1 >= z_on[s][t] + z_sb[s][t])
            # Not from Off to Standby
            @constraint(m, [t=T[2:end]],    z_sb[s][t] <= z_on[s][t-1] + z_sb[s][t-1])
            # Startup cost
            @constraint(m, [t=T[2:end]],    z_start[s][t] >= z_on[s][t] - z_on[s][t-1] - z_sb[s][t-1])
            # Compressor consumption
            @constraint(m, [t=T],      p_C[s][t] == s_in[s][t] * P_C)
            # Max storage fill
            @constraint(m, [t=T],      soc[s][t] <= C_S)
            # SOC
            @constraint(m, [t=T],      soc[s][t] == (t > T[1] ? soc[s][t-1] : 0) - s_out[s][t] + s_in[s][t])
        end


        2+2

        # Solve
        optimize!(m)

        
        value.(p_DA)
        value.(gamma)
        value.(alphaVar)

        df[TT_daily[block],:]


        if termination_status(m) == MOI.OPTIMAL
            df_results[block, :obj]          = -objective_value(m)
            df_results[block, :obj_active]   = innerRealisation_active([value.(p_DA)[t] for t=T], T)
        else
            df_results[block, :obj]             = 0
            df_results[block, :obj_active]      = innerRealisation_active(zeros(length(T)), T)
        end

        print("Block: ", block, " - ")
        print("Objective value estimate: ", df_results[block, :obj])
        print("Objective value actual: ",   df_results[block, :obj_active])
    end

    CSV.write("Data/Output/2-stage/DRO-v3-theta-"*string(theta)*".csv", df_results)
end


