using JuMP

function innerRealisation_active(p_DA, T)
    ## function to handle one realisation of the day ahead market
    ## p_DA is the day ahead market price
    ## T is the time steps for the day
    ## returns the objective value and the solution
    ##
    ##


    m3s = Model(CPLEX.Optimizer)
    #m3s = Model(Gurobi.Optimizer)
    # make model silent
    set_silent(m3s)

    #************************************************************************
    # Variables
    @variables m3s begin
        p_w[T] >= 0     # wind power (possibly curtailed)

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
        z_start[T]  >= 0 # start electrolyzer 
    end

    ## objective function
    @objective(m3s, Max, 
          p_DA'*df[T,:spotMeas]
        + p_B'*df[T,:imbalMeas]
        - sum((p_DAn.+p_Bn).*lambda_TSO)
        + sum(lambda_H.*d)
        - sum(lambda_start.*z_start))
    

    #****************************   Curtailed power    ************************************* 
    @constraint(m3s, p_w .<= df[T,:windMeas].*C_W)

    # energy in must be equal to energy out
    @constraint(m3s, p_w .- p_DA .- p_B .- p_E .- p_C .== 0)

    # positive and negative parts of DA and balance market
    @constraint(m3s,  0 .== p_DAp .- p_DAn .- p_DA)
    @constraint(m3s,  0 .== p_Bp  .- p_Bn  .- p_B)

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


    println([round(value.(p_DA[t-T[1]+1]*df[t,:spotMeas]
    + p_B[t]*df[t,:imbalMeas]
    - (p_DAn[t]+p_Bn[t]).*lambda_TSO
    + lambda_H*d[t]
    - lambda_start.*z_start[t]), digits = 1) for t in T])
    if termination_status(m3s) == MOI.OPTIMAL
        return objective_value(m3s)
    end
    return 0
end


function innerRealisation_passive(p_DA, pe_in, pc_in, T)
    ## function to handle one realisation of the day ahead market
    ## p_DA is the day ahead market price
    ## T is the time steps for the day
    ## returns the objective value and the solution
    ##
    ##


    m3s = Model(CPLEX.Optimizer)
    #m3s = Model(Gurobi.Optimizer)
    # make model silent
    set_silent(m3s)

    #************************************************************************
    # Variables
    @variables m3s begin
        p_w[T] >= 0     # wind power (possibly curtailed)

        p_DAp[T] >= 0   # electrictiy from day ahead market (positive component)
        p_DAn[T] >= 0   # electrictiy from day ahead market (negative component)
        
        p_B[T]          # electrictiy from balance market
        p_Bp[T] >= 0    # electrictiy from balance market (positive component)
        p_Bn[T] >= 0    # electrictiy from balance market (negative component)
        
        p_E[T]  >= 0    # electricity used for electrolyzer
        p_C[T]  >= 0    # electricity used for compressor
        
        e[T]        >= 0 # electrolyzer consumption for each segment
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
    @objective(m3s, Max, 
          p_DA'*df[T,:spotMeas]
        + p_B'*df[T,:imbalMeas]
        - sum((p_DAn.+p_Bn).*lambda_TSO)
        + sum(lambda_H.*d)
        - sum(lambda_start.*z_start))
    

    #****************************   Curtailed power    ************************************* 
    @constraint(m3s, p_w .<= df[T,:windMeas].*C_W)
    @constraint(m3s, p_E .- pe_in  .<= 1e-3)
    @constraint(m3s, p_E .- pe_in  .>= -1e-3)
    @constraint(m3s, p_C .- pc_in  .<= 1e-3)
    @constraint(m3s, p_C .- pc_in  .>= -1e-3)

    # energy in must be equal to energy out
    @constraint(m3s, p_w .- p_DA .- p_B .- p_E .- p_C .== 0)

    # positive and negative parts of DA and balance market
    @constraint(m3s,  0 .== p_DAp .- p_DAn .- p_DA)
    @constraint(m3s,  0 .== p_Bp  .- p_Bn  .- p_B)


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
    @constraint(m3s, sum(d) >= C_D-0.1)
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
    if termination_status(m3s) == MOI.OPTIMAL
        return objective_value(m3s)
    end
    return 0
end

