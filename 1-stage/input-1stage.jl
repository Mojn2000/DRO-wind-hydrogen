# windFor-to-X Modelling and Optimization
#************************************************************************
# Input Data
#************************************************************************
using DataFrames
using CSV
using PyCall
using Plots
using Clustering
#using Distances
using JLD2
using Distributions
using Statistics


digs = 2 # rounding digits
#************************************************************************
# read csv 
df = DataFrame(CSV.File("market data/merged-data.csv"))

# set missing values to previous value for 'imbalMeas'
for i in 2:length(df.imbalMeas)
    if ismissing(df.imbalMeas[i])
        df.imbalMeas[i] = df.spotMeas[i-1]
    end
end
df.imbalPred = df.spotPred 

# time parameters
block_size = 8 # artificial block size in hours
T = collect(1:nrow(df))

# Wind farm
C_W = 3     # cap wind in MW
CF = df[T,:windFor]    # wind production
P_W = CF * C_W          # wind production 
P_W_actual = df[T,:windMeas]*C_W

lambda_DA        = df[T,:spotPred] # electricity day ahead market price
lambda_DA_actual = df[T,:spotMeas]

lambda_B         = df[T,:imbalPred] # electricity balance market price
lambda_B_actual  = df[T,:imbalMeas] 


# Electrolyzer
C_E = C_W/2 # size in MW
sb_load = 0.01 # standby load = 1%
P_sb = C_E * sb_load # standby load = 1%
min_load = 0.15 # minimum load = 15%
P_min = C_E * min_load # minimum load = 15%
p_cell = 30 # cell pressure bar
T_cell = 90 # cell temperature in celsius
i_max = 5000 # maximum cell current density in A/m2
A_cell = 0.2 # cell area in m2
start_cost = 50 # starting cost of production = 50 EUR/MW
lambda_start = C_E * start_cost # starting cost of production
eta_full_load = 17.547 # constant production efficiency kg/MWh
lambda_TSO = 15.06 # TSO grid tariff in EUR/MWh 

# Hydrogen market
lambda_H = 2.10 # EUR per kg

# Hydrogen storage
C_S = C_E * eta_full_load * block_size # max size in kg
soc_0 = 0 # initial storage in MWh

# Compressor
eta_C = 0.75 # mechanical efficiency in %
p_in = 30 # inlet pressure in bar
p_out = 200 # outlet pressure in bar
gamma = 1.4 # adiabatic exponent
T_in = 40 + 273.15 # inlet temperature in K
R = 8.314 # universal gas constant in J/mol*K
M_H2_kg = 2.0159E-03 # molar mass of H2 in kg/mol
P_C = R * T_in / M_H2_kg * gamma / (gamma-1) * 1 / eta_C * ((p_out/p_in)^((gamma-1)/gamma)-1) * 1E-06 / 3600 # compressor consumption in MWh/kg H2

# Daily hours
TT_daily = []
for x in 1:Int(length(T)/block_size)
    hour = []
    for y in 1:block_size
        push!(hour, y+(x-1)*block_size)
    end
    push!(TT_daily,hour)
end

# Demand
C_D_daily = C_E * eta_full_load * 4 *block_size/24 # in kg per day
C_D_monthly = C_D_daily * 30 # in kg per year
C_D_annual = C_D_daily * 365 # in kg per year

# Set demand style
C_D = C_D_daily
TT = TT_daily

#************************************************************************

Q = []
for t in 1:block_size
    push!(Q, [1 0; -1 0; 0 1; 0 -1])
end

h_rhs = []
for t in 1:block_size
    push!(h_rhs, [10000; 10000; 10000; 10000])
end

NE      = 40 # number of historical errors

index_price = Array{Any,1}(undef, length(T))
clusters = load_object("Clusters")


for t in 1:length(T)
    index_price[t] = []
end

j = [1]
for t in Int(720/block_size+1):length(T)
    println(j[1])
    #idx = findall( clusters[546].assignments[t] .== clusters[546].assignments)
    idx = findall( clusters[j[1]].assignments[t] .== clusters[j[1]].assignments)
    #idx = idx =round.(1:7296)
    bb = findall([sum(TT_daily[block].==t)>0 for block=1:length(TT_daily)])
    if ( sum(idx .< (TT_daily[bb[1]][1] - 12)) >=  10 ) 
        idx = idx[idx .< (TT_daily[bb[1]][1] - 12)]
        #idx = idx[idx .< (t-24)]
    end
    #idx = idx[idx .< (TT_daily[bb[1]][1] + 36)]
    idx = reverse(idx)
    index_price[t] = idx[1:minimum([length(idx) NE])]
    if mod(t,24)==0 && t>=744
        j[1]=j[1]+1
        
    end
end


function adjust_vectors(test_da, test_bal, quantile_cut)

    # Check if the vectors already satisfy the condition
    if all(abs.(test_da .- test_bal) .<= quantile_cut)
        return test_da, test_bal
    end

    
    adjusted_test_da = test_da

    adjusted_test_bal = min.(test_bal, test_da .+ quantile_cut)
    adjusted_test_bal = max.(adjusted_test_bal, test_da .- quantile_cut)

    return adjusted_test_da, adjusted_test_bal
end

index_windwindMeas = copy(index_price)

total_da = df[1:912*8,:spotMeas]-df[1:912*8,:spotPred]
total_bal = df[1:912*8,:imbalMeas]-df[1:912*8,:spotPred]

difference = total_da-total_bal
quantile_cut = quantile(abs.(difference), 0.95)

#Using the index vectors make similar vectors for the errors
error_windwindMeas = Array{Any,1}(undef, length(T))
error_DA        = Array{Any,1}(undef, length(T))
error_B         = Array{Any,1}(undef, length(T))

improv = 0.0
# fill error vectors with the errors of the prices and wind windFor
global frac_rounded = 0
for i in 1:length(T)
    dif_p = df[index_windwindMeas[i],:windMeas] .- df[index_windwindMeas[i],:windFor]
    true_p = df[i,:windMeas] .- df[i,:windFor]
    error_windwindMeas[i] = min.(max.(-P_W[i],(dif_p .+ improv*(true_p .-mean(dif_p)) )),C_W-P_W[i])

    global frac_rounded = frac_rounded + sum(-P_W[i] .== error_windwindMeas[i])
    global frac_rounded = frac_rounded + sum(C_W-P_W[i] .== error_windwindMeas[i])


    dif_da = df[index_price[i],:spotMeas] .- df[index_windwindMeas[i],:spotPred]
    true_da = df[i,:spotMeas] .- df[i,:spotPred]
    error_DA[i] = dif_da .+ improv*(true_da .-mean(dif_da))
    
    dif_b = df[index_price[i],:imbalMeas] .- df[index_windwindMeas[i],:imbalPred]
    true_b = df[i,:imbalMeas] .- df[i,:imbalPred]
    error_B[i]  = dif_b .+ improv*(true_b .-mean(dif_b))
end

# create vector of 24 with the length of the index vectors of type Int32
length_error = Array{Int32,1}(undef, length(T))
# fill the length vectors with the length of the index vectors
for i in 1:length(T)
    length_error[i] = Int(length(index_windwindMeas[i]))
end


#************************************************************************
# Coefficients
a1	= 1.5184
a2	= 1.5421E-03
a3	= 9.523E-05
a4	= 9.84E-08
r1 = 4.45153E-05
r2 = 6.88874E-09
d1 = -3.12996E-06
d2 = 4.47137E-07
s = 0.33824
t1 = -0.01539
t2 = 2.00181
t3 = 15.24178
B1 = 4.50E-05
B2 = 1.02116
B3 = -247.26
B4 = 2.06972
B5 = -0.03571
f11 = 478645.74
f12 = -2953.15
f21 = 1.0396
f22 = -0.00104
F_const = 96485.3321
M_H2 = 2.0159 # molar mass of H2 in kg/mol
HHV = 39.41
#************************************************************************
#************************************************************************
#Functions
# Reversible cell voltage
function U_rev(Temp)
    Temp_K = Temp + 273.15
    U_rev = a1 - a2 * Temp_K + a3 * Temp_K * log(Temp_K) + a4 * Temp_K^2
    return U_rev
end
# Real cell voltage
function U_cell(Temp,p,i)
    U_cell = U_rev(Temp) + ((r1 + d1) + r2 * Temp + d2 * p) * i + s * log(10,(t1 + t2 / Temp + t3 / Temp^2) * i + 1) 
    return U_cell
end
# Cell windFor consumption
function P_cell(Temp,p,i)
    P_cell = i * U_cell(Temp,p,i)
    return P_cell
end
# Faraday efficiency (5-parameter)
function eta_F_5(Temp,i)
    eta_F = B1 + B2 * exp((B3 + B4 * Temp + B5 * Temp^2) / i)
    return eta_F
end
# Faraday efficiency
function eta_F(Temp,i)
    eta_F = (i^2 / (f11 + f12 * Temp + i^2)) * (f21 + f22 * Temp)
    return eta_F
end
# Cell production
function M_H_cell(Temp,i)
    M_H_cell = (eta_F(Temp,i) * M_H2 * i) / (2 * F_const)
    M_H_cell_kg_h = M_H_cell * 3.6
    return M_H_cell_kg_h
end
# System production
function M_H_sys(Temp,i,I,n_c)
    M_H_cell = (eta_F(Temp,i) * n_c * M_H2 * I) / (2 * F_const)
    M_H_cell_kg_h = M_H_cell * 3.6
    return M_H_cell_kg_h
end
# Cell efficiency
function eta_cell(Temp,p,i)
    eta_cell = M_H_cell(Temp,i) * HHV / P_cell(Temp,p,i)
    return eta_cell
end
# Number of cells
function n_cell(i_max,A_cell,C_E,Temp,p)
    I_max_cell = i_max * A_cell
    U_max_cell = U_cell(Temp,p,i_max)
    P_max_cell = I_max_cell * U_max_cell
    #n_cell = ceil((C_E * 1000000) / P_max_cell)
    n_cell = (C_E * 1000000) / P_max_cell
    return n_cell
end
# Production curve
function P_curve(P_list,S,i_max,A_cell,C_E,Temp,p)
    N = collect(1:length(P_list))
    a = []
    b = []
    n_x = []
    n_y = []
    n_c = n_cell(i_max,A_cell,C_E,Temp,p)
    include("../find_current.jl")
    i_list = py"find_i_from_p"(P_list,C_E,n_c,A_cell,Temp,p)

    for n=N 
        i = i_list[n]
        I = i_list[n] * A_cell
        U = U_cell(Temp,p,i) * n_c
        P = I * U / 1000000
        M = M_H_sys(Temp,i,I,n_c)
        push!(n_x, P)
        push!(n_y, M)        
    end
    for s=S # a and b for segment
        a_s = (n_y[s] - n_y[s+1]) / (n_x[s] - n_x[s+1])
        b_s = n_y[s] - (a_s * n_x[s])
        push!(a, a_s)
        push!(b, b_s)
    end

    return a,b,n_x,n_y
end
#************************************************************************
# Production curve
P_E_min = min_load # minimum load for production
P_E_opt = 0.28231501
P_E_max = 1
P_segments = [
    [P_E_min,
        P_E_max], #1
    [P_E_min,
        P_E_opt,
        P_E_max], #2
    [],
    [P_E_min,
        (P_E_min+P_E_opt)/2,
        P_E_opt,
        (P_E_opt+P_E_max)/2,
        P_E_max], #4
    [],[],[],
    [P_E_min,
        (P_E_min+(P_E_min+P_E_opt)/2)/2,
        (P_E_min+P_E_opt)/2,
        ((P_E_min+P_E_opt)/2+P_E_opt)/2,
        P_E_opt,
        (P_E_opt+(P_E_opt+P_E_max)/2)/2,
        (P_E_opt+P_E_max)/2,
        ((P_E_opt+P_E_max)/2+P_E_max)/2,
        P_E_max], #8
    [],[],[],
    [P_E_min,
        (P_E_min+(P_E_min+P_E_opt)/2)/2,
        (P_E_min+P_E_opt)/2,
        ((P_E_min+P_E_opt)/2+P_E_opt)/2,
        P_E_opt,
        (P_E_opt+(P_E_opt+(P_E_opt+P_E_max)/2)/2)/2,
        (P_E_opt+(P_E_opt+P_E_max)/2)/2,
        ((P_E_opt+(P_E_opt+P_E_max)/2)/2+(P_E_opt+P_E_max)/2)/2,
        (P_E_opt+P_E_max)/2,
        ((P_E_opt+P_E_max)/2+((P_E_opt+P_E_max)/2+P_E_max)/2)/2,
        ((P_E_opt+P_E_max)/2+P_E_max)/2,
        (((P_E_opt+P_E_max)/2+P_E_max)/2+P_E_max)/2,
        P_E_max] #12
]
segments = 12
P_list = P_segments[segments]
S = collect(1:length(P_list)-1)
curve = P_curve(P_list,S,i_max,A_cell,C_E,T_cell,p_cell)
a = curve[1]
b = curve[2]


