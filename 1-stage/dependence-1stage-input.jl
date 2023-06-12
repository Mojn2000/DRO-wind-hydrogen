using Statistics

include("../innerRealization.jl")
include("input-1stage.jl")

# becasue we transform to min problem, errors must be multiplied by -1
error_DA = error_DA .* -1
error_B  = error_B .* -1

# set length_error to min between length_error and maxN
length_error = [length(error_DA[t]) for t=T]

## empirical mean of errors
mean_P = Array{Any,1}(undef, length(T))
for i in 1:length(T)
    if length_error[i] == 0
        mean_P[i] = [0 0]
    else
        mean_P[i] = [mean(error_DA[i]) mean(error_B[i])]
    end
end

eig_list = []
## compute covariance 
using LinearAlgebra
cov_DA_B = Array{Any,1}(undef, length(T))
for i in 1:length(T)
    if length_error[i] == 0
        cov_DA_B[i] = [0 0; 0 0]
    else
        var_DA = mean((error_DA[i] .- mean_P[i][1]).^2)
        var_B  = mean((error_B[i]  .- mean_P[i][2]).^2)
        covdab = mean(error_DA[i] .* error_B[i]) .- mean_P[i][1]*mean_P[i][2]
        cov_DA_B[i] = [var_DA covdab; covdab var_B]

        # find eigenvalues 
        append!(eig_list, eigvals(cov_DA_B[i]))
    end
end

## prepare data DataFrame
df.p_DA = df.windMeas*0
df.p_DAn= df.windMeas*0
df.p_B  = df.windMeas*0
df.p_Bn = df.windMeas*0
df.p_E  = df.windMeas*0
df.p_C  = df.windMeas*0
df.d    = df.windMeas*0
df.start= df.windMeas*0

blocks = 1:length(TT_daily)
df_results = DataFrame()
df_results[!,:block]        = blocks
df_results[!,:obj]          = blocks .* 0.0
df_results[!,:obj_passive]  = blocks .* 0.0
df_results[!,:obj_active]   = blocks .* 0.0

