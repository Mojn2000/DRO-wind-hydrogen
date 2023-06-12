using JuMP
using Gurobi

# Model
include("input-2stage.jl")
nS = length(TT_daily)
Wdist_matrix = zeros(nS,nS)

# time loop
t = @elapsed begin

j1 = 1
for i in 1:nS,j in (i+1):nS
    Wdist_matrix[i,j] = EarthMoverDistance(P_W[TT_daily[i]], P_W[TT_daily[j]])
    if j1 != i
        println(i)
    end
    j1 = i
end
end

# copy upper triangle of Wdist_matrix into lower triangle
for i in 1:nS,j in (i+1):nS
    Wdist_matrix[j,i] = Wdist_matrix[i,j]
end

# save Wdist_matrix
CSV.write("2-stage/Wdists_wind.csv", DataFrame(Wdist_matrix, :auto))


