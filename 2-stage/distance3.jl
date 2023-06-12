using JuMP
using Gurobi

include("input-2stage.jl")

# read samples
df_samples = CSV.read("2-stage/X_sample.csv", DataFrame, header=false) # DA, B, pow
df_samples = Matrix(df_samples)

nN = length(TT_daily)
nS = size(df_samples)[1]    # number of decretization points

# init dist matrix
dist = zeros(nN,nN+nS)

# compute dist matrix
for i in 1:nN
    a = [df[TT_daily[i],:spotMeas]; df[TT_daily[i],:imbalMeas]; df[TT_daily[i],:windMeas]]
    for j in 1:nN
        b = [df[TT_daily[j],:spotMeas]; df[TT_daily[j],:imbalMeas]; df[TT_daily[j],:windMeas]]
        dist[i,j] = EarthMoverDistance(a,b)
    end
    for j in 1:nS
        b = df_samples[j,:]
        dist[i,j+nN] = EarthMoverDistance(a,b)
    end
end

# save dist matrix
CSV.write("2-stage/dist-meas.csv", DataFrame(dist, :auto))



