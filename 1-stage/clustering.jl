include("input-1stage.jl")
using MultivariateStats
using Distances
using RDatasets
using XGBoost


function adjust_vectors(test_da, test_bal, quantile_cut)

    # Check if the vectors already satisfy the condition
    if all(abs.(test_da .- test_bal) .<= quantile_cut)
        return test_da, test_bal
    end


    adjusted_test_bal = min.(test_bal, test_da .+ quantile_cut)
    adjusted_test_bal = max.(adjusted_test_bal, test_da .- quantile_cut)

    return adjusted_test_bal
end

hor = 36 # forecasting horizon
num_shifts = 8 # Number of shifts
burnin = 90
end_date = length(df[:,:spotMeas])/24-1


# Clustering
adj_ba = adjust_vectors(df[:,:spotMeas], df[:,:imbalMeas], 100)
clust_data = Matrix([adj_ba df[:,:windFor] df[:,:spotPred] df[:,:loadFor]])
forcast_data = Matrix([adj_ba df[:,:windFor] df[:,:spotPred] df[:,:loadFor] df[:,:trans]])



# Add the new columns to the matrix
clust_data_t = clust_data
X_t = clust_data_t[:,:]'
Xf_t = forcast_data'
Xi_t = copy(X_t)
Xif_t = copy(Xf_t)
for i in 1:length(X_t[:,1])
    Xi_t[i,:] = Xi_t[i,:]./std(X_t[i,:])
    Xif_t[i,:] = Xif_t[i,:]./std(Xf_t[i,:])
end

i = 0
Xii_t = Xi_t[:,1:(burnin+1+i)*24]
Xiif_t = Xif_t[:,1:(burnin+1+i)*24]

features = Xiif_t[2:end, 1:end]'
targets = Xiif_t[1, 1:end]'

n_samples, n_features = size(features)
n_targets = size(targets, 2)

shifted_features = zeros(n_samples - num_shifts, n_features * num_shifts)
shifted_targets = zeros(n_samples - num_shifts)

for i in 1:n_samples - num_shifts
    shifted_features[i, :] = vcat(features[i:i+num_shifts-1, :]...)
    shifted_targets[i] = targets[i+num_shifts]
end

# Define the features and target variables
features_train = shifted_features[1:end-hor, 1:end]
targets_train = shifted_targets[1:end-hor]

features_test = shifted_features[end-hor+1:end, 1:end]
targets_test = shifted_targets[end-hor+1:end]

num_round = 50
model = xgboost(features_train, num_round, label = targets_train, eta = 0.05, max_depth = 12)

#res = predict(model, features_test)
res2 = XGBoost.predict(model, features_test)

Xii_t[1,(burnin+1)*24-hor+1:(burnin+1)*24] .= (res2)


Dist_train_un = Distances.pairwise(Distances.SqEuclidean(), Xii_t)

R_train = kmedoids(Dist_train_un,20, maxiter=200)
time_tilnow = 0
end_date_adj = end_date-(burnin)
Rs = Array{Any,1}(undef, end_date_adj)
no_k = 20 
res = copy(Xi_t[1,:])
res .= 0
res[(burnin+1)*24-hor+1:(burnin+1)*24] .= res2
no_k = round.(no_k)
j = 1
for i in (burnin+1):end_date
    t = @elapsed begin
    Xi = Xi_t[:,1:(i+1)*24]
    Xii = Xif_t[:,1:(i+1)*24]
    
    features = Xii[2:end, 1:end]'
    targets = Xii[1, 1:end]'
    
    n_samples, n_features = size(features)
    n_targets = size(targets, 2)
    num_shifts = 8  # Number of shifts
    
    shifted_features = zeros(n_samples - num_shifts, n_features * num_shifts)
    shifted_targets = zeros(n_samples - num_shifts)
    
    for i in 1:n_samples - num_shifts
        shifted_features[i, :] = vcat(features[i:i+num_shifts-1, :]...)
        shifted_targets[i] = targets[i+num_shifts]
    end

    features_train = shifted_features[1:end-hor, 1:end]
    targets_train = shifted_targets[1:end-hor]

    features_test = shifted_features[end-hor+1:end, 1:end]
    targets_test = shifted_targets[end-hor+1:end]
    
    new_train_data = DMatrix(features_train, label=targets_train)
    XGBoost.update(model,20,new_train_data)
    
    res2 = XGBoost.predict(model, features_test)

    Xi[1,(i-1)*24+13:(i+1)*24] .= (res2)
    res[(i-1)*24+13:(i+1)*24] .= (res2)

    distanceMatrix = Distances.pairwise(Distances.SqEuclidean(), Xi)

    Rs[j] = kmedoids(distanceMatrix, Int(no_k); maxiter=200)
    end


    time_tilnow = time_tilnow + t
    print("Iteration: ")
    print(j)
    println("/$end_date_adj")
    print("Time past: ")
    println(time_tilnow)
    print("This round took: ")
    println(t)
    println("##############################")
    j = j+1
end


Rs2 = Array{Any,1}(undef, end_date_adj+1)
Rs2[1] = R_train
for i in 2:length(Rs2)
    println(i)
    Rs2[i] = Rs[i-1]
end
Rs2

res = res.*std(X_t[1,:])
save_object("Clusters",Rs2)