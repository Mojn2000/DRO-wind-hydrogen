using CSV
using DataFrames
using Statistics


#### 1-stage models (24h) ####
## Deterministic model
df = DataFrame(CSV.File("Output/1-stage/DET.csv"))
sum(df[90:end,:obj_passive])
std(df[90:end,:obj_passive])
sum(df[90:end,:obj_active])
std(df[90:end,:obj_active])


## SAA model
df = DataFrame(CSV.File("Output/1-stage/SAA.csv"))
sum(df[90:end,:obj_passive])
std(df[90:end,:obj_passive])
sum(df[90:end,:obj_active])
std(df[90:end,:obj_active])


## DRO model
df = DataFrame(CSV.File("Output/1-stage/DRO.csv"))
sum(df[90:end,:obj_passive])
std(df[90:end,:obj_passive])
sum(df[90:end,:obj_active])
std(df[90:end,:obj_active])


## DRO with correlation model
df = DataFrame(CSV.File("Output/1-stage/DRO-cor.csv"))
sum(df[90:end,:obj_passive])
std(df[90:end,:obj_passive])
sum(df[90:end,:obj_active])
std(df[90:end,:obj_active])


#### 2-stage models (8h) ####
## DRO model
df = DataFrame(CSV.File("Output/2-stage/DRO.csv"))
sum(df[90*3:end,:obj_active])
std(df[90*3:end,:obj_active])



