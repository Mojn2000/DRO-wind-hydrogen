#### Merge all data into one .csv file ####

elspot   = read.csv('dayahead-meas.csv') ## elspot prices
imbal    = read.csv('imbalance-meas.csv') ## imbalance price
windMeas = read.csv('wind-meas.csv') ## wind power measurement
windFor  = read.csv('wind-forecast.csv') ## wind power forecast
load     = read.csv('load-forecast.csv') ## load forecast
trans    = read.csv('trans-schedule.csv') ## scheduled transmission

t1 = as.POSIXct('2022-01-01', tz = 'UTC')
t2 = as.POSIXct('2023-03-01', tz = 'UTC')

df <- data.frame(TimeUTC = seq.POSIXt(t1,t2, 3600))

## merge data into df
elspot = elspot[as.POSIXct(elspot$TimeUTC, tz='UTC') %in% df$TimeUTC, ]
df[['spotMeas']] = elspot$PriceMeas
df[['spotPred']] = elspot$PriceMeas + rt(nrow(df),10)*elspot$PriceMeas/20 # simulate forecasts

imbal = imbal[as.POSIXct(imbal$TimeUTC, tz='UTC') %in% df$TimeUTC, ]
df[['imbalMeas']] = imbal$imbalMeas

windMeas = windMeas[as.POSIXct(windMeas$TimeUTC, tz='UTC') %in% df$TimeUTC, ]
df[['windMeas']] = windMeas$WindMeas

windFor = windFor[as.POSIXct(windFor$TimeUTC, tz='UTC') %in% df$TimeUTC, ]
df[['windFor']] = windFor$GreenPower
df[['windFor']] = df[['windFor']] / max(df[['windFor']]) # re-scale

load = load[as.POSIXct(load$TimeUTC, tz='UTC') %in% df$TimeUTC, ]
df[['loadFor']] = load$Load

trans = trans[as.POSIXct(trans$TimeUTC, tz='UTC') %in% df$TimeUTC, ]
df[['trans']] = trans$TotalFlow


## store df as .csv file
write.csv(df, 'merged-data.csv', quote=F, row.names=F)

