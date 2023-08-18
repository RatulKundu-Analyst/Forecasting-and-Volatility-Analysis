########################################################################
# Load Required Libraries
########################################################################
library(plotly)
library(lmtest)
library(dplyr)
library(TTR)
library(fpp2)
library(urca)
library(tseries)
library(timetk)
library(rugarch)
library(vrtest)
library(tsDyn)
library(FinTS)
library(lubridate)
library(fGarch)

##########################################
# Load Data set 
##########################################
SBI = read.csv("C:/Users/gamin/Desktop/VOLATILITY/DATA/SBI.csv")
head(SBI)

##########################################
# Pre-processing 
##########################################
# Null value treatment 
sum(is.na(SBI))

# Date transformation from sting to yer-month format 
SBI$Date= as.Date(SBI$Date, format = "%b %d, %Y")
SBI_sorted <- SBI[order(SBI$Date), ]
head(SBI_sorted)

# convert to time series data
Z = ts( data = SBI_sorted[,2])
head(Z,48)

# spacial transformation for volatility analysis
R=diff(log(Z))
head(R,47)

##########################################
# Data Visualization and Preliminary Analysis 
##########################################

# Time Plot - line plot
autoplot(Z) +
  ggtitle("Time Plot : Daily Closing Price (INR) of SBI ") +
  ylab("INR")

# Create the box plot
SBI_sorted$Year <- lubridate::year(SBI_sorted$Date)
ggplot(SBI_sorted, aes(x = factor(Year), y = Close)) +
  geom_boxplot( fill = "purple", col = "black" ,show.legend = F, 
      outlier.color = "darkgreen") +
  labs(x = "Year", y = "Stock Price", 
      title = "Year-wise Box Plot of SBI Stock Data" )

# Series appears trend-stationary, use to investigate seasonality.
SBI_sorted %>%
  plot_seasonal_diagnostics(Date,Close,
                        .feature_set= "quarter" ,.geom_color =  "darkgreen", 
                        .geom_outlier_color = "black" , .x_lab = "Quarters",
                        .y_lab = "Stock Price" ,
                        .title = "Quarter-wise Box Plot of SBI Stock Data")
SBI_sorted %>%
  plot_seasonal_diagnostics(Date,Close,
                        .feature_set= "month.lbl" ,.geom_color =  "darkgreen" , 
                        .geom_outlier_color = "black" , .x_lab = "Months", 
                        .y_lab = "Stock Price" ,
                        .title = "Month-wise Box Plot of SBI Stock Data")

# ACF
acf(Z)
# PACF
pacf(Z)
#DW test fo Autocorrelation 
model <- lm(Close ~ Date, data = SBI_sorted)
dw_test = dwtest(model)
dw_test

#adf test
# CREAT ROLLING MEAN & SD
{
  MA= c()
  for (i in 1:5093) {
    MA[i]= sum(Z[1:i])/i
  }
}
{
  SD= c()
  for (i in 1:5093) {
    SD[i]= sd(Z[1:i])
  }
}
ggplot(SBI_sorted, aes(x = Date)) +
  geom_line(aes(y = Close, col = "Stock Price")) +
  geom_line(aes(y = MA, col = "Rolling Mean")) +
  geom_line(aes(y = SD, col = "Rolling SD")) +
  labs(title = "Rolling Statistics for SBI Price",
       x = "Date",
       y = "Price / Rolling Statistics") +
  scale_color_manual(values = c("Stock Price" = "blue", 
            "Rolling Mean" = "green", "Rolling SD" = "red"))

adf.test(Z)

# Data has strong trend, Investigate transformation
# Take the first difference of the data to remove trend
DZ=diff(Z)
autoplot(DZ) +
  ggtitle("Time Plot : Change in Daily Closing Price of SBI ") +
  ylab("INR")


#DW test 
model <- lm(DZ ~ 1, data = NULL)
dw_test = dwtest(model)
dw_test

# ACF
acf(DZ, main = "ACF of 1st Difference series")
# PACF
pacf(DZ, main = "PACF of 1st Difference series")

# CREAT ROLLING MEAN & SD
{
  MA_DZ= c()
  for (i in 1:5092) {
    MA_DZ[i]= sum(DZ[1:i])/i
  }
}
{
  SD_DZ= c()
  for (i in 1:5092) {
    SD_DZ[i]= sd(DZ[1:i])
  }
}
z=diff(SBI_sorted$Pice)
ggplot(DZ, aes(x = 1:5092)) +
  geom_line(aes(y = DZ, col = "Stock Price", )) +
  geom_line(aes(y = MA_DZ, col = "Moving Average")) +
  geom_line(aes(y = SD_DZ, col = "Moving SD")) +
  labs(title = "Rolling Statistics for SBI Price",
       x = "Date",
       y = "Price / Rolling Statistics") +
  scale_color_manual(values = c("Stock Price" = "orange",
              "Moving Average" = "red", "Moving SD" = "black"))
adf.test(DZ)

##########################
#divide TRAIN and TEST set
##########################

#train and test set
train_Z= ts(Z[1:4838])
test_Z= ts(Z[4839:5093])
train_R= ts(R[1:5082])
test_R= ts(R[5083:5092])

#################
#ARIMA MODEL
#################
fit_arima=auto.arima(train_Z, stepwise = F,approximation = F, trace = T,
                     seasonal = F )
print(summary(fit_arima))
checkresiduals(fit_arima)

# forecasting
frcst = forecast(fit_arima, h= 255)
autoplot(frcst, include = 800, ylab= "Z")
rmse_arima=sqrt(mean((as.numeric(frcst$mean)-as.numeric(test_Z))^2))

#############################################
########### VOLATILITY ANALYSIS #############
#############################################

adf.test(R) # stationary test
# volatility  clustering
plot.ts(R, main = "Time Plot : Log-return of Daily Closing Price of SBI ")
acf(R)
ArchTest(R) # ARCH test


################ ARCH ###################

aic_arch = rep(NA, 100)
for(i in 1:100){
  model = garch(train_R, order = c(0,i))
  aic_arch[i]= AIC(model)
}
order= which.min(aic_arch)
order
# Forecast
model_arch = garchFit(formula = ~ garch(12,0), data = train_R, 
                      include.mean = F, cond.dist = "norm", trace = F)
arch_Fcst = predict(model_arch, n.ahead = 10)
AIC_arch = model_arch@fit$ics["AIC"]
rmse_arch=sqrt(mean((arch_Fcst$meanForecast - test_R)^2))
# PLOT
par(bg="#FFCCCB")
plot(c(train_R[4819:4838],test_R), type = "b",
     ylab = "Retun", main= "ARCH(12) : Observed vs Forcasted",col= "black",lwd=2)
lines(c(train_R[4819:4838],arch_Fcst$meanForecast),type= "b" ,col= "red",lwd=2)
lines(c(train_R[4819:4838]),type= "b" ,col= "blue",lwd=1)
abline(v=20.8,col="white",lty=2)
legend("bottomright",legend = c("Forecasted","Observed","Past Observation"),
       col = c("red","black","purple"), pch=c(16,16,16), cex= 0.85, 
       xpd = T,inset = c(-.001,1))

############### GARCH ###################
x=1
order_sgarch = matrix(NA,nrow = 25,ncol = 2)
aic_sgarch = rep(NA, 25)
for(i in 1:5){
  for(j in 1:5){
    model=garch(train_R, order = c(i,j))
    aic_sgarch[x]=AIC(model)
    order_sgarch[x,1]=i
    order_sgarch[x,2]=j
    x=x+1
  }
}
order1= which.min(aic_sgarch)
sgarch_order = c(order_sgarch[order1,1],order_sgarch[order1,2])
# Forecast
model_sgarch = garchFit(formula = ~ garch(3,1), data = train_R, 
                      cond.dist = "norm", trace = F, include.mean = F)
sgarch_Fcst = predict(model_sgarch, n.ahead = 10)
AIC_sgarch = model_sgarch@fit$ics["AIC"]
rmse_sgarch=sqrt(mean((sgarch_Fcst$standardDeviation - test_R)^2))

# Plot
plot(c(train_R[4819:4838],test_R), type="b",
     ylab = "Retun", main= "GARCH(3,1) : Observed vs Forcasted",col= "white",lwd=2)
lines(c(train_R[4819:4838],sgarch_Fcst$meanForecast),type= "b" ,col= "green4",lwd=2)
lines(c(train_R[4819:4838]),type= "b" ,col= "darkblue",lwd=2)
legend("bottomright",legend = c("Forecasted","Observed","Past Observation"),
       col = c("green4","white","darkblue"), pch=c(16,16,16), cex= 0.8, 
       xpd = T,inset = c(-.001,1))
abline(v=20.8,col="black",lty=2)


############### ARMA-GARCH ###################

aic_garch = rep(NA, 400)
order_garch = matrix(NA,nrow = 400,ncol = 4)
x=1
for(i in 1:4){
  for(j in 1:4){
    for(p in 0:4){
      for(q in 0:4){
        garch_spec = ugarchspec(variance.model = list(model="sGARCH" , 
                    garchOrder=c(i,j)) ,mean.model = list(armaOrder=c(p,0,q)))
        model_fit = ugarchfit(garch_spec, data = train_R)
        aic_garch[x]= infocriteria(model_fit)[1]
        order_garch[x,1]=i
        order_garch[x,2]=j
        order_garch[x,3]=p
        order_garch[x,4]=q
        x=x+1
      }
    }
  }
}
order2= which.min(aic_garch)
arma_garch_order = c(order_garch[order2,1],order_garch[order2,2])
arma_order = c(order_garch[order2,3],0,order_garch[order2,4])
# Forecast
SBI_garch = ugarchspec(variance.model = list(model="sGARCH", 
              garchOrder=arma_garch_order), 
              mean.model = list(armaOrder=arma_order) )
SBI_garch_fit = ugarchfit(SBI_garch, data = train_R)
garch_Fcst= ugarchforecast(SBI_garch_fit, n.ahead= 10)
AIC_armagarch=aic_garch[order2]
rmse_arma_garch=sqrt(mean((fitted(garch_Fcst) - test_R)^2))

# Plot
plot(c(train_R[4819:4838],test_R), type="b",ylab = "Retun", 
     main= "ARIM(1,0)-GARCH(2,3) : Observed vs Forcasted    ",col= "white",lwd=2)
lines(c(train_R[4819:4838],fitted(garch_Fcst)),type= "b" ,col= "green3",lwd=2)
lines(c(train_R[4819:4838]),type= "b" ,col= "darkblue",lwd=2)
legend("bottomright",legend = c("Forecast","Observed","Past Observation"),
       col = c("green3","white","darkblue"), pch=c(16,16), cex= 0.8, 
       xpd = T,inset = c(-.001,1))
abline(v=20.8,col="black",lty=2)

news_gacg = newsimpact(SBI_garch_fit)
plot(news_gacg$zx,news_gacg$zy,ylab= news_gacg$yexpr, xlab = news_gacg$xexpr,
     main = "News impact Curve")
plot(SBI_garch_fit)

############### E-GARCH ###################

aic_egarch = rep(NA, 110)
order_egarch = matrix(NA,nrow = 110,ncol = 2)
x=1
for(i in 0:5){
  for(j in 1:5){
    garch_spec = ugarchspec(variance.model = list(model="eGARCH" , 
            garchOrder=c(i,j)) ,mean.model = list(armaOrder=c(0,0)))
    model_fit = ugarchfit(garch_spec, data = train_R)
    aic_egarch[x]= infocriteria(model_fit)[1]
    order_egarch[x,1]=j
    order_egarch[x,2]=i
    x=x+1
  }
}
order3= which.min(aic_egarch)
egarch_order = c(order_egarch[order3,1],order_egarch[order3,2])
# Forecast
SBI_egarch = ugarchspec(variance.model = list(model="eGARCH" , 
        garchOrder=c(4,5)), mean.model = list(armaOrder=c(0,0)))
SBI_egarch_fit = ugarchfit(SBI_egarch, data = train_R)
egarch_Fcst= ugarchforecast(SBI_egarch_fit, n.ahead= 10)
AIC_egarch=aic_egarch[order3]
rmse_egarch=sqrt(sum((fitted(egarch_Fcst) - test_R)^2)/10)

# Plot
par(bg="lightgray")
plot(c(train_R[4819:4838],test_R), type="b",
     ylab = "Retun", main= "E-GARCH(4,5) : Observed vs Forcasted",col= "white",lwd=2)
lines(c(train_R[4819:4838],fitted(egarch_Fcst)),type= "b" ,col= "green3",lwd=2)
lines(c(train_R[4819:4838]),type= "b" ,col= "darkblue",lwd=2)
abline(v=20.8,col="red",lty=2)
legend("bottomright",legend = c("Forecast","Observed","Past Observation"),
       col = c("green3","white","darkblue"), pch=c(16,16), cex= 0.8, 
       xpd = T,inset = c(-.001,1))

####################################################
# Conclusion
####################################################
table= data.frame(
  MODEL = c("ARCH(12)","GARCH(3,1)","ARMA(1,0)-GRCH(2,3)","E-GARCH(4,5)"),
  AIC = c(AIC_arch,AIC_sgarch,AIC_armagarch,AIC_egarch),
  RMSE = c(rmse_arch,rmse_sgarch,rmse_arma_garch,rmse_egarch)
)
