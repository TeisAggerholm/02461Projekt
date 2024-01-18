
############################################################################################## OVERNIGHT
data=read.csv("rewards_DQN_20.csv")

plot(data$X.968.0,col="green4",pch=20,xlab = "Episodes", ylab= "Overall reward")
mean(data$X.968.0)
data
################################################################################# Pilot of 200 episodes: 
data=read.csv("rewards_DQN_21.csv")
plot(data$X.844.0)
mean(data$X.844.0) #-834.8844
var(data$X.844.0) #153192.9
sd(data$X.844.0)
#Sample size requiered for determing real mean: 
1.96**2*(391.4^2)/25**2 #Mindst 941.6094 episoder. Vi tager derfor 950 episoder

######## Real mean of overnight acc. to sample size: 
data=read.csv("rewards_DQN_25.csv")
mean(data$X.526.0) #-815.9747
sd(data$X.526.0) #334.6542

-815.9747 + c(-1,1)*1.96*334.6542/sqrt(950) #-837.2556 -794.6938

################################################################################ Pilot for interval 200 episodes: 
data=read.csv("rewards_interval_model_24.csv")
plot(data$X.613.0)
mean(data$X.613.0) #-492.3216
var(data$X.613.0) #25816.12 
sd(data$X.613.0)

#Sample size requiered for determing real mean: 
1.96**2*(25816.12)/25**2 #158.6803  Vi tager 160 episoder. 


######## Real mean of interval model acc. to sample size:
data=read.csv("rewards_interval_model_26.csv")
mean(data$X.570.0) #-492.8616
sd(data$X.570.0) #171.5224 

-492.8616+c(-1,1)*1.96*171.5224/sqrt(160) #-519.4393 -466.2839

################################################################################ TWO SAMPLE WELCH p-value: 
((492.8616-815.9747) - 0)/sqrt( (171.5224^2/160)+(334.6542^2/950) ) #18.60039

v = 400.7845671

pvalue = 2 * (1-pt(abs(-18.60039), df = 400.7845671))
pvalue 
