#Simulation
n.sample <- rnorm(n = 10000, mean = 55, sd = 4.5)

#Skewness and Kurtosis
library(moments)
skewness(trainingSet$cont1)
kurtosis(trainingSet$cont1)

ggplot(trainingSet, aes(x = cont6), binwidth = 2) + 
  geom_histogram(aes(y = ..density..), fill = 'red', alpha = 0.5) + 
  geom_density(colour = 'blue') + 
  xlab(expression(bold('Simulated Samples'))) + 
  ylab(expression(bold('Density')))

library(forecast)
# to find optimal lambda
lambda = BoxCox.lambda( trainingSet$cont6 )
# now to transform vector
trainingSet$cont6BC = BoxCox( trainingSet$cont6, lambda)

