train.all$TransactionDate <- as.Date(train.all$TransactionDate)

few.columns <- c("TransactionDate", "SalesPrice","Amount","Age","ResidenceArea")

dataset <- train.all[,few.columns]

dataset$Age <- as.factor(dataset$Age)
dataset$ResidenceArea <- as.factor(dataset$ResidenceArea)

#######################################################################################################################################
# Data.visualisation.functions - Start
#######################################################################################################################################
theme_update(plot.title = element_text(hjust = 0.5))

# data.visualization <- function(dataset,target)
# {
#   all.features <- setdiff(names(dataset),target)
#   for(feature in all.features)
#   {
#     if(is(dataset[[feature]],"Date"))
#     {
#       
#     plot01 <-   ggplot(dataset, aes(x=dataset[[feature]], y=dataset[[target]]))+ 
#                 stat_summary(geom="ribbon", fun.data=mean_cl_normal ,
#                              fun.args=list(conf.int=0.95), fill="lightblue")+
#                 stat_summary(geom="line", fun.y=mean, linetype="dashed")+
#                 stat_summary(geom="point", fun.y=mean, color="red")+
#                 labs(title = paste0(target, " (mean) with 95% confidence by ", feature)
#                      , x = target
#                      , y = feature)
#     print(plot01)
#     
#     plot02  <-  ggplot(dataset, aes(x=dataset[[feature]], y=dataset[[target]]))+
#                 geom_line()+
#                 labs(title =paste0( feature , " vs ", target)
#                    , x = feature
#                    , y = target
#                    )
#     print(plot02)
#     
#     }
#     if(is.factor(dataset[[feature]]))
#     {
#     plot3 <- ggplot(dataset, aes(x=dataset[[feature]], y=dataset[[target]])) + 
#               stat_summary(fun.y = mean, geom = "bar") + 
#               stat_summary(fun.data = mean_se, geom = "errorbar", colour = "red") +
#               labs(title = paste0(target, " (mean) by ", feature, " and Errors")
#                    , x = target
#                    , y = feature)
#     print(plot3)
#     }
#     
#     if(is.numeric(dataset[[target]]))
#     {
#     plot100 <- ggplot(dataset, aes(dataset[[target]])) + 
#                geom_density()  +
#                 labs(title = paste0(target, " Density")
#                      , x = target
#                      , y = feature)
#     print(plot100)
#     }
#     
#   }
# }
# 
# data.visualization(dataset,"SalesPrice")
# data.visualization(dataset,"Amount")
# is.factor(dataset$Age)



generic.data.visualisations <- function(plottype,dataset, feature, target)
{
  
  if(plottype == "point")
  {
  plot <-   ggplot(dataset, aes(x=dataset[[feature]], y=dataset[[target]])) +
            geom_point(shape=1) +    
            geom_smooth(method=lm) + 
            labs(title = paste0(feature, " vs ", target)
                 , x = feature
                 , y = target) 
  
  print(plot)
  }
  if(plottype == "timeseries")
  {
  plot01 <-   ggplot(dataset, aes(x=dataset[[feature]], y=dataset[[target]]))+ 
    stat_summary(geom="ribbon", fun.data=mean_cl_normal ,
                 fun.args=list(conf.int=0.95), fill="lightblue")+
    stat_summary(geom="line", fun.y=mean, linetype="dashed")+
    stat_summary(geom="point", fun.y=mean, color="red")+
    labs(title = paste0(target, " (mean) with 95% confidence by ", feature)
         , x = target
         , y = feature)
  print(plot01)
  }
  
  if(plottype == "bar")
  {
    plot02 <-   ggplot(dataset, aes(x=dataset[[feature]], y=dataset[[target]])) +
                stat_summary(fun.y = mean, geom = "bar") +  
                stat_summary(fun.data = mean_se, geom = "errorbar", colour = "red") +
      labs(title = paste0(target, " (mean) by Age and Errors ",feature)
           , x = feature
           , y = target) 
    
    print(plot02)
  }
  
  if(plottype == "density")
  {
    plot03 <-   ggplot(dataset, aes(dataset[[feature]], colour = dataset[[target]])) +
                geom_density()  +
                labs(title = paste0(feature, " density by ", target) ) 
    
    print(plot03)
  }
  
  if(plottype == "discrete.bar")
  {
    plot04 <-   ggplot(dataset, aes(dataset[[feature]])) +
                geom_bar() + 
                labs(title = paste0(feature)
                     , x = feature
                       ) 
    print(plot04)
  }
  if(plottype == "continuous.hist")
  {
    plot05 <-   ggplot(dataset, aes(dataset[[feature]])) +
      geom_histogram() + 
      labs(title = paste0(feature)
           , x = feature
      ) 
    print(plot05)
  }
  if(plottype == "two.line")
  {
    plot05 <-   ggplot(dataset, aes(x=dataset[[feature]], y=dataset[[target]])) +
                geom_line() + 
                labs(title = paste0(target, " trend over ",feature)
                , x = feature
                , y = target
                     ) 
    print(plot05)
  }
  
}


generic.data.visualisations("discrete.bar", dataset, "Age")
generic.data.visualisations("continuous.hist", dataset,"SalesPrice")

generic.data.visualisations("point", dataset, "Amount","SalesPrice")
generic.data.visualisations("timeseries", dataset, "TransactionDate","SalesPrice")
generic.data.visualisations("bar", dataset, "Age","SalesPrice")
dataset$SalesPrice1 <- log(dataset$SalesPrice)
dataset$Amount1 <- log(dataset$Amount)
generic.data.visualisations("density", dataset, "SalesPrice","Age")
generic.data.visualisations("density", dataset, "SalesPrice1","Age")
generic.data.visualisations("density", dataset, "SalesPrice","ResidenceArea")
generic.data.visualisations("density", dataset, "SalesPrice1","ResidenceArea")
generic.data.visualisations("two.line", dataset, "TransactionDate","SalesPrice")




# library(Metrics)
# target <- c(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20)
# pred   <- c(0.02,0.56,1.22,1.98,3.95,5.05,5.95,6.95,8.56,8.99,9.65
#             ,10.55,11.52,12.35,13.95,14.86,15.66,16.88,17.45,18.99,19.01)
# 
# for(i in seq(0,1,0.1))
# {
#   cat("i value : " , i,"\n")
#   kappa <- ScoreQuadraticWeightedKappa(target, round(pred+i), 0, 20)
#   cat("kappa : ", kappa,"\n")
# }
# 
# source <- runif(100, min=0, max=20)
# target <- as.integer(source)
# pred   <- source + runif(100, min=0, max=1)
# 
# for(i in seq(0,1,0.1))
# {
#   cat("i value : " , i,"\n")
#   kappa <- ScoreQuadraticWeightedKappa(target, round(pred+i), 0, 20)
#   cat("kappa : ", kappa,"\n")
# }


######################################################################################################## 
#clustering
train.aggregated <- ddply(train.all, c("CustomerID","Age","ResidenceArea"), function(x) colSums(x[c("Amount", "SalesPrice")]))

m=as.matrix(cbind(train.aggregated$Amount, train.aggregated$SalesPrice),ncol=2)

cl=(kmeans(m,3))
cl$size
cl$withinss
head(train.aggregated)
train.aggregated$cluster=factor(cl$cluster)
centers=as.data.frame(cl$centers)

ggplot(train.aggregated, aes(x=Amount, y=SalesPrice, color=cluster )) +
  geom_point() +
  geom_point(data=centers, aes(x=V1,y=V2, color='Center')) +
  geom_point(data=centers, aes(x=V1,y=V2, color='Center'), size=20, alpha=.3)+
  labs(title ="Sales Clusters", x = "Amount", y = "Sales")

table(train.aggregated$cluster, train.aggregated$Age)
table(train.aggregated$cluster, train.aggregated$ResidenceArea)
########################################################################################################



