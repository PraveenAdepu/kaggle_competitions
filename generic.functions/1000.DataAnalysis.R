###################################################################################################
# Read source files
###################################################################################################
train11 <- read.table("./input/D11", sep=";", header=T, stringsAsFactors = FALSE)
train12 <- read.table("./input/D12", sep=";", header=T, stringsAsFactors = FALSE)
train01 <- read.table("./input/D01", sep=";", header=T, stringsAsFactors = FALSE)
train02 <- read.table("./input/D02", sep=";", header=T, stringsAsFactors = FALSE)

###################################################################################################
# First data inspection
###################################################################################################

names(train11)
head(train11)

###################################################################################################
# rbind to form one dataset
###################################################################################################

train.all <- rbind(train11, train12, train01, train02)
dim(train.all)
# Total dataset dim = 817741 * 9 dim

###################################################################################################
# Update column names to English
# Reference from http://stackoverflow.com/questions/25014904/download-link-for-ta-feng-grocery-dataset
###################################################################################################


column.names <- c("TransactionDateTime",
                  "CustomerID",
                  "Age",
                  "ResidenceArea",
                  "ProductSubclass",
                  "ProductID",
                  "Amount",
                  "Asset",
                  "SalesPrice")

names(train.all)
names(train.all) <- column.names
names(train.all)

train.all$TransactionDate <- as.POSIXct(train.all$TransactionDateTime)
train.all$Day             <- weekdays(as.Date(train.all$TransactionDate))
train.all$WeekEnd         <- weekdays(as.Date(train.all$TransactionDate)) %in% c('Saturday','Sunday')


# Use TransactionDate inplace of TransactionDateTime 
str(train.all)


##########################################################################################################
theme_update(plot.title = element_text(hjust = 0.5))
##########################################################################################################

generic.data.visualisations("discrete.bar", train.all, "Age")

ggplot(train.all, aes(x=TransactionDate, y=SalesPrice))+ 
  stat_summary(geom="ribbon", fun.data=mean_cl_normal ,
               fun.args=list(conf.int=0.95), fill="lightblue")+
  stat_summary(geom="line", fun.y=mean, linetype="dashed")+
  stat_summary(geom="point", fun.y=mean, color="red")+
  labs(title = "SalesPrice (mean) with 95% confidence by Date  ")

ggplot(train.all, aes(x=TransactionDate, y=Amount))+ 
  stat_summary(geom="ribbon", fun.data=mean_cl_normal ,
               fun.args=list(conf.int=0.95), fill="lightblue")+
  stat_summary(geom="line", fun.y=mean, linetype="dashed")+
  stat_summary(geom="point", fun.y=mean, color="red")+
  labs(title = "Amount (mean) with 95% confidence by Date  ")

ggplot(train.all, aes(as.factor(Age), SalesPrice)) + 
  stat_summary(fun.y = mean, geom = "bar") + 
  stat_summary(fun.data = mean_se, geom = "errorbar", colour = "red") +
  labs(title = "SalesPrice (mean) by Age and Errors ", x="Age")

ggplot(train.all, aes(as.factor(ResidenceArea), SalesPrice)) + 
  stat_summary(fun.y = mean, geom = "bar") + 
  stat_summary(fun.data = mean_se, geom = "errorbar", colour = "red")+
  labs(title = "SalesPrice (mean) by ResidenceArea and Errors ", x="ResidenceArea")

##########################################################################################################
# Start - Outlier Analysis ###############################################################################
##########################################################################################################
Sales <- ggplot(train.all, aes(TransactionDate, SalesPrice))
Sales <- Sales + geom_line()
Sales <- Sales + labs(title ="Date vs Sales", x = "Date", y = "Sales")
Sales
#Outcome: Extreem x-axis values, requires outlier investigation

SalesPrice <- ggplot(train.all, aes(SalesPrice))
SalesPrice <- SalesPrice + geom_density() 
SalesPrice <- SalesPrice + labs(title ="SalesPrice density")
SalesPrice
#Outcome: skewness, requires outlier investigation
##########################################################################################################

Amount <- ggplot(train.all, aes(TransactionDate, Amount))
Amount <- Amount + geom_line()
Amount <- Amount + labs(title ="Date vs Amount", x = "Date", y = "Amount")
Amount

Amount <- ggplot(train.all, aes(Amount))
Amount <- Amount + geom_density() 
Amount <- Amount + labs(title ="Amount density")
Amount
#Outcome: skewness, requires outlier investigation
##########################################################################################################

# skewness.columns = c("SalesPrice","Amount")
# 
# for (f in skewness.columns) {
# 
#     lambda   = BoxCox.lambda( train.all[[f]] )
#     skewness = skewness( train.all[[f]] )
#     kurtosis = kurtosis( train.all[[f]] )
#     cat("VARIABLE : ",f, "lambda : ",lambda, "skewness : ",skewness, "kurtosis : ",kurtosis, "\n")
# 
# }
# VARIABLE :  SalesPrice lambda :  -0.2366337 skewness :  441.9793 kurtosis :  300488.4 
# VARIABLE :  Amount     lambda :  -0.8052557 skewness :  260.6988 kurtosis :  93721.68 
##########################################################################################################

# Apply log transformations to check distributions

SalesPrice <- ggplot(train.all, aes(log(SalesPrice)))
SalesPrice <- SalesPrice + geom_density() 
SalesPrice <- SalesPrice + labs(title ="Log(SalesPrice) density")
SalesPrice

Amount <- ggplot(train.all, aes(log(Amount)))
Amount <- Amount + geom_density()
Amount <- Amount + labs(title ="Log(Amount) density", x = "log(Amount)")
Amount

quantile(train.all$Amount)
quantile(train.all$Amount, 0.99)

Amount <- ggplot(subset(train.all, Amount <= quantile(train.all$Amount, 0.99)) , aes(Amount))
Amount <- Amount + geom_density()
Amount <- Amount + labs(title ="Amount density - quantile 0.99", x = "Amount")
Amount


SalesPrice <- ggplot(train.all, aes(log(SalesPrice), colour = Age))
SalesPrice <- SalesPrice + geom_density()
SalesPrice <- SalesPrice + labs(title ="log(SalesPrice) distribution by Age", x = "log(SalesPrice)")
SalesPrice


SalesPrice <- ggplot(train.all, aes(log(SalesPrice), colour = ResidenceArea))
SalesPrice <- SalesPrice + geom_density()
SalesPrice <- SalesPrice + labs(title ="log(SalesPrice) distribution by ResidenceArea",x = "log(SalesPrice)")
SalesPrice

ggplot(train.all, aes(x=Amount, y=SalesPrice)) +
  geom_point(shape=1) +    
  geom_smooth(method=lm) + labs(title = "Sales vs Amount") 


ggplot(subset(train.all, Amount <= 250), aes(x=Amount, y=SalesPrice)) +
  geom_point(shape=1) +   
  geom_smooth(method=lm) + labs(title = "Sales vs (Amount <= 250)")  

# Check all highAmount and Sales records

train.highSalesAmount <- subset(train.all,  Amount > 250)
head(train.highSalesAmount,10)

# We got 8 outlier records
#Outcome : Not sure of any public holiday or fetival season and need third party data for further validation

summary(lm(log(SalesPrice) ~ factor(Age), data = train.all))
summary(lm(log(SalesPrice) ~ factor(ResidenceArea), data = train.all))
summary(lm(log(SalesPrice) ~ TransactionDate, data = train.all))
##########################################################################################################
# End Outlier Analysis ###################################################################################
##########################################################################################################

##########################################################################################################
# Wholesale Transactions Analysis ########################################################################
##########################################################################################################
# Filter high Sales and Amount records
head(train.highSalesAmount,10)
##########################################################################################################


##########################################################################################################
# Retail Transactions Analysis ###########################################################################
##########################################################################################################
Sales <- ggplot(train.all, aes(x = factor(Day, weekdays(min(as.Date(train.all$TransactionDate)) + 0:6)),SalesPrice))
Sales <- Sales + geom_bar(stat = "identity") 
Sales <- Sales + labs(title ="Sales by Day", x = "Day", y = "Sales")  
Sales 

Age <- ggplot(train.all, aes(factor(Age), fill = Day))
Age <- Age + geom_bar() 
Age <- Age + labs(title ="Transactions by Age by Day", x = "Age", y = "Transactions Count")  
Age 

ResidenceArea <- ggplot(train.all, aes(factor(ResidenceArea), fill = Day))
ResidenceArea <- ResidenceArea + geom_bar() 
ResidenceArea <- ResidenceArea + labs(title ="Transactions by ResidenceArea by Day", x = "ResidenceArea", y = "Transactions Count")  
ResidenceArea 

# ProductSubClass - High dimensionality 
ProductSubclass <- ggplot(train.all, aes(factor(ProductSubclass)))
ProductSubclass <- ProductSubclass + geom_bar() 
ProductSubclass <- ProductSubclass + geom_density() 
ProductSubclass <- ProductSubclass + labs(title ="Transactions by ProductSubclass", x = "ProductSubclass", y = "Transactions Count")  
ProductSubclass 


Sales <- ggplot(train.all, aes(TransactionDate, SalesPrice)) 
Sales <- Sales + stat_summary(fun.y = sum,geom = "line",colour = "blue") 
Sales <- Sales + labs(title ="Date vs Sales", x = "Date", y = "Sales")
Sales

##########################################################################################################
##########################################################################################################

summary(lm(log(SalesPrice) ~ factor(Age), data = train.all))
summary(lm(log(SalesPrice) ~ TransactionDate, data = train.all))

######################################################################################################## 
#Forecasting

train.aggregated <- ddply(train.all, c("TransactionDate"), function(x) colSums(x[c("Amount", "SalesPrice")]))
t.test(log(train.aggregated$SalesPrice))
head(train.aggregated)

ggplot(train.aggregated, aes(x=Amount  , y= SalesPrice )) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "lm", formula = y ~ x) +
  stat_fit_glance(method = 'lm',
                  method.args = list(formula =  y ~ x),
                  geom = 'text',
                  aes(label = paste("P-value = ", signif(..p.value.., digits = 4), sep = "")),
                  label.x.npc = 'right', label.y.npc = 0.35, size = 3)

amount.fit      <- ets(ts(train.aggregated$Amount, frequency =  7))
amount.forecast <- forecast(amount.fit, h=10)
plot(amount.forecast)

amount.fit      <- auto.arima(ts(train.aggregated$Amount, frequency =  7))
amount.forecast <- forecast(amount.fit, h=10)
plot(amount.forecast)


amount.fit      <- ets(ts(train.aggregated$SalesPrice, frequency =  7))
amount.forecast <- forecast(amount.fit, h=10)
plot(amount.forecast)

######################################################################################################## 

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
# Association

train.aggregated <- ddply(train.all, c("Age","WeekEnd","Day","ResidenceArea"), function(x) colSums(x[c("Amount", "SalesPrice")]))


train.aggregated$Age           <- as.factor(train.aggregated$Age)
train.aggregated$WeekEnd       <- as.factor(train.aggregated$WeekEnd)
train.aggregated$ResidenceArea <- as.factor(train.aggregated$ResidenceArea)
train.aggregated$Day           <- as.factor(train.aggregated$Day)
train.aggregated$SaleMoreThan150 <- as.factor(ifelse(train.aggregated$SalesPrice > 150, "Yes","No"))
train.aggregated$SaleMoreThan1000 <- as.factor(ifelse(train.aggregated$SalesPrice > 1000, "Yes","No"))
str(train.aggregated[,c("Age","WeekEnd","Day","ResidenceArea","SaleMoreThan150","SaleMoreThan1000")])
rules <- apriori(train.aggregated[,c("Age","Day","ResidenceArea","SaleMoreThan150","SaleMoreThan1000")])
rules.sorted <- sort(rules, by="lift")
inspect(rules.sorted)
plot(rules, method="graph", control=list(type="items"))
########################################################################################################

