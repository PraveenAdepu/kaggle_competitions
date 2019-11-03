sample_sub <- read_csv("./input/sample_submission.csv")

head(sample_sub)
sample_sub$MultipolygonWKT <- NULL

Class1 <- read_csv("./submissions/Prav_sub05_Class1.csv")
Class2 <- read_csv("./submissions/Prav_sub05_Class2.csv")
Class3 <- read_csv("./submissions/Prav_sub05_Class3.csv")
Class4 <- read_csv("./submissions/Prav_sub05_Class4.csv")
Class5 <- read_csv("./submissions/Prav_sub05_Class5.csv")
Class6 <- read_csv("./submissions/Prav_sub05_Class6.csv")
Class7 <- read_csv("./submissions/Prav_sub05_Class7.csv")
Class8 <- read_csv("./submissions/Prav_sub05_Class8.csv")
Class9 <- read_csv("./submissions/Prav_sub05_Class9.csv")
Class10 <- read_csv("./submissions/Prav_sub05_Class10.csv")

Class1_sub <- Class1[Class1$ClassType==1,]
Class2_sub <- Class2[Class2$ClassType==2,]
Class3_sub <- Class3[Class3$ClassType==3,]
Class4_sub <- Class4[Class4$ClassType==4,]
Class5_sub <- Class5[Class5$ClassType==5,]
Class6_sub <- Class6[Class6$ClassType==6,]
Class7_sub <- Class7[Class7$ClassType==7,]
Class8_sub <- Class8[Class8$ClassType==8,]
Class9_sub <- Class9[Class9$ClassType==9,]
Class10_sub <- Class10[Class10$ClassType==10,]


All_sub <- rbind(Class1_sub, 
                 Class2_sub, 
                 Class3_sub, 
                 Class4_sub, 
                 Class5_sub, 
                 Class6_sub, 
                 Class7_sub, 
                 Class8_sub, 
                 Class9_sub, 
                 Class10_sub)

Allsub_Sorted <- left_join(sample_sub, All_sub, by=c("ImageId","ClassType"))

head(Allsub_Sorted)

write_csv(Allsub_Sorted,"./submissions/Prav_sub05_AllClassv3.csv")

