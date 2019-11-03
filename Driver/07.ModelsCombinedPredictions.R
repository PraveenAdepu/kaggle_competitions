

OOF_Predictions <- OOF_ScriptQty

head(OOF_Predictions)
head(OOF_BreakFast)
OOF_Predictions <- left_join(OOF_Predictions, OOF_BreakFast[,c("ID","BreakFast","BreakFastProbability")],by="ID")

OOF_Predictions <- left_join(OOF_Predictions, OOF_Lunch[,c("ID","Lunch","LunchProbability")],by="ID")


OOF_Predictions <- left_join(OOF_Predictions, OOF_Dinner[,c("ID","Dinner","DinnerProbability")],by="ID")

OOF_Predictions <- left_join(OOF_Predictions, OOF_BedTime[,c("ID","BedTime","BedTimeProbability")],by="ID")

head(OOF_Predictions)

