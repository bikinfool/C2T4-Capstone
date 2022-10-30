#*
#*
#*
#*Load Packages as needed.
#

library(doParallel)
library(randomForest)
library(caret)
library(tidyverse)
library(dplyr)
library(plotly)
library(ggplot2)
library(corrplot)
library(beepr)

song <- function() {  
  for (i in 1:10) {  
    beep()   
    Sys.sleep (0.25)  }
  beep(4)
}

#* Increase computing power, add additional cores.
detectCores() 
cl <- makeCluster(3)
registerDoParallel(cl)
getDoParWorkers() # Result: should be 3

# Stop Cluster. After performing your tasks, stop your cluster. 
stopCluster(cl)

# read in data to grid_df
grid_df <- read.csv ("C:\\Users\\johne\\Documents\\Purdue DA\\Final Project\\Grid stability\\Data_for_UCI_named.csv")
grid_df[,14] <- as.factor(grid_df[,14]) 
str(grid_df)
summary(grid_df)

#* run Correlation matrix and plot first 13 attributes. 
#* 14 is a factor and cannot be included
cormatrix <- cor(grid_df[,1:13])
cormatrix
corrplot(cormatrix, method = "square")

# This can be used to check just the one column correlation to the dependent variable
cormatrix <- cor(grid_df[,1:12],grid_df[,13])

#***************************************************************************
#*Can we reduce size of data frame based on  low StDev, NZV or RFE?
#*    Check Standard deviation of each row, then column 
#*    Create new variable x that is just the independent variables.
x<-grid_df[,1:12]
grid_df_min_sd_row <- min(apply(x, 1, sd))
grid_df_min_sd_col <- min(apply(x, 2, sd))
#*    Check with NZV if we can reduce size due to no variation.
nzvgrid <- nearZeroVar(grid_df, saveMetrics = TRUE) 
nzvgrid

# Set up rfeControl to see if some variables might be eliminated.
grid_train_rfe <- grid_df[sample(1:nrow(grid_df), 1000, replace=FALSE),]

ctrl <- rfeControl(functions = rfFuncs, 
                   method = "repeatedcv",
                   repeats = 3,
                   verbose = FALSE)
#* Use rfe and seperate the dependent variables (13/14) (attributes 1-12) 
#* This function was run with both "stab"(num) and "stabf"(factor) variable.
#* 
rfe_res_grid <- rfe(grid_train_rfe[,1:12], 
                    grid_train_rfe$stabf, 
                      sizes=(1:8), 
                      rfeControl=ctrl)

rfe_res_grid
plot(rfe_res_grid, type=c("g", "o"))
#****** List all optimal variagle to seperate later
rfe_res_grid$optVariables
#*
#*
#******************************************************
#*Build new df 8+1 x 10000, in order of most to least significant
grid_df_2 <- grid_df[,c(1,3,2,4,11,9,12,10,14)]
str(grid_df_2)


#***************************************************************
#*   Caret Machine learning section
# create test and train data set (20/80) for use with caret
#*
#*

set.seed(123)
IndexofTrainRows <- createDataPartition(grid_df_2$stabf, p = .80, list = FALSE)
trn_grid2 <- grid_df_2[IndexofTrainRows,]
tst_grid2 <- grid_df_2[-IndexofTrainRows,]
str(trn_grid2)
str(tst_grid2)

#****************************************
##### c50 method
starttime <- Sys.time()
use_c50_grd <- expand.grid( .winnow = c(TRUE,FALSE), .trials=c(1,5,10,15,20,25), .model="tree" )
ftCtrl <- trainControl(method = "repeatedcv", number = 10, repeats = 10)
c50_grid2 <- train(stabf~.,data = trn_grid2,
                      method = "C5.0",
                      trControl=ftCtrl,
                      tuneGrid=use_c50_grd,
                      verbose=FALSE)
c50_grid2
ggplot(c50_grid2)

stoptime <- Sys.time()
c("C-50 Run time was:", stoptime - starttime)
c50predict <- predict(c50_grid2, tst_grid2)
head(c50predict)

confusionMatrix(tst_grid2$stabf, c50predict)
#*
#*
#*Generate final data file from original data frame and the c50 predict information
#*write file to drive for submission
#*
grid_data_final <- grid_df[,c(1:12,14)]
grid_data_final$predict <- c50predict
write.csv (grid_data_final, "C:\\Users\\johne\\Documents\\Purdue DA\\Final Project\\Grid stability\\C2T4capstoe_data_final.csv")




#*******************************************
##### Random Forest method

starttime <- Sys.time()
use_rf_grd2 <- expand.grid(mtry=c(1,5,10,15,20,25))
ftCtrl2 <- trainControl(method = "repeatedcv", number = 10, repeats = 10)
rf_grid2 <- train(stabf~.,data = trn_grid2,
                     method = "rf",
                     trControl=ftCtrl2,
                     tuneGrid=use_rf_grd2)
rf_grid2
stoptime <- Sys.time()

######******** play
song()

######********
c("rf Run time was:", stoptime - starttime)

ggplot(rf_grid2)

rfpredict <- predict(rf_grid2, tst_grid2 )
head(rfpredict)

confusionMatrix(tst_grid2$stabf, rfpredict)


#*******************************************
##### SVM  method
starttime <- Sys.time()
svm_grid2 <- train(stabf~.,data = trn_grid2,
                      method = "svmLinear2",
                      trControl = trainControl("cv", number = 10, verboseIter = TRUE),
                      preProcess =c('center', 'scale'))

svm_grid2

stoptime <- Sys.time()
c("SVMLinear2 Run time was:", stoptime - starttime)
ggplot (svm_grid2)
svmpredict <- predict(svm_grid2, tst_grid2)
head(svmpredict)

confusionMatrix(tst_grid2$stabf, svmpredict)



