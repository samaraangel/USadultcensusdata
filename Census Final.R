##########################################################
#  Final Project 
# Harvard EdX Professional Data Science Certificate 
# Samara Angel 
# Spring 2021
##########################################################

##########################################################
# Data Pre-Processing and Cleaning
##########################################################
# Begin by automatically attaching any necessary packages
# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(ggrepel)) install.packages("ggrepel", repos = "http://cran.us.r-project.org")
if(!require(dslabs)) install.packages("dslabs", repos = "http://cran.us.r-project.org")
if(!require(readr)) install.packages("readr", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(kableExtra)) install.packages("kableExtra", repos = "http://cran.us.r-project.org")
if(!require(colorspace)) install.packages("colorspace", repos = "http://cran.us.r-project.org")
if(!require(mltools)) install.packages("mltools", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(formatR)) install.packages("formatR", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")
if(!require(rattle)) install.packages("rattle", repos = "http://cran.us.r-project.org")

library(corrplot)
library(tidyverse)
library(caret)
library(data.table)
library(dplyr)
library(lubridate)
library(ggplot2)
library(ggrepel)
library(dslabs)
library(readr)
library(knitr)
library(kableExtra)
library(colorspace)
library(mltools)
library(rpart)
library(formatR)
library(e1071)
library(rattle)

#I begin by importing the data from UCI Machine Learning Repository. The data set is the Census Income Data Set.  
basic <- read.table("https://query.data.world/s/ofxhkosdcjihnprr3lctlxwzw44645")
head(basic)
nrow(basic)
ncol(basic)

#I then attach the column names to the data set
colnames(basic) <- c("age", "workclass", "fnlwgt", "education", "education_num","marital_status","occupation","relationship", "race", "sex", "capital_gain", "capital_loss", 
                     "hours_per_week", "native_country", "income")
head(basic)

#Each entry in the data set currently ends with a comma, so I remove that comma. 
data <- basic %>% mutate(age = gsub(",","",basic$age), workclass = gsub(",","",basic$workclass), fnlwgt = gsub(",","",basic$fnlwgt), education = gsub(",","",basic$education), 
                         education_dummy = gsub(",","",basic$education_num), marital_status = gsub(",","",basic$marital_status), occupation = gsub(",","",basic$occupation), 
                         relationship = gsub(",","",basic$relationship), race = gsub(",","",basic$race), sex = gsub(",","",basic$sex), capital_gain = gsub(",","",basic$capital_gain), 
                         capital_loss = gsub(",","",basic$capital_loss), hours_per_week = gsub(",","",basic$hours_per_week), native_country = gsub(",","",basic$native_country))
data

#For simplicity, I then delete any rows with missing data, demarcated in this data set by "?" to create the final data set. 
dataset <- subset(data, data$age !="?" & data$workclass !="?" & data$fnlwgt !="?" & data$education !="?" & data$education_num !="?" & data$marital_status !="?" & 
                    data$occupation !="?" & data$relationship !="?" & data$race !="?" & data$sex !="?" & data$capital_gain !="?" & data$capital_loss !="?" & data$native_country !="?" & 
                    data$income !="?") 
head(dataset)
nrow(dataset)

#To make sure education level isn't impacted by an age restriction, I restrict age to higher than 18
prep_set <- dataset %>% filter(age>=18)
head(prep_set)
nrow(prep_set)
str(prep_set)

#Here I take away variables that aren't relevant. 
#To make the data function the way I need it to for the generalized linear model test, I also reclassify the remaining variables as factors. 
prep_set_2 <- prep_set %>% 
  mutate(age = as.factor(age), fnlwgt = as.numeric(fnlwgt), race=as.factor(race), sex = as.factor(sex), education_dummy= as.factor(education_dummy), 
         marital_status = as.factor(marital_status), occupation = as.factor(occupation), hours_per_week = as.factor(hours_per_week), income = as.factor(income)) %>%
  select(-workclass, -native_country, -relationship, -capital_gain, -capital_loss, -relationship, -education_num, -hours_per_week) %>%
  mutate(education_dummy = ordered(education_dummy, levels = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)))
str(prep_set_2)

#Here, I complete one-hot encoding using the dummyVars function to recode variables as numeric without ranking. 
dummy <- dummyVars(" ~sex", data=prep_set_2)
prep_set_3 <- prep_set_2 %>% mutate(data.frame(predict(dummy, newdata = prep_set_2)))

dummy <- dummyVars(" ~marital_status", data=prep_set_3)
prep_set_4 <- prep_set_3 %>% mutate(data.frame(predict(dummy, newdata = prep_set_3)))

dummy <- dummyVars(" ~occupation", data=prep_set_4)
prep_set_5 <- prep_set_4 %>% mutate(data.frame(predict(dummy, newdata = prep_set_4)))

dummy <- dummyVars(" ~race", data=prep_set_5)
prep_set_6 <- prep_set_5 %>% mutate(data.frame(predict(dummy, newdata = prep_set_5)))

prep_set_factor <- prep_set_6 %>% mutate(income_dummy = ifelse(.$income == "<=50K", 0, 1))
head(prep_set_factor)
str(prep_set_factor)

#Here, I recode variables as numeric for use in other models. I remove the factor-coded variables. 
prep_set_numeric <- prep_set_factor %>% mutate(age = as.numeric(.$age), education_dummy = as.numeric(education_dummy)) %>% select(-marital_status) %>% 
  select(-occupation) %>% select(-race) %>% select(-sex) %>% select(-income) %>% select(-education)
head(prep_set_numeric)
str(prep_set_numeric)

#I then split this into a training set and a final test set.
# The test set will be 20% of two prep_sets I made above. I use income to standardize analysis.
#I do this twice, once for the factor-based prep_set_factor, and once for the numeric-based prep_set_numeric. 

#Here I partition the factor set. 
set.seed(1)
test_index <- createDataPartition(y = prep_set_factor$income, times = 1, p = 0.2, list = FALSE)
train_factor <- prep_set_factor[-test_index,]
head(train_factor)
str(train_factor)
test_factor <- prep_set_factor[test_index,]

#And here I partition the numeric set using the same index as above. 
set.seed(1)
train_numeric <- prep_set_numeric[-test_index,]
head(train_numeric)
str(train_numeric)
test_numeric <- prep_set_numeric[test_index,]

#Although separately partitioned and coded as factors vs. numeric, the two sets are identical in what their variables demonstrate. 

##############################
# EXAMINING THE DATA
##############################
#examine dataset 
head(train_factor)
str(train_factor)

head(test_factor)
str(test_factor)

head(train_numeric)
str(train_numeric)

head(test_numeric)
str(test_numeric)

#determine number of rows and columns
nrow(train_factor)
ncol(train_factor)

nrow(train_numeric)
ncol(train_numeric)

###################################
#Data Visualization
###################################
#Show the head of the train_factor data
trying <- train_factor[1:10]
trying1 <- head(trying, 10)
kbl(trying1, caption = "Head of Train Factor Data Set") %>% 
  kable_styling(latex_options = c("hold_position", "scale_down", "striped"))

#Show the head of the test_factor data 
try <- test_factor[1:10]
trying2 <- head(try, 10)
kbl(trying2, caption = "Head of Test Factor Data Set") %>% 
  kable_styling(latex_options = c("hold_position", "scale_down", "striped"))

#Show the head of the train_numeric data
try1 <- train_numeric[1:9]
trying3 <- head(try1, 10)
kbl(trying3, caption = "Head of Train Numeric Data Set") %>% 
  kable_styling(latex_options = c("hold_position", "scale_down", "striped"))

#Show the head of the test_numeric data 
try2 <- test_numeric[1:9]
trying4 <- head(try2, 10) 
kbl(trying4, caption = "Head of Test Numeric Data Set") %>% 
  kable_styling(latex_options = c("hold_position", "scale_down", "striped"))

#Number of rows and columns in the train_factor data 
littletable <- matrix(c(nrow(train_numeric), ncol(train_numeric), nrow(train_factor), ncol(train_factor)), ncol=2, byrow=TRUE)
colnames(littletable) <- c("Number of Rows", "Number of Columns")
rownames(littletable) <- c("Numeric", "Factor")
littletable %>% kbl(caption = "Number of Rows and Columns in the Train Numeric and Train Factor Data Sets") %>%
  kable_styling(latex_options = c("hold_position", "striped"))

#As you can see, the difference between the train_numeric vs. train_factor and test_numeric vs. test_factor simply is that the numeric train and 
#test set are missing the factor variables (that have not been one-hot encoded). The factor sets have fewer columns, but both sets have the same number of rows.
#The two sets can therefore be used interchangeably depending if a factor or numeric set is required. 

#graph of the count vs. income 
train_factor %>%
  group_by(income) %>% 
  ggplot(aes(x = income, fill = I("aquamarine4"))) + ggtitle("Count vs. Income Level Above or Below 50k") +
  xlab("Income Level") + 
  ylab("Count") + 
  geom_bar()

#We can see that the data set has more people with income <=50K than >50K. This may be a factor of our data, but also makes sense given the wealth distribution in the United States. 
x <- as.data.frame(table(train_factor$income))
percentage_high_income <- x$Freq[1]/sum(x$Freq)
percentage_high_income %>% kbl(caption = "Percentage of Individals with Income Over 50K") %>%
  kable_styling(latex_options = c("hold_position", "striped"))

#Here, we can see a graph of the count vs. education_dummy variable. There are more people with very few or very many years of education than those in between. 
train_numeric %>%
  group_by(education_dummy) %>% 
  summarize(count = n()) %>% 
  ggplot(aes(x = education_dummy, y = count, color = I("aquamarine4"))) + ggtitle("Count vs. Number of Years of Education") +
  xlab("Number of Years of Education") + 
  ylab("Count") + 
  geom_point() + geom_line()

#Looking at the relationship between years of education and income, people with 14-16 years (a college level) of education have more individuals 
#making >50K than individuals making <=50K with 56%, 75%, and 75% respectively. People with 13 years (~one year of college) have 42% making >50K, and 
#people with 9-12 years of education (a high school level) are close to the 20% range of people making more than 50K. People with between 2 and 8 years 
#of education (an elementary and middle school education level) have around 5% making more than 50K, and people with only 1 year of education have no individuals making more than 50K.
train_factor %>% 
  ggplot(aes(x=education_dummy, y=..count.., fill=income))+ 
  geom_bar(aes(fill=income), position = "dodge") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  geom_text(stat='count', aes(label =..count..), color = "black", angle=90, position = position_dodge(0.8), size=1.8, hjust=0) + 
  ggtitle("Count vs. Number of Years of Education By Income") + scale_fill_manual(values = c("aquamarine4","maroon3"))

#Math calculations for the above:
220/(220+73)
341/(341+110)
742/(742+574)
1683/(1683+2350)
191/(191+610)
272/(272+762)
1085/(1085+4280)
1291/(1291+6554)
20/(20+251)
48/(48+673)
49/(49+508)
20/(20+331)
28/(28+422)
10/(10+218)
6/(6+112)
32/(0+32)

#Here, I examine the relation between race and income. 
train_factor %>% 
  ggplot(aes(x=race, y=..count.., fill=income)) +
  geom_bar(aes(fill = income), position = "dodge") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  geom_text(stat='count', aes(label =..count..), color = "black", vjust=-0.2, position = position_dodge(0.9), size=3) + 
  ggtitle("Count vs. Race By Income") + scale_fill_manual(values = c("aquamarine4","maroon3"))

#Here, across race more people are making less than 50K than more than 50K. 
#26% of white people and 28% of Asian/Pacific Islanders made >50K,  while only 13% of Black people, 12% of American-Indian/Eskimo people, and 10% of other made >50K. 
#Math calculations for the above:
5466/(5466+15039)
18/(18+168)
292/(292+1955)
201/(201+502)
29/(196+29)

#Here, I examine the relation between sex and income. 
train_factor %>% 
  ggplot(aes(x=sex,y=..count.., fill=income)) +
  geom_bar(aes(fill = income), position = "dodge") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  geom_text(stat='count', aes(label =..count..), color = "black", vjust=-0.5, position = position_dodge(0.9), size=3) + 
  ggtitle("Count vs. Sex By Income") + scale_fill_manual(values = c("aquamarine4","maroon3"))

#Here, we can see that for both genders, the number of people making less than 50K is higher than those making more than 50K. However, visually it appears that about 1/3 of 
#men make more than 50K, while only around 1/10 of women make more than 50K. 
#Math calculations for the above: 
5123/(5123+11016)
883/(883+6844)

#Here, I examine the relation between marital status and income.For those who are widowed, separated, never married, were married but have an absent spouse, or are divorced, 
#the percent of people making >50K is between about 5 and 10 percent. For people married, whether to a civilian or armed fources spouse, about 45% of people have income >50K. 
train_factor %>% 
  ggplot(aes(x=marital_status,y=..count.., fill=income)) +
  geom_bar(aes(fill = income), position = "dodge") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  geom_text(stat='count', aes(label =..count..), color = "black", vjust=-0.2, position = position_dodge(0.9), size=3) + 
  ggtitle("Count vs. Marital Status By Income") + scale_fill_manual(values = c("aquamarine4","maroon3"))

#Math calculations for the above: 
70/(70+595)
52/(52+704)
379/(379+7172)
22/(267)
5121/(5121+6113)
7/(7+9)
355/(355+3000)

######################################
#Models 
######################################
#Simplest algorithm is to guess the outcome which gives an accuracy of 50% as expected. Sensitivity and specificity are also around 50% 
y_hat <- sample(c("<=50K", ">50K"), nrow(train_factor), replace = TRUE) %>% factor(levels = levels(train_factor$income))
accuracy_guess <- confusionMatrix(data = y_hat, reference = train_factor$income)$overall["Accuracy"]
sensitivity_guess<- confusionMatrix(data = y_hat, reference = train_factor$income)$byClass["Sensitivity"]
specificity_guess <- confusionMatrix(data = y_hat, reference = train_factor$income)$byClass["Specificity"]
rmse_results <- tibble(method = "Guessing Model", Accuracy = accuracy_guess, Sensitivity = sensitivity_guess, Specificity = specificity_guess)
rmse_results %>% kbl(caption = "Accuracy, Specificity, Sensitivity Results - Including Guessing Model") %>%
  kable_styling(latex_options = c("hold_position", "striped"))
#This gives us an overall accuracy of 0.4970, with a sensitivity of 0.5011 and a specificity of 0.4848.

#Create a correlation plot using train_numeric to understand correlation between variables
col1 <-colorRampPalette(c("aquamarine4", "white", "maroon3"))

corrplot(cor(train_numeric), tl.cex = .4, tl.srt = 35, mar=c(0,0,1,0), tl.col = "black", cex.main = 1, title = "Correlation Plot Of Train Numeric Variables", col = col1(100)) 

#There appears to be a correlation between several of the variables: marital status and sex; sex and occupation; education and occupation; education and income; 
#marital status and income; and sex and income. Here, we run into a challenge of covariates and multicollinearity. Using correlated coviariates can be challenging. 
#For example, with education by number of years, it logically makes sense that the higher the number of years of education, the higher a person's income level. 
#However, because we have sliced income into a binary variable, once income is exceeding 50K we don't glean any more information frome education level. 
#With race, because there are 4 covariates (because there are five races recorded, and therefore four covariates), there is more information. 

#generalized linear model using only race.  
fit_glm <- glm(income ~ race,
               data=train_factor, family=binomial(link = logit))
p_hat_logistic <- predict(fit_glm, test_factor, type = "response")
y_hat_logistic <- factor(ifelse(p_hat_logistic > 0.25, "<=50K", ">50K")) %>% factor
accuracy_glm_race <- confusionMatrix(data = y_hat_logistic, reference = test_factor$income)$overall["Accuracy"]
sensitivity_glm_race<- confusionMatrix(data = y_hat_logistic, reference = test_factor$income)$byClass["Sensitivity"]
specificity_glm_race <- confusionMatrix(data = y_hat_logistic, reference = test_factor$income)$byClass["Specificity"]
rmse_results <- rmse_results %>% add_row(method = "GLM Race Model", Accuracy = accuracy_glm_race, Sensitivity = sensitivity_glm_race, Specificity = specificity_glm_race)
rmse_results %>% kbl(caption = "Accuracy, Specificity, Sensitivity Results - Including GLM Race Model") %>%
  kable_styling(latex_options = c("hold_position", "striped"))
#This gives us an overall accuracy of 0.6711, with a sensitivity of 0.8766 and a specificity of 0.0599.

#generalized linear model using only sex
fit_glm <- glm(income ~ sex,
               data=train_factor, family="binomial")
p_hat_logistic <- predict(fit_glm, test_factor, type = "response")
y_hat_logistic <- factor(ifelse(p_hat_logistic > 0.25, "<=50K", ">50K")) %>% factor
accuracy_glm_sex <- confusionMatrix(data = y_hat_logistic, reference = test_factor$income)$overall["Accuracy"]
sensitivity_glm_sex<- confusionMatrix(data = y_hat_logistic, reference = test_factor$income)$byClass["Sensitivity"]
specificity_glm_sex <- confusionMatrix(data = y_hat_logistic, reference = test_factor$income)$byClass["Specificity"]
rmse_results <- rmse_results %>% add_row(method = "GLM Sex Model", Accuracy = accuracy_glm_sex, Sensitivity = sensitivity_glm_sex, Specificity = specificity_glm_sex)
rmse_results %>% kbl(caption = "Accuracy, Specificity, Sensitivity Results - Including GLM Sex Model") %>%
  kable_styling(latex_options = c("hold_position", "striped"))
#This gives us an overall accuracy of 0.4950, with a sensitivity of 0.6158 and a specificity of 0.1358.

#generalized linear model using only education
fit_glm <- glm(income ~ education_dummy,
               data=train_factor, family="binomial")
fitted <- predict(fit_glm) #linear predictor roughly proportional to the probabilities
p_hat_logistic <- predict(fit_glm, test_factor, type = "response")
y_hat_logistic <- factor(ifelse(p_hat_logistic > 0.25, "<=50K", ">50K")) %>% factor
accuracy_glm_edu <- confusionMatrix(data = y_hat_logistic, reference = test_factor$income)$overall["Accuracy"]
sensitivity_glm_edu<- confusionMatrix(data = y_hat_logistic, reference = test_factor$income)$byClass["Sensitivity"]
specificity_glm_edu <- confusionMatrix(data = y_hat_logistic, reference = test_factor$income)$byClass["Specificity"]
rmse_results <- rmse_results %>% add_row(method = "GLM Education Model", Accuracy = accuracy_glm_edu, Sensitivity = sensitivity_glm_edu, Specificity = specificity_glm_edu)
rmse_results %>% kbl(caption = "Accuracy, Specificity, Sensitivity Results - Including GLM Education Model") %>%
  kable_styling(latex_options = c("hold_position", "striped"))
#This gives us an overall accuracy of 0.2840, with a sensitivity of 0.2203 and a specificity of 0.4734.

#generalized linear model using only marital status
fit_glm <- glm(income ~ marital_status,
               data=train_factor, family="binomial")
p_hat_logistic <- predict(fit_glm, test_factor, type = "response")
y_hat_logistic <- factor(ifelse(p_hat_logistic > 0.25, "<=50K", ">50K")) %>% factor
accuracy_glm_ms <- confusionMatrix(data = y_hat_logistic, reference = test_factor$income)$overall["Accuracy"]
sensitivity_glm_ms<- confusionMatrix(data = y_hat_logistic, reference = test_factor$income)$byClass["Sensitivity"]
specificity_glm_ms <- confusionMatrix(data = y_hat_logistic, reference = test_factor$income)$byClass["Specificity"]
rmse_results <- rmse_results %>% add_row(method = "GLM Marital Status Model", Accuracy = accuracy_glm_ms, Sensitivity = sensitivity_glm_ms, Specificity = specificity_glm_ms)
rmse_results %>% kbl(caption = "Accuracy, Specificity, Sensitivity Results - Including GLM Marital Status Model") %>%
  kable_styling(latex_options = c("hold_position", "striped"))
#This gives us an overall accuracy of 0.2889, with a sensitivity of 0.3395 and a specificity of 0.1385.

#generalized linear model using only occupation
fit_glm <- glm(income ~ occupation,
               data=train_factor, family="binomial")
p_hat_logistic <- predict(fit_glm, test_factor, type = "response")
y_hat_logistic <- factor(ifelse(p_hat_logistic > 0.25, "<=50K", ">50K")) %>% factor
accuracy_glm_occ <- confusionMatrix(data = y_hat_logistic, reference = test_factor$income)$overall["Accuracy"]
sensitivity_glm_occ <- confusionMatrix(data = y_hat_logistic, reference = test_factor$income)$byClass["Sensitivity"]
specificity_glm_occ <- confusionMatrix(data = y_hat_logistic, reference = test_factor$income)$byClass["Specificity"]
rmse_results <- rmse_results %>% add_row(method = "GLM Occupation Model", Accuracy = accuracy_glm_occ, Sensitivity = sensitivity_glm_occ, Specificity = specificity_glm_occ)
rmse_results %>% kbl(caption = "Accuracy, Specificity, Sensitivity Results - Including GLM Occupation Model") %>%
  kable_styling(latex_options = c("hold_position", "striped"))
#This gives us an overall accuracy of 0.3396, with a sensitivity of 0.3509 and a specificity of 0.3063.

#generalized linear model using only age
fit_glm <- glm(income ~ age,
               data=train_factor, family="binomial")
p_hat_logistic <- predict(fit_glm, test_factor, type = "response")
y_hat_logistic <- factor(ifelse(p_hat_logistic > 0.25, "<=50K", ">50K")) %>% factor
accuracy_glm_age <- confusionMatrix(data = y_hat_logistic, reference = test_factor$income)$overall["Accuracy"]
sensitivity_glm_age<- confusionMatrix(data = y_hat_logistic, reference = test_factor$income)$byClass["Sensitivity"]
specificity_glm_age <- confusionMatrix(data = y_hat_logistic, reference = test_factor$income)$byClass["Specificity"]
rmse_results <- rmse_results %>% add_row(method = "GLM Age Model", Accuracy = accuracy_glm_age, Sensitivity = sensitivity_glm_age, Specificity = specificity_glm_age)
rmse_results %>% kbl(caption = "Accuracy, Specificity, Sensitivity Results - Including GLM Age Model") %>%
  kable_styling(latex_options = c("hold_position", "striped"))
#This gives us an overall accuracy of 0.4204, with a sensitivity of 0.5065 and a specificity of 0.1644.

#generalized linear model using all predictors to demonstrate covariation/multi-collinearity
fit_glm <- glm(income ~ marital_status + occupation + education_dummy + sex + race + age,
               data=train_factor, family="binomial")
p_hat_logistic <- predict(fit_glm, test_factor, type = "response")
y_hat_logistic <- factor(ifelse(p_hat_logistic > 0.25, "<=50K", ">50K")) %>% factor
accuracy_glm_all <- confusionMatrix(data = y_hat_logistic, reference = test_factor$income)$overall["Accuracy"]
sensitivity_glm_all<- confusionMatrix(data = y_hat_logistic, reference = test_factor$income)$byClass["Sensitivity"]
specificity_glm_all <- confusionMatrix(data = y_hat_logistic, reference = test_factor$income)$byClass["Specificity"]
rmse_results <- rmse_results %>% add_row(method = "GLM All Predictors Model", Accuracy = accuracy_glm_all, Sensitivity = sensitivity_glm_all, Specificity = specificity_glm_all)
rmse_results %>% kbl(caption = "Accuracy, Specificity, Sensitivity Results - Including GLM All Predictors Model") %>%
  kable_styling(latex_options = c("hold_position", "striped"))
#This gives us an overall accuracy of 0.2145, with a sensitivity of 0.2342 and a specificity of 0.1558.

#classification (decision) trees
train_rpart <- train(income ~ education_dummy + sex.Female + age + race.Amer.Indian.Eskimo + race.Asian.Pac.Islander + race.Black + race.Other + occupation.Tech.support + 
                       occupation.Sales + occupation.Protective.serv + occupation.Prof.specialty + occupation.Priv.house.serv + occupation.Handlers.cleaners + 
                       occupation.Machine.op.inspct + occupation.Other.service + occupation.Craft.repair + occupation.Exec.managerial + occupation.Farming.fishing + 
                       occupation.Armed.Forces + occupation.Adm.clerical + marital_status.Married.spouse.absent + marital_status.Never.married + marital_status.Separated + 
                       marital_status.Married.AF.spouse + marital_status.Married.civ.spouse + marital_status.Divorced,
                     method = "rpart",
                     tuneGrid = data.frame(cp = seq(0.0, 0.1, len = 25)),
                     data = train_factor)

fancyRpartPlot(train_rpart$finalModel, sub="", cex.main=5)

train_rpart %>% ggplot(aes()) + geom_point(color = I("aquamarine4")) + geom_line(color = I("aquamarine4")) + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  ggtitle("Accuracy (Bootstrap) vs. Complexity Parameter for Classification (Decision) Tree Model") + theme(plot.title = element_text(size = 10))

y_hat <- predict(train_rpart, test_factor)
accuracy_tree <- confusionMatrix(data = y_hat, reference = test_factor$income)$overall["Accuracy"]
sensitivity_tree <- confusionMatrix(data = y_hat, reference = test_factor$income)$byClass["Sensitivity"]
specificity_tree <- confusionMatrix(data = y_hat, reference = test_factor$income)$byClass["Specificity"]
rmse_results <- rmse_results %>% add_row(method = "Classification (Decision) Tree Model", Accuracy = accuracy_tree, Sensitivity = sensitivity_tree, Specificity = specificity_tree)
rmse_results %>% kbl(caption = "Accuracy, Specificity, Sensitivity Results - Including Classification (Decision) Tree Model") %>%
  kable_styling(latex_options = c("hold_position", "striped"))

#This gives us an overall accuracy of 0.8184, with a sensitivity of 0.9308 and a specificity of 0.4840.

