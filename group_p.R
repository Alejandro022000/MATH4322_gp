data <- read.table("C:/Users/ale07/OneDrive/School/Spring 2023/Data Science and Machine Learning/Group project/winequality-white.csv", header=TRUE,  sep = ";") # nolint
#create linear regression model with all variables and quality as the response
model <- lm(quality ~ ., data = data)
#summary(model)
summary(model)
#plot(model)
plot(model)