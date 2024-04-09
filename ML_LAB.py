import pandas as pd
df = pd.read_csv('car_dataset.csv')
# print(df)

new_df = df[['enginesize','price']]
# new_df

x = new_df['enginesize']
y = new_df['price']
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#shapes of splitted data = sizes of the x_train and y_train should be same . also , tests.
print("X_train: ",x_train.shape)
print("X_test: ",x_test.shape)
print("Y_train: ",y_train.shape)
print("y_test: ",y_test.shape)

mymodel=LinearRegression()
mymodel.fit(x_train.values.reshape(-1,1),y_train)


LinearRegression()

print(mymodel.coef_)
print(mymodel.intercept_)

y_pred = mymodel.predict(x_test.values.reshape(-1,1))


mse= mean_squared_error(y_test,y_pred)
print("MSE -------> ",mse)


import math
rmse = math.sqrt(mse)
print("RMSE ---------->",rmse)


mae = mean_absolute_error(y_test,y_pred)
print("MAE -------> ",mae)


r2 = r2_score(y_test,y_pred)
print("R2 -----> ",r2)


import matplotlib.pyplot as plt
import seaborn as sns


plt.scatter(y_test,y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")

plt.show()

sns.regplot(x=x,y=y,ci=None,color='red')
plt.show()


df.plot(x='horsepower',ylabel='price')
plt.show()