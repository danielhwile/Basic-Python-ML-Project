import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def data_import():
    path = r"./insurance.csv"
    df = pd.read_csv(path)
    data = pd.DataFrame(df)
    return data
database = data_import()

#Now we need to clean the data... currently the data has non numerical reporting like:
# Male // Female
# Smoker: Yes // No
# Area NorthEast // NorthWest // SouthEast // SouthWest

#We will change that to the following
# Male:0 Female: 1
# Smoker: Yes:1 // No: 0
# Area northeast:0 // northwest:1 // southeast:2 // southwest:3

database[['sex']] = database[['sex']].replace(['male'],0)
database[['sex']] = database[['sex']].replace(['female'],1)

database[['smoker']] = database[['smoker']].replace(['yes'],1)
database[['smoker']] = database[['smoker']].replace(['no'],0)

database[['region']] = database[['region']].replace(['northeast'],0)
database[['region']] = database[['region']].replace(['northwest'],1)
database[['region']] = database[['region']].replace(['southeast'],2)
database[['region']] = database[['region']].replace(['southwest'],3)

x = database[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
xa = [['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
y = database[['charges']]


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.8, test_size=.2, random_state=6)

mlr = LinearRegression()

mlr.fit(x_train, y_train)


y_predict = mlr.predict(x_test)

y_test_list = y_test['charges'].tolist()
y_test_list.sort()
y_predict_list = []
for i in y_predict:
    y_predict_list.append(i)
y_predict_list.sort()

x_axis = range(1,269)

plt.scatter(y_test, y_predict, alpha=0.4, label="Predict")
#plt.scatter(x_axis, y_test_list, alpha=0.4, label="actual")
plt.xlabel("Actual Cost")
plt.ylabel("Predicted Cost")
#plt.legend(loc="upper left")
plt.title("Actual Costs vs Predicted Insurance Cost")
plt.xlim(0,50000)
plt.ylim(0,50000)
plt.show()