import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('Position_Salaries.csv')
x = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x, y)

from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=10)
poly_x = pf.fit_transform(x)
lr_poly = LinearRegression()
lr_poly.fit(poly_x, y)

plt.scatter(x, y, color='green')
plt.plot(x, lr.predict(x), color='blue')
plt.title('Linear Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

plt.scatter(x, y, color='red')
plt.plot(x, lr_poly.predict(poly_x), color='yellow')
plt.title('Polynomial Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# smoother curve
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color='red')
plt.plot(x_grid, lr_poly.predict(pf.fit_transform(x_grid)), color='yellow')
plt.title('Polynomial Regression, smooth curve')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# single prediction from lr
np.set_printoptions(2)
print(lr.predict([[6.5]]))

# single prediction from plr
print(lr_poly.predict(pf.fit_transform([[6.5]])))
