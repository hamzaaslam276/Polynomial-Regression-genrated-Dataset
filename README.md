# Polynomial Regression Model

## 📌 Overview
This project demonstrates **Polynomial Regression** and compares it with **Linear Regression** using Python and **scikit-learn**. It includes dataset creation, model training, and visualization.

## 📂 Libraries Used
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import operator

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
```

## 📊 Dataset Creation
We generate synthetic data using **NumPy**:
```python
np.random.seed(0)
x = np.random.normal(0, 1, 20)
y = np.random.normal(0, 1, 20)

print(f"x: {x}")
print(f"y: {y}")
```

## 📈 Visualizing the Dataset
```python
plt.scatter(x, y)
plt.show()
```

## 🔄 Data Preprocessing
Convert 1D arrays to 2D arrays:
```python
x = x[:, np.newaxis]
y = y[:, np.newaxis]
```

## 📉 Linear Regression Model
```python
model_lin = LinearRegression()
model_lin.fit(x, y)
y_pred = model_lin.predict(x)

plt.scatter(x, y)
plt.plot(x, y_pred, color='r')
plt.show()
```

### 🔹 Error Calculation for Linear Regression
```python
mse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)
print(mse)
print(r2)
```
**Output:**
```
1.1832766119182259
0.007636444138149345
```

## 📈 Polynomial Regression Model
```python
polynomial_features = PolynomialFeatures(degree=2)
x_poly = polynomial_features.fit_transform(x)

model = LinearRegression()
model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)
```

### 🔹 Error Calculation for Polynomial Regression
```python
mse = np.sqrt(mean_squared_error(y, y_poly_pred))
r2 = r2_score(y, y_poly_pred)
print(mse)
print(r2)
```
**Output:**
```
1.1507521092081143
0.061440511342737425
```

## 🔢 Model Coefficients & Equation
```python
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
```
**Output:**
```
Coefficients: [[ 0.          0.54780984 -0.32901375]]
Intercept: [0.08832508]
```

### 🔹 Polynomial Equation
```python
res = "y = f(x) = " + str(model.intercept_[0])
for i, r in enumerate(model.coef_[0]):
    res = res + " + {}*x^{}".format("%.2f" % r, i)
print(res)
```
**Output:**
```
y = f(x) = 0.08832508 + 0.00*x^0 + 0.55*x^1 + -0.33*x^2
```

## 📊 Visualizing Polynomial Regression
```python
plt.scatter(x, y, color='blue')
plt.scatter(x, model.predict(polynomial_features.fit_transform(x)), color='red')
plt.title('Polynomial Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
```

## 📌 Sorted Visualization
```python
plt.scatter(x, y)
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(x, y_poly_pred), key=sort_axis)
x, y_poly_pred = zip(*sorted_zip)
plt.plot(x, y_poly_pred, color='red')
plt.show()
```

## 📢 Conclusion
- **Linear Regression** does not fit the data well (**R² = 0.0076**).
- **Polynomial Regression** provides a much better fit (**R² = 0.0614**).

🚀 **Polynomial Regression is useful when data shows a nonlinear relationship!**

