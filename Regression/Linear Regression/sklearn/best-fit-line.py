
'''
Finding the best fit linear slope for a dataset example
'''

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')


# test data
xs = np.array([1,2,3,4,5,6], dtype=np.float64)
ys = np.array([5,4,6,5,6,7], dtype=np.float64)


# generate best fit slope based on averages and square means
def best_fit_slope_and_intercept(xs, ys):
    m = ((mean(xs) * mean(ys)) - mean(xs * ys)) / ((mean(xs) ** 2) - mean(xs ** 2))
    b = mean(ys) - m*mean(xs)

    return m, b


# y = mx + c
m,b = best_fit_slope_and_intercept(xs, ys)

regression_line = [(m * x) + b for x in xs] # create list of y values


# predictions
predict_x = 8
predict_y = (m * predict_x) + b


# plot
plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, color='g')
plt.plot(xs, regression_line)
plt.show()
