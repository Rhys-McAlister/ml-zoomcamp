x = [1.0, -2.0, 3.0]
w = [-3.0, -1.0, 2.0]
b = 1.0

xw0 = x[1] * w[1]
xw1 = x[2] * w[2]
xw2 = x[3] * w[3]

z = xw0 + xw1 + xw2 + b

y = max(z, 0.0)

using DataFrames
using GLM
using Statistics
# TODO: The reponse data is blank subtracted, maybe consider adding an automatic blank subraction method
data = DataFrame(response=[0.221, 0.057, 0.119, 0.73, 0.559, 0.383], predictor=[0.2, 0.05, 0.1, 0.8, 0.6, 0.4])
# predictor = [0.2, 0.05, 0.1, 0.8, 0.6, 0.4]
# response = [0.221, 0.057, 0.119, 0.73, 0.559, 0.383]

ols = lm(@formula(response ~ predictor), data)

lm1 = fit(LinearModel, @formula(response ~ predictor), data)





