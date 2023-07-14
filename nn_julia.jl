using DataFrames
using GLM
using Statistics
using Plots
using TypedTables
# TODO: The reponse data is blank subtracted, maybe consider adding an automatic blank subraction method
data = DataFrame(response=[0.221, 0.057, 0.119, 0.73, 0.559, 0.383], predictor=[0.2, 0.05, 0.1, 0.8, 0.6, 0.4])
# predictor = [0.2, 0.05, 0.1, 0.8, 0.6, 0.4]
# response = [0.221, 0.057, 0.119, 0.73, 0.559, 0.383]

ols = lm(@formula(response ~ predictor), data)

lm1 = fit(LinearModel, @formula(response ~ predictor), data)


r2(lm1)
r2(ols)

deviance(lm1)
stderror(lm1)
dof_residual(lm1)

coeftable(lm1)
b0 = coef(lm1)[1]
b1 = coef(lm1)[2]

fitted_values = b0 .+ b1 .* data.predictor
sse = sum((fitted_values .- data.response).^2)
syx = sqrt(sse / (length(data.response) - 2))

regression_ss = sum((fitted_values .- mean(data.response)).^2)
residual_ss = sum((fitted_values .- data.response).^2)

X = [0.221, 0.057, 0.119, 0.73, 0.559, 0.383]
Y = [0.2, 0.05, 0.1, 0.8, 0.6, 0.4]

