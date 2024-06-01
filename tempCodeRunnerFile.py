# Linear Regression Model
X = NL.values.reshape(-1, 1)
y = NT.values
linear_model = LinearRegression()
linear_model.fit(X, y)
y_pred_linear = linear_model.predict(X)
rms_error_linear = np.sqrt(np.mean((y - y_pred_linear) ** 2))
print("RMS Error (Linear Regression):", rms_error_linear)