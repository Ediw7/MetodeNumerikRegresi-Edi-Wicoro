import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Path to the CSV file
file_path = 'D:/Kuliah/Semester 4/Metnum/Tugas3_Regresi_Metode Numerik B_Edi Wicoro/student_performance.csv'

# Read the CSV file with delimiter ';'
data = pd.read_csv(file_path, delimiter=';')

# Check the columns in the dataframe to ensure 'NL' and 'NT' columns are present
print("Columns in data:", data.columns)

# Assuming columns in CSV are 'NL' for Number of Exercises and 'NT' for Test Scores
NL = data['NL']
NT = data['NT']

# Linear Regression Model
X = NL.values.reshape(-1, 1)
y = NT.values
linear_model = LinearRegression()
linear_model.fit(X, y)
y_pred_linear = linear_model.predict(X)
rms_error_linear = np.sqrt(np.mean((y - y_pred_linear) ** 2))
print("RMS Error (Linear Regression):", rms_error_linear)

# Plotting the graph
plt.scatter(NL, NT, label='Original Data')
plt.plot(NL, y_pred_linear, color='red', label='Linear Regression')
plt.xlabel('Number of Exercises (NL)')
plt.ylabel('Test Scores (NT)')
plt.legend()
plt.title(' (Linear Regression)')
plt.show()