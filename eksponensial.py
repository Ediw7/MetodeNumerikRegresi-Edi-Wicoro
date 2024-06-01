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

X = NL.values.reshape(-1, 1)
y = np.log(NT.values)  # Take natural logarithm of NT
exponential_model = LinearRegression()
exponential_model.fit(X, y)
y_pred_exponential = exponential_model.predict(X)
y_pred_exponential = np.exp(y_pred_exponential)  # Take exponential to get original scale
rms_error_exponential = np.sqrt(np.mean((NT - y_pred_exponential) ** 2))
print("RMS Error (Exponential Regression):", rms_error_exponential)

# Plotting the graph
plt.scatter(NL, NT, label='Original Data')
plt.plot(NL, y_pred_exponential, color='green', label='Exponential Regression')
plt.xlabel('Number of Exercises (NL)')
plt.ylabel('Test Scores (NT)')
plt.legend()
plt.title('(Exponential Regression)')
plt.show()