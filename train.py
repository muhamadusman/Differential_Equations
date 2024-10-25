from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

# Load the CSV file
csv_file_path = 'f_vs_eta.csv'
data = pd.read_csv(csv_file_path)

# Display the first few rows of the data to understand the structure
data.head()



# Separate inputs and outputs
X = data.iloc[:, :-2].values  # All columns except the last 2 as input
y = data.iloc[:, -2:].values  # Last 2 columns as output (target)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the FCN model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))  # First hidden layer
model.add(Dense(32, activation='relu'))  # Second hidden layer
model.add(Dense(2, activation='linear'))  # Output layer with 2 outputs for regression

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])  # For regression

# Train the model and capture the training history
history = model.fit(X_train, y_train, epochs=90, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
loss, mae = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test MAE: {mae}')

# Save the trained model
model.save('your_model_multitarget.h5')

# Make predictions on the test set
predictions = model.predict(X_test)

# Calculate mean absolute error between ground truth and predictions
mean_error = np.mean(np.abs(y_test - predictions))
print(f'Mean Absolute Error: {mean_error}')


# Save test cases, inputs (entire row), ground truth, and predictions to a new CSV file
results = pd.DataFrame(
    np.hstack((X_test, y_test, predictions)),
    columns=[f'Input_{i+1}' for i in range(X_test.shape[1])] + 
            ['Ground Truth 1', 'Ground Truth 2', 'Prediction 1', 'Prediction 2']
)

# Calculate absolute errors and mean error for each test case
results['Absolute Error 1'] = np.abs(results['Ground Truth 1'] - results['Prediction 1'])
results['Absolute Error 2'] = np.abs(results['Ground Truth 2'] - results['Prediction 2'])
results['Mean Error'] = results[['Absolute Error 1', 'Absolute Error 2']].mean(axis=1)

# Save results to CSV
results.to_csv('test_results_multitarget_updated.csv', index=False)
print('Test results with full input rows saved to test_results_multitarget_updated.csv')


# Plot the learning curve
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Learning Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Save the plot to a file
plot_file_path = 'learning_curve_multitarget.png'
plt.savefig(plot_file_path)

# Clear the plot to avoid display if running in an interactive environment
plt.close()
