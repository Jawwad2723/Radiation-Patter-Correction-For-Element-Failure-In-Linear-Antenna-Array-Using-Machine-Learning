import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import tensorflow as tf

# Load data
data = pd.read_csv("combinationsDataSet.csv", delimiter=",")

# Split data into features (X) and target (y)
x = data.iloc[2:, 9:]  # Features (from column 9 onwards)
y = data.iloc[2:, 1:9]  # Target (columns 1-8)

# Set random seeds for reproducibility
random_seed = 42
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

# Split the data into training, testing, and validation sets
X_temp, X_test, Y_temp, Y_test = train_test_split(x, y, test_size=0.2, random_state=random_seed)
X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=0.125, random_state=random_seed)

# Scale the feature data
scaler_x = StandardScaler()
X_train = scaler_x.fit_transform(X_train)
X_val = scaler_x.transform(X_val)
X_test = scaler_x.transform(X_test)

# Scale the target data
scaler_y = StandardScaler()
Y_train = scaler_y.fit_transform(Y_train)
Y_val = scaler_y.transform(Y_val)
Y_test = scaler_y.transform(Y_test)

# Load the pre-trained model
loaded_model = keras.models.load_model('ChebyshevANDFailureTrainedModel.h5')

# Example input for prediction
input_values1 = np.array([1.32E-16, 0.00051379, 0.002055, 0.004623, 0.0082165, 0.012833, 0.018468, 0.025115, 0.032763, 0.041397, 0.050995, 0.061529, 0.07296, 0.085239, 0.098305, 0.11208, 0.12648, 0.14139, 0.15669, 0.17222, 0.18782, 0.20329, 0.21842, 0.23299, 0.24673, 0.25937, 0.27062, 0.28019, 0.28776, 0.29303, 0.2957, 0.29547, 0.29208, 0.28529, 0.2749, 0.26078, 0.24284, 0.22108, 0.19557, 0.16649, 0.1341, 0.09878, 0.061004, 0.021362, 0.019458, 0.060671, 0.10141, 0.14075, 0.1777, 0.21126, 0.24046, 0.26434, 0.28203, 0.29277, 0.29593, 0.2911, 0.27805, 0.25681, 0.22765, 0.19116, 0.14817, 0.099813, 0.047479, 0.0072087, 0.062425, 0.1162, 0.16647, 0.21115, 0.2482, 0.27572, 0.292, 0.29562, 0.28549, 0.26095, 0.22177, 0.16823, 0.10109, 0.021622, 0.06843, 0.16689, 0.27123, 0.37859, 0.48596, 0.59019, 0.68819, 0.77696, 0.85376, 0.91618, 0.96224, 0.99048, 1, 0.99048, 0.96224, 0.91618, 0.85376, 0.77696, 0.68819, 0.59019, 0.48596, 0.37859, 0.27123, 0.16689, 0.06843, 0.021622, 0.10109, 0.16823, 0.22177, 0.26095, 0.28549, 0.29562, 0.292, 0.27572, 0.2482, 0.21115, 0.16647, 0.1162, 0.062425, 0.0072087, 0.047479, 0.099813, 0.14817, 0.19116, 0.22765, 0.25681, 0.27805, 0.2911, 0.29593, 0.29277, 0.28203, 0.26434, 0.24046, 0.21126, 0.1777, 0.14075, 0.10141, 0.060671, 0.019458, 0.021362, 0.061004, 0.09878, 0.1341, 0.16649, 0.19557, 0.22108, 0.24284, 0.26078, 0.2749, 0.28529, 0.29208, 0.29547, 0.2957, 0.29303, 0.28776, 0.28019, 0.27062, 0.25937, 0.24673, 0.23299, 0.21842, 0.20329, 0.18782, 0.17222, 0.15669, 0.14139, 0.12648, 0.11208, 0.098305, 0.085239, 0.07296, 0.061529, 0.050995, 0.041397, 0.032763, 0.025115, 0.018468, 0.012833, 0.0082165, 0.004623, 0.002055, 0.00051379, 1.32E-16]).reshape(1, -1)

# Scale the input data using the fitted scaler (scaler_x)
scaled_input1 = scaler_x.transform(input_values1)

# Make predictions using the trained model
predicted_output1 = loaded_model.predict(scaled_input1)

# Inverse scale the predictions to get them back to the original scale
original_scale_output1 = scaler_y.inverse_transform(predicted_output1)

# Output the predictions in the original scale
print("Predicted output on original scale:", original_scale_output1)
