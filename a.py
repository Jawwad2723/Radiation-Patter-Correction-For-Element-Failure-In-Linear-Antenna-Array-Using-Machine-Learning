
#scaler_x_new = StandardScaler()

data = pd.read_csv(file_path, delimiter=",", on_bad_lines='skip')
data.head()
x = data.iloc[2:, 9:]  # Features
y = data.iloc[2:, 1:9]  # Target

random_seed = 42
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

# Split the data (70% train, 20% test, 10% validation)
X_temp, X_test, Y_temp, Y_test = train_test_split(x, y, test_size=0.2, random_state=random_seed)
X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=0.125, random_state=random_seed)
"""
scaler_x = StandardScaler()
X_train = scaler_x.fit_transform(X_train)
X_val = scaler_x.transform(X_val)
X_test = scaler_x.transform(X_test)

scaler_y = StandardScaler()
Y_train = scaler_y.fit_transform(Y_train)
Y_val = scaler_y.transform(Y_val)
Y_test = scaler_y.transform(Y_test)

nan_mask = ~np.isnan(X_train).any(axis=1)
X_train = X_train[nan_mask]
Y_train = Y_train[nan_mask]

nan_mask_val = ~np.isnan(X_val).any(axis=1)
X_val = X_val[nan_mask_val]
Y_val = Y_val[nan_mask_val]

nan_mask_test = ~np.isnan(X_test).any(axis=1)
X_test = X_test[nan_mask_test]
Y_test = Y_test[nan_mask_test]
"""

nan_rows = np.where(np.isnan(X_train))[0]
if len(nan_rows) > 0:
    print(f"Rows with NaN values found: {nan_rows}")
    print("NaN values in X_train:")
    for row in nan_rows:
        print(X_train[row])
else:
    print("No NaN values found in X_train")

# Find rows with Inf values
inf_rows = np.where(np.isinf(X_train))[0]
if len(inf_rows) > 0:
    print(f"Rows with Inf values found: {inf_rows}")
    print("Inf values in X_train:")
    for row in inf_rows:
        print(X_train[row])
else:
    print("No Inf values found in X_train")

from tensorflow.keras.layers import Dropout
model = Sequential([
    Dense(512, activation='relu', input_dim=X_train.shape[1], kernel_initializer=HeNormal()),
    #Dense(4096, activation='relu',kernel_initializer=HeNormal()),
    #Dense(4096, activation='relu',kernel_initializer=HeNormal()),
    #Dense(2048, activation='relu', kernel_initializer=HeNormal()),
    #Dense(2048, activation='relu', kernel_initializer=HeNormal()),
    Dense(512, activation='relu', kernel_initializer=HeNormal()),
    Dense(256, activation='relu', kernel_initializer=HeNormal()),
    Dense(128, activation='relu', kernel_initializer=HeNormal()),
    Dense(64, activation='relu', kernel_initializer=HeNormal()),
    Dense(32, activation='relu', kernel_initializer=HeNormal()),
    Dense(Y_train.shape[1], activation='linear')
])

model.compile(optimizer=SGD(learning_rate=0.001), loss='mean_squared_error')

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=150,
    batch_size=16,
    callbacks=[early_stopping]
)


# Plot training and validation loss
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

# Test prediction
Y_pred_test = model.predict(X_test)

# Validation prediction
Y_pred_val = model.predict(X_val)

# Inverse transform predictions

#Y_pred_test = scaler_y.inverse_transform(Y_pred_test)
#Y_test = scaler_y.inverse_transform(Y_test)

#Y_pred_val = scaler_y.inverse_transform(Y_pred_val)
#Y_val = scaler_y.inverse_transform(Y_val)

# Flatten predictions for metrics
Y_pred_test = np.ravel(Y_pred_test)
Y_test = np.ravel(Y_test)
Y_pred_val = np.ravel(Y_pred_val)
Y_val = np.ravel(Y_val)

# Calculate metrics for testing
r2_test = r2_score(Y_test, Y_pred_test)
mse_test = mean_squared_error(Y_test, Y_pred_test)

print(f"Test R2 Score: {r2_test}, Test MSE: {mse_test}")

# Calculate metrics for validation
r2_val = r2_score(Y_val, Y_pred_val)
mse_val = mean_squared_error(Y_val, Y_pred_val)

print(f"Validation R2 Score: {r2_val}, Validation MSE: {mse_val}")

# Plot actual vs predicted for validation
plt.figure(figsize=(8, 4))
plt.scatter(Y_val, Y_pred_val, alpha=0.7)
plt.plot([min(Y_val), max(Y_val)], [min(Y_val), max(Y_val)], color='red', linestyle='--', label='Ideal')
plt.title('Validation: Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()
plt.show()

# Plot actual vs predicted for test
plt.figure(figsize=(8, 4))
plt.scatter(Y_test, Y_pred_test, alpha=0.7)
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='red', linestyle='--', label='Ideal')
plt.title('Test: Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()
plt.show()
# Input values (reshape to match model input shape)
input_values = np.array([0.11465, 0.11465, 0.11466, 0.11472, 0.11489, 0.11524, 0.11588, 0.11692, 0.11849, 0.12073, 0.12376, 0.1277, 0.13264, 0.13864, 0.14572, 0.15385, 0.16299, 0.17303, 0.18387, 0.19534, 0.20729, 0.21952, 0.23182, 0.24396, 0.25571, 0.26682, 0.27701, 0.28601, 0.29356, 0.29936, 0.30316, 0.30468, 0.3037, 0.30001, 0.29342, 0.28382, 0.27114, 0.2554, 0.23669, 0.21523, 0.19142, 0.16587, 0.13958, 0.11432, 0.093277, 0.08176, 0.084617, 0.10036, 0.1231, 0.14813, 0.1726, 0.19475, 0.21336, 0.22751, 0.23652, 0.23991, 0.23739, 0.22888, 0.21453, 0.1947, 0.16998, 0.14121, 0.10947, 0.076173, 0.043613, 0.020697, 0.033726, 0.058575, 0.080381, 0.096425, 0.10555, 0.10746, 0.10317, 0.096298, 0.095188, 0.11115, 0.14849, 0.20315, 0.27017, 0.34582, 0.42708, 0.51122, 0.59558, 0.67759, 0.7547, 0.82456, 0.88498, 0.93408, 0.97031, 0.99252, 1, 0.99252, 0.97031, 0.93408, 0.88498, 0.82456, 0.7547, 0.67759, 0.59558, 0.51122, 0.42708, 0.34582, 0.27017, 0.20315, 0.14849, 0.11115, 0.095188, 0.096298, 0.10317, 0.10746, 0.10555, 0.096425, 0.080381, 0.058575, 0.033726, 0.020697, 0.043613, 0.076173, 0.10947, 0.14121, 0.16998, 0.1947, 0.21453, 0.22888, 0.23739, 0.23991, 0.23652, 0.22751, 0.21336, 0.19475, 0.1726, 0.14813, 0.1231, 0.10036, 0.084617, 0.08176, 0.093277, 0.11432, 0.13958, 0.16587, 0.19142, 0.21523, 0.23669, 0.2554, 0.27114, 0.28382, 0.29342, 0.30001, 0.3037, 0.30468, 0.30316, 0.29936, 0.29356, 0.28601, 0.27701, 0.26682, 0.25571, 0.24396, 0.23182, 0.21952, 0.20729, 0.19534, 0.18387, 0.17303, 0.16299, 0.15385, 0.14572, 0.13864, 0.13264, 0.1277, 0.12376, 0.12073, 0.11849, 0.11692, 0.11588, 0.11524, 0.11489, 0.11472, 0.11466, 0.11465, 0.11465]).reshape(1, -1)

# Scale the input values
#scaled_input = scaler_x.transform(input_values)

# Make predictions
predicted_output = model.predict(input_values)

# Inverse transform predictions
#original_scale_output = scaler_y.inverse_transform(predicted_output)

print("Predicted output on original scale:", predicted_output)
model.save('CombinationsTrainedModel.h5')

# Load the saved model
from tensorflow.keras.models import load_model

loaded_model = load_model("CombinationsTrainedModel.h5")

# Verify it's loaded correctly (optional)
loaded_model.summary()

import pandas as pd
data = pd.read_csv("combinationsDataSet.csv" ,delimiter=",")
data.head()

x_new=data.iloc[2: ,9: ]
y_new=data.iloc[2:,1:9]

from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
# Set random seeds for reproducibility
random_seed = 42
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

# Split the data (70% train, 20% test, 10% validation)
X_temp, X_test, Y_temp, Y_test = train_test_split(x_new, y_new, test_size=0.2, random_state=random_seed)
X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=0.125, random_state=random_seed)

from sklearn.preprocessing import StandardScaler

# Scale the data
scaler_x_new = StandardScaler()
X_train = scaler_x_new.fit_transform(X_train)
X_val = scaler_x_new.transform(X_val)
X_test = scaler_x_new.transform(X_test)

scaler_y_new = StandardScaler()
Y_train = scaler_y_new.fit_transform(Y_train)
Y_val = scaler_y_new.transform(Y_val)
Y_test = scaler_y_new.transform(Y_test)

nan_mask = ~np.isnan(X_train).any(axis=1)
X_train = X_train[nan_mask]
Y_train = Y_train[nan_mask]

nan_mask_val = ~np.isnan(X_val).any(axis=1)
X_val = X_val[nan_mask_val]
Y_val = Y_val[nan_mask_val]

nan_mask_test = ~np.isnan(X_test).any(axis=1)
X_test = X_test[nan_mask_test]
Y_test = Y_test[nan_mask_test]


nan_rows = np.where(np.isnan(X_train))[0]
if len(nan_rows) > 0:
    print(f"Rows with NaN values found: {nan_rows}")
    print("NaN values in X_train:")
    for row in nan_rows:
        print(X_train[row])
else:
    print("No NaN values found in X_train")

# Find rows with Inf values
inf_rows = np.where(np.isinf(X_train))[0]
if len(inf_rows) > 0:
    print(f"Rows with Inf values found: {inf_rows}")
    print("Inf values in X_train:")
    for row in inf_rows:
        print(X_train[row])
else:
    print("No Inf values found in X_train")

from tensorflow.keras.optimizers import Adam

initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True
)
optimizer = Adam(learning_rate=lr_schedule)

# Compile the model
model.compile(loss='mean_squared_error', optimizer=optimizer)
# Train the model
history_new = model.fit(
    X_train, Y_train,
    batch_size=16,
    validation_data=(X_val, Y_val),
    epochs=30,
    verbose=1
)

from sklearn.metrics import r2_score, mean_squared_error

# Test prediction
Y_pred_test_new = model.predict(X_test)
Y_pred_val_new = model.predict(X_val)

# Inverse transform predictions
Y_pred_test_new = scaler_y_new.inverse_transform(Y_pred_test_new)
Y_test_new = scaler_y_new.inverse_transform(Y_test)

Y_pred_val_new = scaler_y_new.inverse_transform(Y_pred_val_new)
Y_val_new = scaler_y_new.inverse_transform(Y_val)

# Flatten predictions for metrics
Y_pred_test_new = np.ravel(Y_pred_test_new)
Y_test_new = np.ravel(Y_test_new)
Y_pred_val_new = np.ravel(Y_pred_val_new)
Y_val_new = np.ravel(Y_val_new)

# Calculate metrics for testing
r2_test_new = r2_score(Y_test_new, Y_pred_test_new)
mse_test_new = mean_squared_error(Y_test_new, Y_pred_test_new)

print(f"New Test R2 Score: {r2_test_new}, New Test MSE: {mse_test_new}")

# Calculate metrics for validation
r2_val_new = r2_score(Y_val_new, Y_pred_val_new)
mse_val_new = mean_squared_error(Y_val_new, Y_pred_val_new)

print(f"New Validation R2 Score: {r2_val_new}, New Validation MSE: {mse_val_new}")
# Input values (reshape to match model input shape)
input_values = np.array([0.25, 0.25, 0.25, 0.24998, 0.24994, 0.24986, 0.24971, 0.24947, 0.24909, 0.24854, 0.24778, 0.24676, 0.24542, 0.2437, 0.24156, 0.23891, 0.23571, 0.23187, 0.22734, 0.22204, 0.21591, 0.20889, 0.20094, 0.19199, 0.18201, 0.171, 0.15894, 0.14588, 0.13188, 0.11707, 0.1017, 0.086164, 0.071248, 0.058434, 0.050403, 0.050323, 0.05862, 0.072504, 0.089159, 0.10684, 0.12447, 0.1413, 0.15673, 0.17021, 0.18125, 0.18938, 0.19414, 0.19514, 0.19202, 0.18451, 0.17243, 0.15575, 0.13468, 0.10988, 0.083204, 0.060352, 0.056734, 0.081057, 0.12016, 0.16515, 0.21246, 0.26002, 0.30617, 0.34938, 0.38815, 0.421, 0.44653, 0.46345, 0.47058, 0.46695, 0.45181, 0.42471, 0.38553, 0.3346, 0.27283, 0.2023, 0.12905, 0.081896, 0.12448, 0.21501, 0.31693, 0.42153, 0.52475, 0.62352, 0.71516, 0.79724, 0.86758, 0.92434, 0.96599, 0.99144, 1, 0.99144, 0.96599, 0.92434, 0.86758, 0.79724, 0.71516, 0.62352, 0.52475, 0.42153, 0.31693, 0.21501, 0.12448, 0.081896, 0.12905, 0.2023, 0.27283, 0.3346, 0.38553, 0.42471, 0.45181, 0.46695, 0.47058, 0.46345, 0.44653, 0.421, 0.38815, 0.34938, 0.30617, 0.26002, 0.21246, 0.16515, 0.12016, 0.081057, 0.056734, 0.060352, 0.083204, 0.10988, 0.13468, 0.15575, 0.17243, 0.18451, 0.19202, 0.19514, 0.19414, 0.18938, 0.18125, 0.17021, 0.15673, 0.1413, 0.12447, 0.10684, 0.089159, 0.072504, 0.05862, 0.050323, 0.050403, 0.058434, 0.071248, 0.086164, 0.1017, 0.11707, 0.13188, 0.14588, 0.15894, 0.171, 0.18201, 0.19199, 0.20094, 0.20889, 0.21591, 0.22204, 0.22734, 0.23187, 0.23571, 0.23891, 0.24156, 0.2437, 0.24542, 0.24676, 0.24778, 0.24854, 0.24909, 0.24947, 0.24971, 0.24986, 0.24994, 0.24998, 0.25, 0.25, 0.25]).reshape(1, -1)

# Scale the input values
scaled_input = scaler_x_new.transform(input_values)

# Make predictions
predicted_output = model.predict(scaled_input)

# Inverse transform predictions
original_scale_output = scaler_y_new.inverse_transform(predicted_output)

print("Predicted output on original scale:", original_scale_output)

from tensorflow import keras

# Load the model
loaded_model = keras.models.load_model('ChebyshevANDFailureTrainedModel.h5')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf

scaler_x_new = StandardScaler()
scaler_y_new = StandardScaler()
# Input values (reshape to match model input shape)
input_values = np.array([0.25, 0.25, 0.25, 0.24998, 0.24994, 0.24986, 0.24971, 0.24947, 0.24909, 0.24854, 0.24778, 0.24676, 0.24542, 0.2437, 0.24156, 0.23891, 0.23571, 0.23187, 0.22734, 0.22204, 0.21591, 0.20889, 0.20094, 0.19199, 0.18201, 0.171, 0.15894, 0.14588, 0.13188, 0.11707, 0.1017, 0.086164, 0.071248, 0.058434, 0.050403, 0.050323, 0.05862, 0.072504, 0.089159, 0.10684, 0.12447, 0.1413, 0.15673, 0.17021, 0.18125, 0.18938, 0.19414, 0.19514, 0.19202, 0.18451, 0.17243, 0.15575, 0.13468, 0.10988, 0.083204, 0.060352, 0.056734, 0.081057, 0.12016, 0.16515, 0.21246, 0.26002, 0.30617, 0.34938, 0.38815, 0.421, 0.44653, 0.46345, 0.47058, 0.46695, 0.45181, 0.42471, 0.38553, 0.3346, 0.27283, 0.2023, 0.12905, 0.081896, 0.12448, 0.21501, 0.31693, 0.42153, 0.52475, 0.62352, 0.71516, 0.79724, 0.86758, 0.92434, 0.96599, 0.99144, 1, 0.99144, 0.96599, 0.92434, 0.86758, 0.79724, 0.71516, 0.62352, 0.52475, 0.42153, 0.31693, 0.21501, 0.12448, 0.081896, 0.12905, 0.2023, 0.27283, 0.3346, 0.38553, 0.42471, 0.45181, 0.46695, 0.47058, 0.46345, 0.44653, 0.421, 0.38815, 0.34938, 0.30617, 0.26002, 0.21246, 0.16515, 0.12016, 0.081057, 0.056734, 0.060352, 0.083204, 0.10988, 0.13468, 0.15575, 0.17243, 0.18451, 0.19202, 0.19514, 0.19414, 0.18938, 0.18125, 0.17021, 0.15673, 0.1413, 0.12447, 0.10684, 0.089159, 0.072504, 0.05862, 0.050323, 0.050403, 0.058434, 0.071248, 0.086164, 0.1017, 0.11707, 0.13188, 0.14588, 0.15894, 0.171, 0.18201, 0.19199, 0.20094, 0.20889, 0.21591, 0.22204, 0.22734, 0.23187, 0.23571, 0.23891, 0.24156, 0.2437, 0.24542, 0.24676, 0.24778, 0.24854, 0.24909, 0.24947, 0.24971, 0.24986, 0.24994, 0.24998, 0.25, 0.25, 0.25]).reshape(1, -1)
scaler_x_new.fit(input_values)

# Scale the input values
scaled_input = scaler_x_new.transform(input_values)

# Make predictions
predicted_output = loaded_model.predict(scaled_input)

scaler_y_new.fit(predicted_output)
# Inverse transform predictions
original_scale_output = scaler_y_new.inverse_transform(predicted_output)

print("Predicted output on original scale:", original_scale_output)
