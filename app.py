import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from keras.layers import Dense, Dropout, LSTM, BatchNormalization, Bidirectional
from keras.models import load_model
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Title of the application
st.title("Prédiction du cours d'une action")

# Initialize today's date
now = datetime.now()
today = datetime.now().strftime('%Y-%m-%d')

# Specify the stock
user_input = st.text_input("Entrez le nom de l'action:", "AAPL")
start = st.text_input("Entrez la date de début:", "2010-01-01")
end = st.text_input("Entrez la date de fin:", today)
df = yf.download(user_input, start=start, end=end)

# Describe the data
st.subheader("Données de {} à {}".format(start, end))
df = df.reset_index()
df = df.drop(['Date', 'Adj Close'], axis=1)
st.write(df.describe())

# Visualization
st.subheader("Cours du prix de clôture de {} à {}".format(start, end))
fig = plt.figure(figsize=(16, 8))
plt.plot(df['Close'])
st.pyplot(fig)

st.subheader("Prix de clôture vs Courbe des moyennes mobiles 100MA & 200MA")
ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()
fig2 = plt.figure(figsize=(16, 8))
plt.plot(df['Close'])
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
st.pyplot(fig2)

# Split data into training (70%) and testing data (30%)
data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])

# Scaling and transforming to matrix
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# Splitting x_train and y_train by 100 days step
x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i - 100:i])
    y_train.append(data_training_array[i, 0])

# Converting x_train and y_train to matrices
x_train, y_train = np.array(x_train), np.array(y_train)

# Load the model
model = load_model("prediction_boursiere1_saved_model.h5")

# Add 100 rows to testing data
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
data_testing_array = scaler.fit_transform(final_df)

# Splitting x_test and x_train by 100 days step
x_test = []
y_test = []
for i in range(100, data_testing_array.shape[0]):
    x_test.append(data_testing_array[i - 100:i])
    y_test.append(data_testing_array[i, 0])

# Converting x_test and y_test to matrices
x_test, y_test = np.array(x_test), np.array(y_test)

# Predicting the test data
y_predicted = model.predict(x_test)

# Undo scaling
scale_factor = 1 / scaler.scale_
y_predicted = y_predicted * scale_factor
y_testing = y_test * scale_factor

# Visualize the closing price prediction VS real closing price
st.subheader("Prédictions VS valeurs réelles")
fig3 = plt.figure(figsize=(16, 8))
plt.plot(y_testing, 'b', label='Prix réel')
plt.plot(y_predicted, 'r', label='Prix prédit')
plt.xlabel('Temps')
plt.ylabel('Prix de clôture')
plt.legend()
st.pyplot(fig3)

x = []
x.append(y_test[-100:])
x = np.array(x)
x = np.expand_dims(x, axis=2)  # Add a new axis to represent the number of features
pred_tomorrow = model.predict(x)
st.subheader("Prédiction des prix de clôture pour demain")
next_day = datetime.now() + timedelta(1)
tomorrow = next_day.strftime('%d-%m-%Y')
st.write("Prédiction du prix de clôture pour le {}:  {}".format(tomorrow, float(pred_tomorrow*scale_factor)))
