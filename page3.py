import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from trySendModel import sendInputs
from page2 import global_var

data2 = pd.DataFrame()
def sendtofile3(df_jaugeage):
    global data2
    data2 = df_jaugeage

def show_data_pagee(df_daily):
    # Afficher le logo dans le coin supérieur gauche de la page principale
    st.sidebar.image("so3.png", width=100)
    # Afficher les données
    st.sidebar.title("Menu")
    button_c = st.sidebar.button("Display daily measurement dataset")
    
    if button_c:
        st.subheader("Daily measurement dataset")
        st.write(df_daily)
        

# Add "NONE FLP WHP" to the sidebar
    show_algorithm_options(df_daily, data2)

# Prédire les nouvelles valeurs en utilisant les prédictions précédentes
predictions_flp = []
# Prédire les nouvelles valeurs en utilisant les prédictions précédentes
predictions_whp = []


def show_algorithm_options(df_daily, data2):
    if data2.empty :
        st.warning('To make a prediction with LSTM, you first need to train the models on the daily datasets.!!')
    else:
        st.sidebar.title("AI Model Configuration")
        algorithm_options = ["None", "LSTM"]
        algorithmm = st.sidebar.selectbox("Select AI model", algorithm_options)
            
        if algorithmm in ["LSTM"]:
            num_input_neurons = 128 
            num_output_neurons = 2
            num_hidden_layers = 1
            #  input_activation = "relu"
            num_neurons = 64 
            #  output_activation = "relu"
            #  hidden_activation  = "relu"
            model = Sequential()
            
            seq_size = st.sidebar.checkbox("Choose a sequence size")
            if not seq_size:
                st.warning("Choose a sequence size")
            elif seq_size:
                seq_size = st.sidebar.number_input("Sequence size", min_value=1, step=1, value=7)

            # Couche d'entrée
            use_input_layer = st.sidebar.checkbox("Select an input layer")
            if not use_input_layer and model is not None:
                st.warning("Choose a sequence size")
            elif use_input_layer and model is not None:
                num_input_neurons = st.sidebar.number_input("Number of neurons in input layer", min_value=1, step=1, value=64)
                model.add(LSTM(num_input_neurons, input_shape=(seq_size, 1)))

            # hidden layer
            use_hidden_layer = st.sidebar.checkbox("Choose the number of hidden layer")
            if not use_hidden_layer and model is not None:
                st.warning("Choose the number of hidden layer")
            elif use_hidden_layer and model is not None:
                num_hidden_layers = st.sidebar.number_input("Number of hidden layer", min_value=1, step=1, value=1)
                for i in range(num_hidden_layers):
                    num_neurons = st.sidebar.number_input(f"Number of neurons in the hidden layer {i+1}", min_value=1, step=1, value=32)
                    model.add(Dense(num_neurons))
                    

            # Couche de sortie
            use_output_layer = st.sidebar.checkbox("Select an output layer")
            if not use_output_layer and model is not None:
                st.warning("Select an output layer")
            elif use_output_layer and model is not None:
                num_output_neurons = st.sidebar.number_input("Number of neurons in the output layer", min_value=1, step=1, value=1)
                model.add(Dense(num_output_neurons))

            # Ajouter une option pour le nombre d'époques
            use_epochs = st.sidebar.checkbox("Choose the number of epochs")
            if use_epochs:
                num_epochs = st.sidebar.number_input("Number of epochs", min_value=1, step=1, value=150)

            # Ajouter une option pour la taille du batch
            use_batch_size = st.sidebar.checkbox("Choose the number of batch")
            if use_batch_size:
                batch_size = st.sidebar.number_input("Size of the batch", min_value=1, step=1, value=32)
            
            nbr_valeur =  st.sidebar.checkbox("Select the periode of forcast") 
            if not nbr_valeur:
                st.warning("Select the periode of forcast")
            elif nbr_valeur:
                nbr_valeur = st.sidebar.number_input("The periode of forcast", min_value=1, step=1, value=4)

    #khlas if
        st.sidebar.title("Model configuration")
        model_options = ["None", "FLP_LSTM Model", "WHP_LSTM Model"]
        model_type = st.sidebar.selectbox("Select the IA model", model_options)

        if algorithmm == "LSTM":
            st.subheader("AI LSTM")
            st.write("Type of model: ", model_type)

            if model_type == "FLP_LSTM Model":

                st.subheader("Forecast of FLP")
                # Remove the 'START_DATE' column
                # Load the data
                data = df_daily

                # Convert the 'START_DATE' column to datetime with the original format
                data['START_DATE'] = pd.to_datetime(data['START_DATE'])
                data['jour'] = data['START_DATE'].dt.day
                data['month'] = data['START_DATE'].dt.month
                data['year'] = data['START_DATE'].dt.year

                supprimer_doublons = st.sidebar.checkbox("Delete repeated data")
                if supprimer_doublons:
                    # Identify duplicate rows
                    doublons = data.duplicated()
                    # Remove duplicate rows
                    data = data[~doublons]

                # Handling of missing values
                handle_nan = st.sidebar.checkbox("Handling of missing values NAN", key="checkbox_flp")
                if not handle_nan:
                    st.warning("Handling of missing values")
                else:
                    handle_nan = st.sidebar.selectbox("Choose", options=["None", "Delete data point that contain NAN", "Replace with the precedent value", "Replace NAN with avrg"])
                    if handle_nan == "Delete data point that contain NAN":
                        data = data.dropna()
                        data2 = data2.dropna()
                    elif handle_nan == "Replace with the precedent value":
                        data = data.fillna(method='pad')
                        data2 = data2.fillna(method='pad')
                    elif handle_nan == "Replace NAN with avrg":
                        data = data.fillna(data.mean())
                        data2 = data2.fillna(data.mean())

                # Normalize the data using standardization
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(data[['FLP']])

                # Prepare input and output sequences for model training
                def create_sequences(data, seq_size):
                    X, y = [], []
                    for i in range(len(data) - seq_size):
                        X.append(data[i:i + seq_size])
                        y.append(data[i + seq_size])
                    return np.array(X), np.array(y)

                seq_size = seq_size

                # Prepare input and output sequences for model training
                train_data, test_data = train_test_split(scaled_data, test_size=0.2, shuffle=False)

                train_X, train_y = create_sequences(train_data, seq_size)
                test_X, test_y = create_sequences(test_data, seq_size)

                # Reshape the input data to have a dimension of 3
                train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
                test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))

                # Create and train the LSTM model
                model = Sequential()
                model.add(LSTM(num_input_neurons, input_shape=(seq_size, 1)))  # 1 for the FLP column
                for i in range(num_hidden_layers):
                    model.add(Dense(num_neurons))
                model.add(Dense(num_output_neurons))  # 1 for the FLP column
                model.compile(loss='mean_squared_error', optimizer='adam')
                model.summary()

                if st.button("Model training"):

                    # Training the model
                    if model is not None:
                        if use_epochs and use_batch_size:
                            model.fit(train_X, train_y, epochs=num_epochs, batch_size=batch_size, verbose=2)
                        elif use_epochs:
                            model.fit(train_X, train_y, epochs=num_epochs, verbose=2)
                        elif use_batch_size:
                            model.fit(train_X, train_y, batch_size=batch_size, verbose=2)
                        else:
                            model.fit(train_X, train_y, verbose=2)



                            # Make predictions on the test set
                    test_predictions = model.predict(test_X)

                    # Inverse scaling of predictions
                    test_predictions = scaler.inverse_transform(test_predictions)

                    # Inverse scaling of test_y
                    test_y = scaler.inverse_transform(test_y)

                    # Calculate evaluation metrics on the test set
                    r2 = r2_score(test_y, test_predictions)
                    mse = mean_squared_error(test_y, test_predictions)
                    mae = mean_absolute_error(test_y, test_predictions)

                    st.write("R2 Score (FLP):", r2)
                    st.write("Mean Squared Error (FLP):", mse)
                    st.write("Mean Absolute Error (FLP):", mae)

                    # Prendre les 7 dernières valeurs pour prédire la 8ème
                    last_sequence = scaled_data[-seq_size:]
                    last_sequence = np.reshape(last_sequence, (1, seq_size, 1))
                    
                    for _ in range(nbr_valeur):  # Prédire 7 nouvelles valeurs
                        # Faire une prédiction en utilisant la séquence actuelle
                        next_value = model.predict(last_sequence)[0][0]
                        # Ajouter la nouvelle prédiction à la liste des prédictions
                    
                        # Mettre à jour la séquence précédente en éliminant la première valeur et en ajoutant la prédiction actuelle
                        last_sequence = np.append(last_sequence[:, 1:, :], [[[next_value]]], axis=1)
                        # Convertir la prédiction normalisée en valeur brute
                        next_value = scaler.inverse_transform([[next_value]])[0][0]
                        predictions_flp.append(next_value)
                        # Afficher la prédiction actuelle
                        st.write("Forcast", _ + 1, ":", next_value)
                    

                    # Plot the graph of predicted values
                    plt.plot(predictions_flp[-nbr_valeur:], color='orange')
                    plt.xlabel("INDEX")
                    plt.ylabel("Predicted values")
                    plt.title("Predicted values in relation with INDEX")
                    plt.grid(True)
                    st.pyplot(plt)

                elif st.button("Cancel", key="Cancel_button_d"):
                    # Display a cancellation message
                    st.warning("The operation was canceled!")

            
        
            elif model_type == "WHP_LSTM Model":
                st.subheader("Forecast of WHP")
                # Remove the 'START_DATE' column
                # Load the data
                data = df_daily

                # Convert the 'START_DATE' column to datetime with the original format
                data['START_DATE'] = pd.to_datetime(data['START_DATE'])
                data['jour'] = data['START_DATE'].dt.day
                data['month'] = data['START_DATE'].dt.month
                data['year'] = data['START_DATE'].dt.year

                supprimer_doublons = st.sidebar.checkbox("Delete repeated data")
                if supprimer_doublons:
                    # Identify duplicate rows
                    doublons = data.duplicated()
                    # Remove duplicate rows
                    data = data[~doublons]

                # Handling of missing values
                handle_nan = st.sidebar.checkbox("Handling of missing values NAN", key="checkbox_whp")
                if not handle_nan:
                    st.warning("Handling of missing values")
                else:
                    handle_nan = st.sidebar.selectbox("Choose", options=["None", "Delete data point that contain NAN", "Replace with the precedent value", "Replace NAN with avrg"])
                    if handle_nan == "Delete data point that contain NAN":
                        data = data.dropna()
                        data2 = data2.dropna()
                    elif handle_nan == "Replace with the precedent value":
                        data = data.fillna(method='pad')
                        data2 = data2.fillna(method='pad')
                    elif handle_nan == "Replace NAN with avrg":
                        data = data.fillna(data.mean())
                        data2 = data2.fillna(data.mean())

                # Normalize the data using standardization
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(data[['WHP']])

                # Prepare input and output sequences for model training
                def create_sequences(data, seq_size):
                    X, y = [], []
                    for i in range(len(data) - seq_size):
                        X.append(data[i:i + seq_size])
                        y.append(data[i + seq_size])
                    return np.array(X), np.array(y)

                seq_size = seq_size

                # Prepare input and output sequences for model training
                train_data, test_data = train_test_split(scaled_data, test_size=0.2, shuffle=False)

                train_X, train_y = create_sequences(train_data, seq_size)
                test_X, test_y = create_sequences(test_data, seq_size)

                # Reshape the input data to have a dimension of 3
                train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
                test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))

                # Create and train the LSTM model
                model = Sequential()
                model.add(LSTM(num_input_neurons, input_shape=(seq_size, 1)))  # 1 for the WHP column
                for i in range(num_hidden_layers):
                    model.add(Dense(num_neurons))
                model.add(Dense(num_output_neurons))  # 1 for the WHP column
                model.compile(loss='mean_squared_error', optimizer='adam')
                model.summary()

                if st.button("Model training"):

                    # Training the model
                    if model is not None:
                        if use_epochs and use_batch_size:
                            model.fit(train_X, train_y, epochs=num_epochs, batch_size=batch_size, verbose=2)
                        elif use_epochs:
                            model.fit(train_X, train_y, epochs=num_epochs, verbose=2)
                        elif use_batch_size:
                            model.fit(train_X, train_y, batch_size=batch_size, verbose=2)
                        else:
                            model.fit(train_X, train_y, verbose=2)

                            # Make predictions on the test set
                    test_predictions = model.predict(test_X)
                   
                    # Inverse scaling of predictions
                    test_predictions = scaler.inverse_transform(test_predictions)

                    # Inverse scaling of test_y
                    test_y = scaler.inverse_transform(test_y)

                    # Calculate evaluation metrics on the test set
                    r2 = r2_score(test_y, test_predictions)
                    mse = mean_squared_error(test_y, test_predictions)
                    mae = mean_absolute_error(test_y, test_predictions)

                    st.write("R2 Score (WHP):", r2)
                    st.write("Mean Squared Error (WHP):", mse)
                    st.write("Mean Absolute Error (WHP):", mae)


                     # Prendre les 7 dernières valeurs pour prédire la 8ème
                    last_sequence = scaled_data[-seq_size:]
                    last_sequence = np.reshape(last_sequence, (1, seq_size, 1))
                    
                    for _ in range(nbr_valeur):  # Prédire 7 nouvelles valeurs
                        # Faire une prédiction en utilisant la séquence actuelle
                        next_value = model.predict(last_sequence)[0][0]
                        # Ajouter la nouvelle prédiction à la liste des prédictions
                    
                        # Mettre à jour la séquence précédente en éliminant la première valeur et en ajoutant la prédiction actuelle
                        last_sequence = np.append(last_sequence[:, 1:, :], [[[next_value]]], axis=1)
                        # Convertir la prédiction normalisée en valeur brute
                        next_value = scaler.inverse_transform([[next_value]])[0][0]
                        predictions_whp.append(next_value)
                        # Afficher la prédiction actuelle
                        st.write("Forcast", _ + 1, ":", next_value)


                    # Plot the graph of predicted values
                    plt.plot(predictions_whp[-nbr_valeur:], color='orange')
                    plt.xlabel("INDEX")
                    plt.ylabel("Predicted values")
                    plt.title("Predicted values in relation with INDEX")
                    plt.grid(True)
                    st.pyplot(plt)

                elif st.button("Cancel", key="cancel_button_d"):
                    # Display a cancellation message
                    st.warning("The operation was canceled!")
                    

        if not st.button("Click here to make new predictions", key="button"):
              st.warning("Please select the FLP and WHP models before clicking on the button to make new predictions.")
        else:
              sendInputs(predictions_flp,predictions_whp,data,nbr_valeur,global_var,data2)