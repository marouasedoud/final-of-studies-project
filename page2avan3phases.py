import streamlit as st
import subprocess
from trySendModel import model_general
# Installation des bibliothèques nécessaires
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN

# the global var that contains the three models
# ************************ !!!!! DONT'T TOUCH THIIIIIIIIIIIS !!!!****************************   
# the lines : 0- simple + general , 3- critique + general , 6- non critique + general , 9- simple + water
#             1- simple + oil     , 4- critique + oil     , 7- non critique + oil     , 10- critique + water
#             2- simple + gaz     , 5- critique + gaz     , 8- non critique  + gaz    , 11- non critique + water
#                                            (0:model, 1:scalerX, 2:scalerY)
#the columns:[0]RANDOM   [1]ANN   [2]RNN
global_var = [[[0,0,0], [0,0,0], [0,0,0]],
              [[0,0,0], [0,0,0], [0,0,0]],
              [[0,0,0], [0,0,0], [0,0,0]],

              [[0,0,0], [0,0,0], [0,0,0]],
              [[0,0,0], [0,0,0], [0,0,0]],
              [[0,0,0], [0,0,0], [0,0,0]],

              [[0,0,0], [0,0,0], [0,0,0]],
              [[0,0,0], [0,0,0], [0,0,0]],
              [[0,0,0], [0,0,0], [0,0,0]],
              
              [[0,0,0], [0,0,0], [0,0,0]],
              [[0,0,0], [0,0,0], [0,0,0]],
              [[0,0,0], [0,0,0], [0,0,0]]]



def show_data_page(df_jaugeage):
    # Afficher le logo dans le coin supérieur gauche de la page principale
    st.sidebar.image("so3.png", width=100)
    # Afficher les données
    st.sidebar.title("Menu")
    button_clicked = st.sidebar.button("Display wells gauging dataset", key="gauging_data_button")
    if button_clicked:
        st.subheader("Gauging dataset")
        st.write(df_jaugeage)

    show_algorithm_options(df_jaugeage)

def show_algorithm_options(df_jaugeage):
    dj = df_jaugeage
    st.sidebar.title("AI Model Configuration")
    algorithm_options = ["None", "ANN", "RFR", "RNN"]
    algorithm = st.sidebar.selectbox("Select AI model", algorithm_options)

    if algorithm in ["ANN", "RNN" , "RFR"]:
     
     model = None
     # Déclarer et initialiser la variable test_size
     test_size = 0.2  # Valeur par défaut, peut être modifiée ensuite
     n_estimators = 100
     
     configurer_modele = st.sidebar.checkbox("Choose size of test dataset")

     if not configurer_modele:
        st.warning("Choose size of test dataset")
     else:
      test_size = st.sidebar.number_input("Size of test dataset %", min_value=0.0, max_value=1.0, step=0.01, value=0.2)
      model = Sequential()

    if algorithm in ["ANN"]:
      num_input_neurons = 128 
      num_output_neurons = 2
      num_hidden_layers = 1
      input_activation = "relu"
      num_neurons = 64 
      output_activation = "relu"
      hidden_activation  = "relu"
     
      # Couche d'entrée
      use_input_layer = st.sidebar.checkbox("Select an input layer")
      if not use_input_layer and model is not None:
          st.warning("Select an input layer")
      elif use_input_layer and model is not None:
          num_input_neurons = st.sidebar.number_input("Numbre of neurons in the input layer", min_value=1, step=1, value=128)
          input_activation = st.sidebar.selectbox("Activation function of the input layer", options=['relu', 'sigmoid', 'tanh'])
          model.add(Dense(num_input_neurons, input_dim=7, activation=input_activation))

      # Couches cachées
      use_hidden_layer = st.sidebar.checkbox("Choose the number of hidden layers")
      if not use_hidden_layer and model is not None:
          st.warning("Choose the number of hidden layers")
      elif use_hidden_layer and model is not None:
          num_hidden_layers = st.sidebar.number_input("Number of hidden layers", min_value=1, step=1, value=1)
          for i in range(num_hidden_layers):
              num_neurons = st.sidebar.number_input(f"Number of neurons in the hidden layer {i+1}", min_value=1, step=1, value=64)
              hidden_activation = st.sidebar.selectbox(f"Activation function of the hidden layer {i+1}", options=['relu', 'sigmoid', 'tanh'])
              model.add(Dense(num_neurons, activation=hidden_activation))

      # Couche de sortie
      use_output_layer = st.sidebar.checkbox("Select an output layer")
      if not use_output_layer and model is not None:
          st.warning("Select an output layer")
      elif use_output_layer and model is not None:
          num_output_neurons = st.sidebar.number_input("Number of neurons in the output layer", min_value=1, step=1, value=2)
          output_activation = st.sidebar.selectbox("Activation function of the output layer", options=['linear', 'sigmoid'])
          model.add(Dense(num_output_neurons, activation=output_activation))

      # Ajouter une option pour le nombre d'époques
      use_epochs = st.sidebar.checkbox("Choose the number of epochs")
      if use_epochs:
          num_epochs = st.sidebar.number_input("Number of epochs", min_value=1, step=1, value=150)

      # Ajouter une option pour la taille du batch
      use_batch_size = st.sidebar.checkbox("Choose the size of batch")
      if use_batch_size:
          batch_size = st.sidebar.number_input("Size of batch", min_value=1, step=1, value=32)

    if algorithm in ["RNN"]:
      num_input_neurons = 128 
      num_output_neurons = 2
      num_hidden_layers = 1
      input_activation = "relu"
      num_neurons = 64 
      output_activation = "relu"
      hidden_activation  = "relu"
      model = Sequential()

    # Couche d'entrée
      use_input_layer = st.sidebar.checkbox("Select an input layer")

      if not use_input_layer and model is not None:
          st.warning("Select an input layer")
      elif use_input_layer and model is not None:
        num_input_neurons = st.sidebar.number_input("Numbre of neurons in the input layer", min_value=1, step=1, value=6)
        input_activation = st.sidebar.selectbox("Activation function of the input layer", options=['relu', 'sigmoid', 'tanh'])
        model.add(SimpleRNN(num_input_neurons, input_shape=(1, 7), activation= input_activation, name="input_layer"))


    # Couches cachées
      use_hidden_layer = st.sidebar.checkbox("Choose the number of hidden layers")
      if not use_hidden_layer and model is not None:
          st.warning("Choose the number of hidden layers")
      elif use_hidden_layer and model is not None:
        num_hidden_layers = st.sidebar.number_input("Number of hidden layers", min_value=1, step=1, value=1)
        for i in range(num_hidden_layers):
            num_neurons = st.sidebar.number_input(f"Number of neurons in the hidden layer {i+1}", min_value=1, step=1, value=64)
            hidden_activation = st.sidebar.selectbox(f"Activation function of the hidden layer {i+1}", options=['relu', 'sigmoid', 'tanh'])
            model.add(Dense(num_neurons, activation=hidden_activation, name=f"couche_cachee_{i+1}"))

    # Couche de sortie
      use_output_layer = st.sidebar.checkbox("Select an output layer")
      if not use_output_layer and model is not None:
          st.warning("Select an output layer")
      elif use_output_layer and model is not None:
        num_output_neurons = st.sidebar.number_input("Number of neurons in the output layer", min_value=1, step=1, value=2)
        output_activation = st.sidebar.selectbox("Activation function of the output layer", options=['linear', 'sigmoid'])
        model.add(Dense(num_output_neurons, activation=output_activation, name="couche_sortie"))

            # Ajouter une option pour le nombre d'époques
      use_epochs = st.sidebar.checkbox("Choose the number of epochs")
      if use_epochs:
          num_epochs = st.sidebar.number_input("Number of epochs", min_value=1, step=1, value=150)

      # Ajouter une option pour la taille du batch
      use_batch_size = st.sidebar.checkbox("Choose the size of batch")
      if use_batch_size:
          batch_size = st.sidebar.number_input("Size of batch", min_value=1, step=1, value=32)
         
    if algorithm in ["RFR"]:
      n_estimators = st.sidebar.checkbox("Choose the number of estimations")
      if not n_estimators and model is not None:
         st.warning("Choose the number of estimations")
      elif  n_estimators and model is not None:
            n_estimators = st.sidebar.number_input("Number of estimations", min_value=1, value=100, step=1)

    st.sidebar.title("Select Flow Model ")
    model_options = ["None", "All regim of flow Model ", "Critical flow Model", "NonCritical flow Model "]
    model_type = st.sidebar.selectbox("Select Flow Model ", model_options)

    if algorithm == "ANN":
        st.subheader("IA ANN")
        st.write("Type : ", model_type)

        if model_type == "All regim of flow Model ":
            st.sidebar.title("Select Fluid Model")
            model_selection = st.sidebar.selectbox("Select Fluid Model", ["None", "3Phase  Model", "2Phase  Model", "Oil Model", "GAS Model", "Water Model"])

            if model_selection == "3Phase  Model":
                st.write("ANN - 3Phase  Model")
                data= df_jaugeage

                #Convertir la colonne 'START_DATE' en datetime avec le format d'origine
                data['START_DATE'] = pd.to_datetime(data['START_DATE'])
                data['jour'] = data['START_DATE'].dt.day
                data['month'] = data['START_DATE'].dt.month
                data['year'] = data['START_DATE'].dt.year

                # Checkbox pour Delete repeted data
                supprimer_doublons = st.sidebar.checkbox("Delete repeted data")
                if supprimer_doublons:
                  # Identifier les lignes en double
                  doublons = data.duplicated()
                  # Supprimer les lignes en double
                  data = data[~doublons]
                
                # Gestion des valeurs manquantes
                handle_nan = st.sidebar.checkbox("Handling of missing values NAN", key="checkbox_nang")
                if not handle_nan:
                    st.warning("Handling of missing values NAN")
                else:
                    handle_nan = st.sidebar.selectbox("Choose", options=["None", "Delete data point that contain NAN", "Replace with the precedent value", "Replace NAN with zero", "Replace NAN with avrg"])
                    if handle_nan == "Delete data point that contain NAN":
                        data = data.dropna()
                    elif handle_nan == "Replace with the precedent value":
                        data = data.fillna(method='pad')
                    elif handle_nan == "Replace NAN with zero":
                        data = data.fillna(0)
                    elif handle_nan == "Replace NAN with avrg":
                        data = data.fillna(data.mean())

                # Extraire les caractéristiques et la cible
                X = data[['ID', 'jour', 'month', 'year', 'CHOKE', 'WHP', 'FLP']]
                y = data[['quantiter oil', 'quantiter gaz', 'water']]

                # Normaliser les données en utilisant MinMaxScaler
                scaler_XA = MinMaxScaler()
                scaler_yA = MinMaxScaler()
                X = scaler_XA.fit_transform(X)
                y = scaler_yA.fit_transform(y)

                # Séparer le jeu de données en ensembles d'entraînement et de test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

                # Créer et Model training ANN
                model = Sequential()
                model.add(Dense(num_input_neurons, input_dim=7, activation=input_activation))
                for i in range(num_hidden_layers):
                    model.add(Dense(num_neurons, activation=hidden_activation))
                model.add(Dense(num_output_neurons, activation=output_activation))

                model.compile(loss='mean_squared_error', optimizer='adam')

                if st.button("Model training"):
                    
                      # Entraînement du modèle
                    history = None
                    if model is not None:
                          if use_epochs and use_batch_size:
                              history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_test, y_test))
                          elif use_epochs:
                              history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test))
                          elif use_batch_size:
                              history = model.fit(X_train, y_train, batch_size=batch_size, validation_data=(X_test, y_test))
                          else:
                              history = model.fit(X_train, y_train, validation_data=(X_test, y_test))
                    # save model's info
                    global_var[0][1][0] = model
                    global_var[0][1][1] = scaler_XA
                    global_var[0][1][2] = scaler_yA
                    # Faire des prédictions sur l'ensemble de test
                    y_pred_test = model.predict(X_test)
                    y_pred_train = model.predict(X_train)

                    # Inverser la normalisation des prédictions et des vraies valeurs
                    y_pred_test = scaler_yA.inverse_transform(y_pred_test)
                    y_test = scaler_yA.inverse_transform(y_test)

                    # Calculer les erreurs MSE et MAE pour chaque variable
                    mse_oil = mean_squared_error(y_test[:, 0], y_pred_test[:, 0])
                    mae_oil = mean_absolute_error(y_test[:, 0], y_pred_test[:, 0])
                    mse_gaz = mean_squared_error(y_test[:, 1], y_pred_test[:, 1])
                    mae_gaz = mean_absolute_error(y_test[:, 1], y_pred_test[:, 1])

                    # Afficher les résultats
                    st.write("MSE oil:", mse_oil)
                    st.write("MAE oil:", mae_oil)
                    st.write("MSE gaz:", mse_gaz)
                    st.write("MAE gaz:", mae_gaz)

                    # Calculer le Correlation coefficient R^2
                    R_squared_test = r2_score(y_test, y_pred_test)
                    R_squared_train = r2_score(y_train, y_pred_train)

                    st.write("Correlation coefficient R^2 (test set) : ", R_squared_test)
                    st.write("Correlation coefficient R^2 (training set) : ", R_squared_train)
                elif st.button("Cancel", key="cancel_button_g"):
                  # Afficher un message d'annulation
                  st.warning("The operation was canceled !")

            if model_selection == "2Phase  Model":
                st.write("ANN - 2Phase  Model")
                data= df_jaugeage

                #Convertir la colonne 'START_DATE' en datetime avec le format d'origine
                data['START_DATE'] = pd.to_datetime(data['START_DATE'])
                data['jour'] = data['START_DATE'].dt.day
                data['month'] = data['START_DATE'].dt.month
                data['year'] = data['START_DATE'].dt.year

                # Checkbox pour Delete repeted data
                supprimer_doublons = st.sidebar.checkbox("Delete repeted data")
                if supprimer_doublons:
                  # Identifier les lignes en double
                  doublons = data.duplicated()
                  # Supprimer les lignes en double
                  data = data[~doublons]
                
                # Gestion des valeurs manquantes
                handle_nan = st.sidebar.checkbox("Handling of missing values NAN", key="checkbox_nang")
                if not handle_nan:
                    st.warning("Handling of missing values NAN")
                else:
                    handle_nan = st.sidebar.selectbox("Choose", options=["None", "Delete data point that contain NAN", "Replace with the precedent value", "Replace NAN with zero", "Replace NAN with avrg"])
                    if handle_nan == "Delete data point that contain NAN":
                        data = data.dropna()
                    elif handle_nan == "Replace with the precedent value":
                        data = data.fillna(method='pad')
                    elif handle_nan == "Replace NAN with zero":
                        data = data.fillna(0)
                    elif handle_nan == "Replace NAN with avrg":
                        data = data.fillna(data.mean())

                # Extraire les caractéristiques et la cible
                X = data[['ID', 'jour', 'month', 'year', 'CHOKE', 'WHP', 'FLP']]
                y = data[['quantiter oil', 'quantiter gaz']]

                # Normaliser les données en utilisant MinMaxScaler
                scaler_XA = MinMaxScaler()
                scaler_yA = MinMaxScaler()
                X = scaler_XA.fit_transform(X)
                y = scaler_yA.fit_transform(y)

                # Séparer le jeu de données en ensembles d'entraînement et de test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

                # Créer et Model training ANN
                model = Sequential()
                model.add(Dense(num_input_neurons, input_dim=7, activation=input_activation))
                for i in range(num_hidden_layers):
                    model.add(Dense(num_neurons, activation=hidden_activation))
                model.add(Dense(num_output_neurons, activation=output_activation))

                model.compile(loss='mean_squared_error', optimizer='adam')

                if st.button("Model training"):
                    
                      # Entraînement du modèle
                    history = None
                    if model is not None:
                          if use_epochs and use_batch_size:
                              history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_test, y_test))
                          elif use_epochs:
                              history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test))
                          elif use_batch_size:
                              history = model.fit(X_train, y_train, batch_size=batch_size, validation_data=(X_test, y_test))
                          else:
                              history = model.fit(X_train, y_train, validation_data=(X_test, y_test))
                    # save model's info
                    global_var[0][1][0] = model
                    global_var[0][1][1] = scaler_XA
                    global_var[0][1][2] = scaler_yA
                    # Faire des prédictions sur l'ensemble de test
                    y_pred_test = model.predict(X_test)
                    y_pred_train = model.predict(X_train)

                    # Inverser la normalisation des prédictions et des vraies valeurs
                    y_pred_test = scaler_yA.inverse_transform(y_pred_test)
                    y_test = scaler_yA.inverse_transform(y_test)

                    # Calculer les erreurs MSE et MAE pour chaque variable
                    mse_oil = mean_squared_error(y_test[:, 0], y_pred_test[:, 0])
                    mae_oil = mean_absolute_error(y_test[:, 0], y_pred_test[:, 0])
                    mse_gaz = mean_squared_error(y_test[:, 1], y_pred_test[:, 1])
                    mae_gaz = mean_absolute_error(y_test[:, 1], y_pred_test[:, 1])

                    # Afficher les résultats
                    st.write("MSE oil:", mse_oil)
                    st.write("MAE oil:", mae_oil)
                    st.write("MSE gaz:", mse_gaz)
                    st.write("MAE gaz:", mae_gaz)

                    # Calculer le Correlation coefficient R^2
                    R_squared_test = r2_score(y_test, y_pred_test)
                    R_squared_train = r2_score(y_train, y_pred_train)

                    st.write("Correlation coefficient R^2 (test set) : ", R_squared_test)
                    st.write("Correlation coefficient R^2 (training set) : ", R_squared_train)
                elif st.button("Cancel", key="cancel_button_g"):
                  # Afficher un message d'annulation
                  st.warning("The operation was canceled !")

            elif model_selection == "Oil Model":
                st.write("ANN - Oil Model")

                data= df_jaugeage

                #Convertir la colonne 'START_DATE' en datetime avec le format d'origine
                data['START_DATE'] = pd.to_datetime(data['START_DATE'])
                data['jour'] = data['START_DATE'].dt.day
                data['month'] = data['START_DATE'].dt.month
                data['year'] = data['START_DATE'].dt.year

                # Checkbox pour Delete repeted data
                supprimer_doublons = st.sidebar.checkbox("Delete repeted data")
                if supprimer_doublons:
                  # Identifier les lignes en double
                  doublons = data.duplicated()
                  # Supprimer les lignes en double
                  data = data[~doublons]
                
                    # Gestion des valeurs manquantes
                handle_nan = st.sidebar.checkbox("Handling of missing values NAN", key="checkbox_nanp")
                if not handle_nan:
                    st.warning("Handling of missing values NAN")
                else:
                    handle_nan = st.sidebar.selectbox("Choose", options=["None", "Delete data point that contain NAN", "Replace with the precedent value", "Replace NAN with zero", "Replace NAN with avrg"])
                    if handle_nan == "Delete data point that contain NAN":
                        data = data.dropna()
                    elif handle_nan == "Replace with the precedent value":
                        data = data.fillna(method='pad')
                    elif handle_nan == "Replace NAN with zero":
                        data = data.fillna(0)
                    elif handle_nan == "Replace NAN with avrg":
                        data = data.fillna(data.mean())
                  
              
                # Extraire les caractéristiques et la cible
                X = data[[ 'ID','jour', 'month', 'year', 'CHOKE', 'WHP', 'FLP']]
                y = data[['quantiter oil']]

                # Normaliser les données en utilisant MinMaxScaler
                scaler_XA = MinMaxScaler()
                scaler_yA = MinMaxScaler()
                X = scaler_XA.fit_transform(X)
                y = scaler_yA.fit_transform(y)

                # Séparer le jeu de données en ensembles d'entraînement et de test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

                 # Créer et Model training ANN
                model = Sequential()
                model.add(Dense(num_input_neurons, input_dim=7, activation= input_activation))
                for i in range(num_hidden_layers):
                  model.add(Dense(num_neurons, activation= hidden_activation))
                model.add(Dense(num_output_neurons, activation= output_activation))
               
                model.compile(loss='mean_squared_error', optimizer='adam')

                if st.button("Model training"):
                       # Entraînement du modèle
                    history = None
                    if model is not None:
                          if use_epochs and use_batch_size:
                              history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_test, y_test))
                          elif use_epochs:
                              history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test))
                          elif use_batch_size:
                              history = model.fit(X_train, y_train, batch_size=batch_size, validation_data=(X_test, y_test))
                          else:
                              history = model.fit(X_train, y_train, validation_data=(X_test, y_test))
                    
                    # save model's info
                    global_var[1][1][0] = model
                    global_var[1][1][1] = scaler_XA
                    global_var[1][1][2] = scaler_yA
                    # Faire des prédictions sur l'ensemble de test
                    y_pred_test = model.predict(X_test)
                    y_pred_test = y_pred_test.reshape(-1, 1)
                    y_pred_train = model.predict(X_train)
                    y_pred_train = y_pred_train.reshape(-1, 1)


                    # Inverser la normalisation des prédictions et des vraies valeurs
                    y_pred_test = scaler_yA.inverse_transform(y_pred_test)
                    y_test = scaler_yA.inverse_transform(y_test)

                    # Calculer l'erreur quadratique moyenne des prédictions
                    mse_test = mean_squared_error(y_test, y_pred_test)
                    st.write("Mean squared error (test set):", mse_test)

                    # Calculer l'erreur absolue moyenne des prédictions
                    mae_test = mean_absolute_error(y_test, y_pred_test)
                    st.write("Mean Absolute Error (test set):", mae_test)

                    # Calculer le Correlation coefficient R^2
                    R_squared_test = r2_score(y_test, y_pred_test)
                    R_squared_train = r2_score(y_train, y_pred_train)

                    st.write("Correlation coefficient R^2 (test set) : ", R_squared_test)
                    st.write("Correlation coefficient R^2 (training set) : ", R_squared_train)
                elif st.button("Cancel", key="cancel_button_g"):
                  # Afficher un message d'annulation
                  st.warning("The operation was canceled !")

            elif model_selection == "GAS Model":
                st.write("ANN - GAS Model")
                data= df_jaugeage

                #Convertir la colonne 'START_DATE' en datetime avec le format d'origine
                data['START_DATE'] = pd.to_datetime(data['START_DATE'])
                data['jour'] = data['START_DATE'].dt.day
                data['month'] = data['START_DATE'].dt.month
                data['year'] = data['START_DATE'].dt.year

                # Checkbox pour Delete repeted data
                supprimer_doublons = st.sidebar.checkbox("Delete repeted data")
                if supprimer_doublons:
                  # Identifier les lignes en double
                  doublons = data.duplicated()
                  # Supprimer les lignes en double
                  data = data[~doublons]
                
                    # Gestion des valeurs manquantes
                handle_nan = st.sidebar.checkbox("Handling of missing values NAN", key="checkbox_nang")
                if not handle_nan:
                    st.warning("Handling of missing values NAN")
                else:
                    handle_nan = st.sidebar.selectbox("Choose", options=["None", "Delete data point that contain NAN", "Replace with the precedent value", "Replace NAN with zero", "Replace NAN with avrg"])
                    if handle_nan == "Delete data point that contain NAN":
                        data = data.dropna()
                    elif handle_nan == "Replace with the precedent value":
                        data = data.fillna(method='pad')
                    elif handle_nan == "Replace NAN with zero":
                        data = data.fillna(0)
                    elif handle_nan == "Replace NAN with avrg":
                        data = data.fillna(data.mean())
                
                # Extraire les caractéristiques et la cible
                X = data[['ID','jour', 'month', 'year', 'CHOKE', 'WHP', 'FLP']]
                y = data[['quantiter gaz']]

                # Normaliser les données en utilisant MinMaxScaler
                scaler_XA = MinMaxScaler()
                scaler_yA = MinMaxScaler()
                X = scaler_XA.fit_transform(X)
                y = scaler_yA.fit_transform(y)

                # Séparer le jeu de données en ensembles d'entraînement et de test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

                 # Créer et Model training ANN
                model = Sequential()
                model.add(Dense(num_input_neurons, input_dim=7, activation= input_activation))
                for i in range(num_hidden_layers):
                  model.add(Dense(num_neurons, activation= hidden_activation))
                model.add(Dense(num_output_neurons, activation= output_activation))
               
                model.compile(loss='mean_squared_error', optimizer='adam')

                if st.button("Model training"):
                       # Entraînement du modèle
                    history = None
                    if model is not None:
                          if use_epochs and use_batch_size:
                              history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_test, y_test))
                          elif use_epochs:
                              history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test))
                          elif use_batch_size:
                              history = model.fit(X_train, y_train, batch_size=batch_size, validation_data=(X_test, y_test))
                          else:
                              history = model.fit(X_train, y_train, validation_data=(X_test, y_test))

                    # save model's info
                    global_var[2][1][0] = model
                    global_var[2][1][1] = scaler_XA
                    global_var[2][1][2] = scaler_yA
                    # Faire des prédictions sur l'ensemble de test
                    y_pred_test = model.predict(X_test)
                    y_pred_test = y_pred_test.reshape(-1, 1)
                    y_pred_train = model.predict(X_train)
                    y_pred_train = y_pred_train.reshape(-1, 1)


                    # Inverser la normalisation des prédictions et des vraies valeurs
                    y_pred_test = scaler_yA.inverse_transform(y_pred_test)
                    y_test = scaler_yA.inverse_transform(y_test)

                    # Calculer l'erreur quadratique moyenne des prédictions
                    mse_test = mean_squared_error(y_test, y_pred_test)
                    st.write("Mean squared error (test set):", mse_test)

                    # Calculer l'erreur absolue moyenne des prédictions
                    mae_test = mean_absolute_error(y_test, y_pred_test)
                    st.write("Mean Absolute Error (test set):", mae_test)

                    # Calculer le Correlation coefficient R^2
                    R_squared_test = r2_score(y_test, y_pred_test)
                    R_squared_train = r2_score(y_train, y_pred_train)

                    st.write("Correlation coefficient R^2 (test set) : ", R_squared_test)
                    st.write("Correlation coefficient R^2 (training set) : ", R_squared_train)
                
                elif st.button("Cancel", key="cancel_button_g"):
                  # Afficher un message d'annulation
                  st.warning("The operation was canceled !")

            elif model_selection == "Water Model":
                st.write("ANN - Water Model")
                data= df_jaugeage

                #Convertir la colonne 'START_DATE' en datetime avec le format d'origine
                data['START_DATE'] = pd.to_datetime(data['START_DATE'])
                data['jour'] = data['START_DATE'].dt.day
                data['month'] = data['START_DATE'].dt.month
                data['year'] = data['START_DATE'].dt.year


                 # Checkbox pour Bootstrapping
                doubler_lignes = st.sidebar.checkbox("Bootstrapping")

                if doubler_lignes:
                  # Bootstrapping en ajoutant le DataFrame à lui-même
                  # Double the dataset by duplicating the rows
                  data = pd.concat([data, data], ignore_index=True)
                #   data = data.append(data, ignore_index=True)
                
                    # Gestion des valeurs manquantes
                handle_nan = st.sidebar.checkbox("Handling of missing values NAN", key="checkbox_nanw")
                if not handle_nan:
                    st.warning("Handling of missing values NAN")
                else:
                    handle_nan = st.sidebar.selectbox("Choose", options=["None", "Delete data point that contain NAN", "Replace with the precedent value", "Replace NAN with avrg"])
                    if handle_nan == "Delete data point that contain NAN":
                        data = data.dropna()
                    elif handle_nan == "Replace with the precedent value":
                        data = data.fillna(method='pad')
                    elif handle_nan == "Replace NAN with avrg":
                        data = data.fillna(data.mean())
                
                # Extraire les caractéristiques et la cible
                X = data[['ID','jour', 'month', 'year', 'CHOKE', 'WHP', 'FLP']]
                y = data[['water']]

                # Normaliser les données en utilisant MinMaxScaler
                scaler_XA = MinMaxScaler()
                scaler_yA = MinMaxScaler()
                X = scaler_XA.fit_transform(X)
                y = scaler_yA.fit_transform(y)

                # Séparer le jeu de données en ensembles d'entraînement et de test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

                 # Créer et Model training ANN
                model = Sequential()
                model.add(Dense(num_input_neurons, input_dim=7, activation= input_activation))
                for i in range(num_hidden_layers):
                  model.add(Dense(num_neurons, activation= hidden_activation))
                model.add(Dense(num_output_neurons, activation= output_activation))
               
                model.compile(loss='mean_squared_error', optimizer='adam')

                if st.button("Model training"):
                       # Entraînement du modèle
                    history = None
                    if model is not None:
                          if use_epochs and use_batch_size:
                              history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_test, y_test))
                          elif use_epochs:
                              history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test))
                          elif use_batch_size:
                              history = model.fit(X_train, y_train, batch_size=batch_size, validation_data=(X_test, y_test))
                          else:
                              history = model.fit(X_train, y_train, validation_data=(X_test, y_test))

                    # # save model's info
                    global_var[9][1][0] = model
                    global_var[9][1][1] = scaler_XA
                    global_var[9][1][2] = scaler_yA
                    # Faire des prédictions sur l'ensemble de test
                    y_pred_test = model.predict(X_test)
                    y_pred_test = y_pred_test.reshape(-1, 1)
                    y_pred_train = model.predict(X_train)
                    y_pred_train = y_pred_train.reshape(-1, 1)


                    # Inverser la normalisation des prédictions et des vraies valeurs
                    y_pred_test = scaler_yA.inverse_transform(y_pred_test)
                    y_test = scaler_yA.inverse_transform(y_test)

                    # Calculer l'erreur quadratique moyenne des prédictions
                    mse_test = mean_squared_error(y_test, y_pred_test)
                    st.write("Mean squared error (test set):", mse_test)

                    # Calculer l'erreur absolue moyenne des prédictions
                    mae_test = mean_absolute_error(y_test, y_pred_test)
                    st.write("Mean Absolute Error (test set):", mae_test)

                    # Calculer le Correlation coefficient R^2
                    R_squared_test = r2_score(y_test, y_pred_test)
                    R_squared_train = r2_score(y_train, y_pred_train)

                    st.write("Correlation coefficient R^2 (test set) : ", R_squared_test)
                    st.write("Correlation coefficient R^2 (training set) : ", R_squared_train)
                
                elif st.button("Cancel", key="cancel_button_g"):
                  # Afficher un message d'annulation
                  st.warning("The operation was canceled !")
                
        if model_type == "Critical flow Model":
              st.sidebar.title("Select Fluid Model")
              model_selection = st.sidebar.selectbox("Select Fluid Model", ("None","2Phase  Critical Model", "Oil Critical Model  ", "Gas Critical Model  ", "Water Critical Model "))
              if model_selection == "2Phase  Critical Model  ":
                st.write("ANN - 2Phase  Critical Model  ")
            
                data= df_jaugeage

                #Convertir la colonne 'START_DATE' en datetime avec le format d'origine
                data['START_DATE'] = pd.to_datetime(data['START_DATE'])
                data['jour'] = data['START_DATE'].dt.day
                data['month'] = data['START_DATE'].dt.month
                data['year'] = data['START_DATE'].dt.year

                # Checkbox pour Delete repeted data
                supprimer_doublons = st.sidebar.checkbox("Delete repeted data")
                if supprimer_doublons:
                  # Identifier les lignes en double
                  doublons = data.duplicated()
                  # Supprimer les lignes en double
                  data = data[~doublons]
                
                  # Gestion des valeurs manquantes
                handle_nan = st.sidebar.checkbox("Handling of missing values NAN", key="checkbox_nancg")
                if not handle_nan:
                    st.warning("Handling of missing values NAN")
                else:
                    handle_nan = st.sidebar.selectbox("Choose", options=["None", "Delete data point that contain NAN", "Replace with the precedent value", "Replace NAN with zero", "Replace NAN with avrg"])
                    if handle_nan == "Delete data point that contain NAN":
                        data = data.dropna()
                    elif handle_nan == "Replace with the precedent value":
                        data = data.fillna(method='pad')
                    elif handle_nan == "Replace NAN with zero":
                        data = data.fillna(0)
                    elif handle_nan == "Replace NAN with avrg":
                        data = data.fillna(data.mean())

                data = data.drop(data[data['FLP'] / data['WHP'] > 0.75].index)

              
                # Extraire les caractéristiques et la cible
                X = data[[ 'ID','jour', 'month', 'year', 'CHOKE', 'WHP', 'FLP']]
                y = data[['quantiter oil', 'quantiter gaz']]

                # Normaliser les données en utilisant MinMaxScaler
                scaler_XA = MinMaxScaler()
                scaler_yA = MinMaxScaler()
                X = scaler_XA.fit_transform(X)
                y = scaler_yA.fit_transform(y)

                # Séparer le jeu de données en ensembles d'entraînement et de test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size, random_state=42)

                # Créer et Model training ANN
                model = Sequential()
                model.add(Dense(num_input_neurons, input_dim=7, activation= input_activation))
                for i in range(num_hidden_layers):
                  model.add(Dense(num_neurons, activation= hidden_activation))
                # model.add(Dense(32, activation='relu'))
                model.add(Dense(num_output_neurons, activation= output_activation))
               
                model.compile(loss='mean_squared_error', optimizer='adam')

                if st.button("Model training"):
                       # Entraînement du modèle
                    history = None
                    if model is not None:
                          if use_epochs and use_batch_size:
                              history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_test, y_test))
                          elif use_epochs:
                              history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test))
                          elif use_batch_size:
                              history = model.fit(X_train, y_train, batch_size=batch_size, validation_data=(X_test, y_test))
                          else:
                              history = model.fit(X_train, y_train, validation_data=(X_test, y_test))
                    
                    # save model's info
                    global_var[3][1][0] = model
                    global_var[3][1][1] = scaler_XA
                    global_var[3][1][2] = scaler_yA
                    # Faire des prédictions sur l'ensemble de test
                    y_pred_test = model.predict(X_test)
                    y_pred_train = model.predict(X_train)

                    # Inverser la normalisation des prédictions et des vraies valeurs
                    y_pred_test = scaler_yA.inverse_transform(y_pred_test)
                    y_test = scaler_yA.inverse_transform(y_test)

                    # Calculer les erreurs MSE et MAE pour chaque variable
                    mse_oil = mean_squared_error(y_test[:, 0], y_pred_test[:, 0])
                    mae_oil = mean_absolute_error(y_test[:, 0], y_pred_test[:, 0])
                    mse_gaz = mean_squared_error(y_test[:, 1], y_pred_test[:, 1])
                    mae_gaz = mean_absolute_error(y_test[:, 1], y_pred_test[:, 1])

                    # Afficher les résultats
                    st.write("MSE oil:", mse_oil)
                    st.write("MAE oil:", mae_oil)
                    st.write("MSE gaz:", mse_gaz)
                    st.write("MAE gaz:", mae_gaz)

                    # Calculer le Correlation coefficient R^2
                    R_squared_test = r2_score(y_test, y_pred_test)
                    R_squared_train = r2_score(y_train, y_pred_train)

                    st.write("Correlation coefficient R^2 (test set) : ", R_squared_test)
                    st.write("Correlation coefficient R^2 (training set) : ", R_squared_train)
                
                elif st.button("Cancel", key="cancel_button_g"):
                  # Afficher un message d'annulation
                  st.warning("The operation was canceled !")
                
              elif model_selection == "Oil Critical Model  ":
                st.write("ANN - Oil Critical Model  ")
                data= df_jaugeage

                #Convertir la colonne 'START_DATE' en datetime avec le format d'origine
                data['START_DATE'] = pd.to_datetime(data['START_DATE'])
                data['jour'] = data['START_DATE'].dt.day
                data['month'] = data['START_DATE'].dt.month
                data['year'] = data['START_DATE'].dt.year

                # Checkbox pour Delete repeted data
                supprimer_doublons = st.sidebar.checkbox("Delete repeted data")
                if supprimer_doublons:
                  # Identifier les lignes en double
                  doublons = data.duplicated()
                  # Supprimer les lignes en double
                  data = data[~doublons]
                
                    # Gestion des valeurs manquantes
                handle_nan = st.sidebar.checkbox("Handling of missing values NAN", key="checkbox_nanco")
                if not handle_nan:
                    st.warning("Handling of missing values NAN")
                else:
                    handle_nan = st.sidebar.selectbox("Choose", options=["None", "Delete data point that contain NAN", "Replace with the precedent value", "Replace NAN with zero", "Replace NAN with avrg"])
                    if handle_nan == "Delete data point that contain NAN":
                        data = data.dropna()
                    elif handle_nan == "Replace with the precedent value":
                        data = data.fillna(method='pad')
                    elif handle_nan == "Replace NAN with zero":
                        data = data.fillna(0)
                    elif handle_nan == "Replace NAN with avrg":
                        data = data.fillna(data.mean())

                data = data.drop(data[data['FLP'] / data['WHP'] > 0.75].index)

              
                # Extraire les caractéristiques et la cible
                X = data[['ID','jour', 'month', 'year', 'CHOKE', 'WHP', 'FLP']]
                y = data[['quantiter oil']]

                # Normaliser les données en utilisant MinMaxScaler
                scaler_XA = MinMaxScaler()
                scaler_yA = MinMaxScaler()
                X = scaler_XA.fit_transform(X)
                y = scaler_yA.fit_transform(y)

                # Séparer le jeu de données en ensembles d'entraînement et de test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

                 # Créer et Model training ANN
                model = Sequential()
                model.add(Dense(num_input_neurons, input_dim=7, activation= input_activation))
                for i in range(num_hidden_layers):
                  model.add(Dense(num_neurons, activation= hidden_activation))
                model.add(Dense(num_output_neurons, activation= output_activation))
               
                model.compile(loss='mean_squared_error', optimizer='adam')

                if st.button("Model training"):
                       # Entraînement du modèle
                    history = None
                    if model is not None:
                          if use_epochs and use_batch_size:
                              history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_test, y_test))
                          elif use_epochs:
                              history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test))
                          elif use_batch_size:
                              history = model.fit(X_train, y_train, batch_size=batch_size, validation_data=(X_test, y_test))
                          else:
                              history = model.fit(X_train, y_train, validation_data=(X_test, y_test))

                    # save model's info
                    global_var[4][1][0] = model
                    global_var[4][1][1] = scaler_XA
                    global_var[4][1][2] = scaler_yA
                    # Faire des prédictions sur l'ensemble de test
                    y_pred_test = model.predict(X_test)
                    y_pred_test = y_pred_test.reshape(-1, 1)
                    y_pred_train = model.predict(X_train)
                    y_pred_train = y_pred_train.reshape(-1, 1)


                    # Inverser la normalisation des prédictions et des vraies valeurs
                    y_pred_test = scaler_yA.inverse_transform(y_pred_test)
                    y_test = scaler_yA.inverse_transform(y_test)

                    # Calculer l'erreur quadratique moyenne des prédictions
                    mse_test = mean_squared_error(y_test, y_pred_test)
                    st.write("Mean squared error (test set):", mse_test)

                    # Calculer l'erreur absolue moyenne des prédictions
                    mae_test = mean_absolute_error(y_test, y_pred_test)
                    st.write("Mean Absolute Error (test set):", mae_test)

                    # Calculer le Correlation coefficient R^2
                    R_squared_test = r2_score(y_test, y_pred_test)
                    R_squared_train = r2_score(y_train, y_pred_train)

                    st.write("Correlation coefficient R^2 (test set) : ", R_squared_test)
                    st.write("Correlation coefficient R^2 (training set) : ", R_squared_train)
                
                elif st.button("Cancel", key="cancel_button_g"):
                  # Afficher un message d'annulation
                  st.warning("The operation was canceled !")

              elif model_selection == "Gas Critical Model  ":
                st.write("ANN - Gas Critical Model  ")
                data= df_jaugeage

                #Convertir la colonne 'START_DATE' en datetime avec le format d'origine
                data['START_DATE'] = pd.to_datetime(data['START_DATE'])
                data['jour'] = data['START_DATE'].dt.day
                data['month'] = data['START_DATE'].dt.month
                data['year'] = data['START_DATE'].dt.year

                # Checkbox pour Delete repeted data
                supprimer_doublons = st.sidebar.checkbox("Delete repeted data")
                if supprimer_doublons:
                  # Identifier les lignes en double
                  doublons = data.duplicated()
                  # Supprimer les lignes en double
                  data = data[~doublons]
                
                    # Gestion des valeurs manquantes
                handle_nan = st.sidebar.checkbox("Handling of missing values NAN", key="checkbox_nancg")
                if not handle_nan:
                    st.warning("Handling of missing values NAN")
                else:
                    handle_nan = st.sidebar.selectbox("Choose", options=["None", "Delete data point that contain NAN", "Replace with the precedent value", "Replace NAN with zero", "Replace NAN with avrg"])
                    if handle_nan == "Delete data point that contain NAN":
                        data = data.dropna()
                    elif handle_nan == "Replace with the precedent value":
                        data = data.fillna(method='pad')
                    elif handle_nan == "Replace NAN with zero":
                        data = data.fillna(0)
                    elif handle_nan == "Replace NAN with avrg":
                        data = data.fillna(data.mean())
                
                data = data.drop(data[data['FLP'] / data['WHP'] > 0.75].index)

              
                # Extraire les caractéristiques et la cible
                X = data[['ID','jour', 'month', 'year', 'CHOKE', 'WHP', 'FLP']]
                y = data[['quantiter gaz']]

                # Normaliser les données en utilisant MinMaxScaler
                scaler_XA = MinMaxScaler()
                scaler_yA = MinMaxScaler()
                X = scaler_XA.fit_transform(X)
                y = scaler_yA.fit_transform(y)

                # Séparer le jeu de données en ensembles d'entraînement et de test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

                 # Créer et Model training ANN
                model = Sequential()
                model.add(Dense(num_input_neurons, input_dim=7, activation= input_activation))
                for i in range(num_hidden_layers):
                  model.add(Dense(num_neurons, activation= hidden_activation))
                model.add(Dense(num_output_neurons, activation= output_activation))
               
                model.compile(loss='mean_squared_error', optimizer='adam')

                if st.button("Model training"):
                       # Entraînement du modèle
                    history = None
                    if model is not None:
                          if use_epochs and use_batch_size:
                              history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_test, y_test))
                          elif use_epochs:
                              history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test))
                          elif use_batch_size:
                              history = model.fit(X_train, y_train, batch_size=batch_size, validation_data=(X_test, y_test))
                          else:
                              history = model.fit(X_train, y_train, validation_data=(X_test, y_test))

                    # save model's info
                    global_var[5][1][0] = model
                    global_var[5][1][1] = scaler_XA
                    global_var[5][1][2] = scaler_yA
                    # Faire des prédictions sur l'ensemble de test
                    y_pred_test = model.predict(X_test)
                    y_pred_test = y_pred_test.reshape(-1, 1)
                    y_pred_train = model.predict(X_train)
                    y_pred_train = y_pred_train.reshape(-1, 1)


                    # Inverser la normalisation des prédictions et des vraies valeurs
                    y_pred_test = scaler_yA.inverse_transform(y_pred_test)
                    y_test = scaler_yA.inverse_transform(y_test)

                    # Calculer l'erreur quadratique moyenne des prédictions
                    mse_test = mean_squared_error(y_test, y_pred_test)
                    st.write("Mean squared error (test set):", mse_test)

                    # Calculer l'erreur absolue moyenne des prédictions
                    mae_test = mean_absolute_error(y_test, y_pred_test)
                    st.write("Mean Absolute Error (test set):", mae_test)

                    # Calculer le Correlation coefficient R^2
                    R_squared_test = r2_score(y_test, y_pred_test)
                    R_squared_train = r2_score(y_train, y_pred_train)

                    st.write("Correlation coefficient R^2 (test set) : ", R_squared_test)
                    st.write("Correlation coefficient R^2 (training set) : ", R_squared_train)
                
                elif st.button("Cancel", key="cancel_button_g"):
                  # Afficher un message d'annulation
                  st.warning("The operation was canceled !")

              elif model_selection == "Water Critical Model ":
                st.write("ANN - Water Critical Model ")
                data= df_jaugeage

                #Convertir la colonne 'START_DATE' en datetime avec le format d'origine
                data['START_DATE'] = pd.to_datetime(data['START_DATE'])
                data['jour'] = data['START_DATE'].dt.day
                data['month'] = data['START_DATE'].dt.month
                data['year'] = data['START_DATE'].dt.year

                  # Checkbox pour Bootstrapping
                doubler_lignes = st.sidebar.checkbox("Bootstrapping")

                if doubler_lignes:
                  # Bootstrapping en ajoutant le DataFrame à lui-même
                  # Double the dataset by duplicating the rows
                  data = pd.concat([data, data], ignore_index=True)
                #   data = data.append(data, ignore_index=True)


                
                  # Gestion des valeurs manquantes
                handle_nan = st.sidebar.checkbox("Handling of missing values NAN", key="checkbox_nancw")
                if not handle_nan:
                    st.warning("Handling of missing values NAN")
                else:
                    handle_nan = st.sidebar.selectbox("Choose", options=["None", "Delete data point that contain NAN", "Replace with the precedent value", "Replace NAN with avrg"])
                    if handle_nan == "Delete data point that contain NAN":
                        data = data.dropna()
                    elif handle_nan == "Replace with the precedent value":
                        data = data.fillna(method='pad')
                    elif handle_nan == "Replace NAN with avrg":
                        data = data.fillna(data.mean())

                data = data.drop(data[data['FLP'] / data['WHP'] > 0.75].index)

              
                # Extraire les caractéristiques et la cible
                X = data[['ID','jour', 'month', 'year', 'CHOKE', 'WHP', 'FLP']]
                y = data[['water']]

                # Normaliser les données en utilisant MinMaxScaler
                scaler_XA = MinMaxScaler()
                scaler_yA = MinMaxScaler()
                X = scaler_XA.fit_transform(X)
                y = scaler_yA.fit_transform(y)

                # Séparer le jeu de données en ensembles d'entraînement et de test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

                 # Créer et Model training ANN
                model = Sequential()
                model.add(Dense(num_input_neurons, input_dim=7, activation= input_activation))
                for i in range(num_hidden_layers):
                  model.add(Dense(num_neurons, activation= hidden_activation))
                model.add(Dense(num_output_neurons, activation= output_activation))
               
                model.compile(loss='mean_squared_error', optimizer='adam')

                if st.button("Model training"):
                       # Entraînement du modèle
                    history = None
                    if model is not None:
                          if use_epochs and use_batch_size:
                              history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_test, y_test))
                          elif use_epochs:
                              history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test))
                          elif use_batch_size:
                              history = model.fit(X_train, y_train, batch_size=batch_size, validation_data=(X_test, y_test))
                          else:
                              history = model.fit(X_train, y_train, validation_data=(X_test, y_test))

                    # # save model's info
                    global_var[10][1][0] = model
                    global_var[10][1][1] = scaler_XA
                    global_var[10][1][2] = scaler_yA
                    # Faire des prédictions sur l'ensemble de test
                    y_pred_test = model.predict(X_test)
                    y_pred_test = y_pred_test.reshape(-1, 1)
                    y_pred_train = model.predict(X_train)
                    y_pred_train = y_pred_train.reshape(-1, 1)


                    # Inverser la normalisation des prédictions et des vraies valeurs
                    y_pred_test = scaler_yA.inverse_transform(y_pred_test)
                    y_test = scaler_yA.inverse_transform(y_test)

                    # Calculer l'erreur quadratique moyenne des prédictions
                    mse_test = mean_squared_error(y_test, y_pred_test)
                    st.write("Mean squared error (test set):", mse_test)

                    # Calculer l'erreur absolue moyenne des prédictions
                    mae_test = mean_absolute_error(y_test, y_pred_test)
                    st.write("Mean Absolute Error (test set):", mae_test)

                    # Calculer le Correlation coefficient R^2
                    R_squared_test = r2_score(y_test, y_pred_test)
                    R_squared_train = r2_score(y_train, y_pred_train)

                    st.write("Correlation coefficient R^2 (test set) : ", R_squared_test)
                    st.write("Correlation coefficient R^2 (training set) : ", R_squared_train)
                
                elif st.button("Cancel", key="cancel_button_g"):
                  # Afficher un message d'annulation
                  st.warning("The operation was canceled !")

        if model_type == "NonCritical flow Model ":
              st.sidebar.title("Select Fluid Model")
              model_selection = st.sidebar.selectbox("Select Fluid Model", ("None","2Phase  Noncritical Model", "Oil Noncritical Model", "Gas Noncritical Model", "Water Noncritical Model"))
              if model_selection == "2Phase  Noncritical Model":
                st.write("ANN - 2Phase  Noncritical Model")
            
                data= df_jaugeage

                #Convertir la colonne 'START_DATE' en datetime avec le format d'origine
                data['START_DATE'] = pd.to_datetime(data['START_DATE'])
                data['jour'] = data['START_DATE'].dt.day
                data['month'] = data['START_DATE'].dt.month
                data['year'] = data['START_DATE'].dt.year


                # Checkbox pour Delete repeted data
                supprimer_doublons = st.sidebar.checkbox("Delete repeted data")
                if supprimer_doublons:
                  # Identifier les lignes en double
                  doublons = data.duplicated()
                  # Supprimer les lignes en double
                  data = data[~doublons]
                
                    # Gestion des valeurs manquantes
                handle_nan = st.sidebar.checkbox("Handling of missing values NAN", key="checkbox_nanncg")
                if not handle_nan:
                    st.warning("Handling of missing values NAN")
                else:
                    handle_nan = st.sidebar.selectbox("Choose", options=["None", "Delete data point that contain NAN", "Replace with the precedent value", "Replace NAN with zero", "Replace NAN with avrg"])
                    if handle_nan == "Delete data point that contain NAN":
                        data = data.dropna()
                    elif handle_nan == "Replace with the precedent value":
                        data = data.fillna(method='pad')
                    elif handle_nan == "Replace NAN with zero":
                        data = data.fillna(0)
                    elif handle_nan == "Replace NAN with avrg":
                        data = data.fillna(data.mean())

                data = data.drop(data[data['FLP'] / data['WHP'] <=  0.75].index)
              
                # Extraire les caractéristiques et la cible
                X = data[[ 'ID','jour', 'month', 'year', 'CHOKE', 'WHP', 'FLP']]
                y = data[['quantiter oil', 'quantiter gaz']]

                # Normaliser les données en utilisant MinMaxScaler
                scaler_XA = MinMaxScaler()
                scaler_yA = MinMaxScaler()
                X = scaler_XA.fit_transform(X)
                y = scaler_yA.fit_transform(y)

                # Séparer le jeu de données en ensembles d'entraînement et de test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size, random_state=42)

                # Créer et Model training ANN
                model = Sequential()
                model.add(Dense(num_input_neurons, input_dim=7, activation= input_activation))
                for i in range(num_hidden_layers):
                  model.add(Dense(num_neurons, activation= hidden_activation))
                # model.add(Dense(32, activation='relu'))
                model.add(Dense(num_output_neurons, activation= output_activation))
               
                model.compile(loss='mean_squared_error', optimizer='adam')

                if st.button("Model training"):
                       # Entraînement du modèle
                    history = None
                    if model is not None:
                          if use_epochs and use_batch_size:
                              history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_test, y_test))
                          elif use_epochs:
                              history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test))
                          elif use_batch_size:
                              history = model.fit(X_train, y_train, batch_size=batch_size, validation_data=(X_test, y_test))
                          else:
                              history = model.fit(X_train, y_train, validation_data=(X_test, y_test))

                    # save model's info
                    global_var[6][1][0] = model
                    global_var[6][1][1] = scaler_XA
                    global_var[6][1][2] = scaler_yA
                    # Faire des prédictions sur l'ensemble de test
                    y_pred_test = model.predict(X_test)
                    y_pred_train = model.predict(X_train)

                    # Inverser la normalisation des prédictions et des vraies valeurs
                    y_pred_test = scaler_yA.inverse_transform(y_pred_test)
                    y_test = scaler_yA.inverse_transform(y_test)

                    # Calculer les erreurs MSE et MAE pour chaque variable
                    mse_oil = mean_squared_error(y_test[:, 0], y_pred_test[:, 0])
                    mae_oil = mean_absolute_error(y_test[:, 0], y_pred_test[:, 0])
                    mse_gaz = mean_squared_error(y_test[:, 1], y_pred_test[:, 1])
                    mae_gaz = mean_absolute_error(y_test[:, 1], y_pred_test[:, 1])

                    # Afficher les résultats
                    st.write("MSE oil:", mse_oil)
                    st.write("MAE oil:", mae_oil)
                    st.write("MSE gaz:", mse_gaz)
                    st.write("MAE gaz:", mae_gaz)

                    # Calculer le Correlation coefficient R^2
                    R_squared_test = r2_score(y_test, y_pred_test)
                    R_squared_train = r2_score(y_train, y_pred_train)

                    st.write("Correlation coefficient R^2 (test set) : ", R_squared_test)
                    st.write("Correlation coefficient R^2 (training set) : ", R_squared_train)

                elif st.button("Cancel", key="cancel_button_g"):
                  # Afficher un message d'annulation
                  st.warning("The operation was canceled !")
              
              elif model_selection == "Oil Noncritical Model":
                st.write("ANN - Oil Noncritical Model")
                data= df_jaugeage

                #Convertir la colonne 'START_DATE' en datetime avec le format d'origine
                data['START_DATE'] = pd.to_datetime(data['START_DATE'])
                data['jour'] = data['START_DATE'].dt.day
                data['month'] = data['START_DATE'].dt.month
                data['year'] = data['START_DATE'].dt.year

                # Checkbox pour Delete repeted data
                supprimer_doublons = st.sidebar.checkbox("Delete repeted data")
                if supprimer_doublons:
                  # Identifier les lignes en double
                  doublons = data.duplicated()
                  # Supprimer les lignes en double
                  data = data[~doublons]
                
                    # Gestion des valeurs manquantes
                handle_nan = st.sidebar.checkbox("Handling of missing values NAN", key="checkbox_nannco")
                if not handle_nan:
                    st.warning("Handling of missing values NAN")
                else:
                    handle_nan = st.sidebar.selectbox("Choose", options=["None", "Delete data point that contain NAN", "Replace with the precedent value", "Replace NAN with zero", "Replace NAN with avrg"])
                    if handle_nan == "Delete data point that contain NAN":
                        data = data.dropna()
                    elif handle_nan == "Replace with the precedent value":
                        data = data.fillna(method='pad')
                    elif handle_nan == "Replace NAN with zero":
                        data = data.fillna(0)
                    elif handle_nan == "Replace NAN with avrg":
                        data = data.fillna(data.mean())

                data = data.drop(data[data['FLP'] / data['WHP'] <=  0.75].index)

                # Extraire les caractéristiques et la cible
                X = data[['ID','jour', 'month', 'year', 'CHOKE', 'WHP', 'FLP']]
                y = data[['quantiter oil']]

                # Normaliser les données en utilisant MinMaxScaler
                scaler_XA = MinMaxScaler()
                scaler_yA = MinMaxScaler()
                X = scaler_XA.fit_transform(X)
                y = scaler_yA.fit_transform(y)

                # Séparer le jeu de données en ensembles d'entraînement et de test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

                 # Créer et Model training ANN
                model = Sequential()
                model.add(Dense(num_input_neurons, input_dim=7, activation= input_activation))
                for i in range(num_hidden_layers):
                  model.add(Dense(num_neurons, activation= hidden_activation))
                model.add(Dense(num_output_neurons, activation= output_activation))
               
                model.compile(loss='mean_squared_error', optimizer='adam')

                if st.button("Model training"):
                       # Entraînement du modèle
                    history = None
                    if model is not None:
                          if use_epochs and use_batch_size:
                              history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_test, y_test))
                          elif use_epochs:
                              history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test))
                          elif use_batch_size:
                              history = model.fit(X_train, y_train, batch_size=batch_size, validation_data=(X_test, y_test))
                          else:
                              history = model.fit(X_train, y_train, validation_data=(X_test, y_test))

                    # save model's info
                    global_var[7][1][0] = model
                    global_var[7][1][1] = scaler_XA
                    global_var[7][1][2] = scaler_yA
                    # Faire des prédictions sur l'ensemble de test
                    y_pred_test = model.predict(X_test)
                    y_pred_test = y_pred_test.reshape(-1, 1)
                    y_pred_train = model.predict(X_train)
                    y_pred_train = y_pred_train.reshape(-1, 1)


                    # Inverser la normalisation des prédictions et des vraies valeurs
                    y_pred_test = scaler_yA.inverse_transform(y_pred_test)
                    y_test = scaler_yA.inverse_transform(y_test)

                    # Calculer l'erreur quadratique moyenne des prédictions
                    mse_test = mean_squared_error(y_test, y_pred_test)
                    st.write("Mean squared error (test set):", mse_test)

                    # Calculer l'erreur absolue moyenne des prédictions
                    mae_test = mean_absolute_error(y_test, y_pred_test)
                    st.write("Mean Absolute Error (test set):", mae_test)

                    # Calculer le Correlation coefficient R^2
                    R_squared_test = r2_score(y_test, y_pred_test)
                    R_squared_train = r2_score(y_train, y_pred_train)

                    st.write("Correlation coefficient R^2 (test set) : ", R_squared_test)
                    st.write("Correlation coefficient R^2 (training set) : ", R_squared_train)

                elif st.button("Cancel", key="cancel_button_g"):
                  # Afficher un message d'annulation
                  st.warning("The operation was canceled !")   

              elif model_selection == "Gas Noncritical Model":
                st.write("ANN - Gas Noncritical Model")
                data= df_jaugeage

                #Convertir la colonne 'START_DATE' en datetime avec le format d'origine
                data['START_DATE'] = pd.to_datetime(data['START_DATE'])
                data['jour'] = data['START_DATE'].dt.day
                data['month'] = data['START_DATE'].dt.month
                data['year'] = data['START_DATE'].dt.year

                # Checkbox pour Delete repeted data
                supprimer_doublons = st.sidebar.checkbox("Delete repeted data")
                if supprimer_doublons:
                  # Identifier les lignes en double
                  doublons = data.duplicated()
                  # Supprimer les lignes en double
                  data = data[~doublons]
                
                    # Gestion des valeurs manquantes
                handle_nan = st.sidebar.checkbox("Handling of missing values NAN", key="checkbox_nanncg")
                if not handle_nan:
                    st.warning("Handling of missing values NAN")
                else:
                    handle_nan = st.sidebar.selectbox("Choose", options=["None", "Delete data point that contain NAN", "Replace with the precedent value", "Replace NAN with zero", "Replace NAN with avrg"])
                    if handle_nan == "Delete data point that contain NAN":
                        data = data.dropna()
                    elif handle_nan == "Replace with the precedent value":
                        data = data.fillna(method='pad')
                    elif handle_nan == "Replace NAN with zero":
                        data = data.fillna(0)
                    elif handle_nan == "Replace NAN with avrg":
                        data = data.fillna(data.mean())

                data = data.drop(data[data['FLP'] / data['WHP'] <=  0.75].index)

                # Extraire les caractéristiques et la cible
                X = data[['ID','jour', 'month', 'year', 'CHOKE', 'WHP', 'FLP']]
                y = data[['quantiter gaz']]

                # Normaliser les données en utilisant MinMaxScaler
                scaler_XA = MinMaxScaler()
                scaler_yA = MinMaxScaler()
                X = scaler_XA.fit_transform(X)
                y = scaler_yA.fit_transform(y)

                # Séparer le jeu de données en ensembles d'entraînement et de test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

                 # Créer et Model training ANN
                model = Sequential()
                model.add(Dense(num_input_neurons, input_dim=7, activation= input_activation))
                for i in range(num_hidden_layers):
                  model.add(Dense(num_neurons, activation= hidden_activation))
                model.add(Dense(num_output_neurons, activation= output_activation))
               
                model.compile(loss='mean_squared_error', optimizer='adam')

                if st.button("Model training"):
                       # Entraînement du modèle
                    history = None
                    if model is not None:
                          if use_epochs and use_batch_size:
                              history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_test, y_test))
                          elif use_epochs:
                              history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test))
                          elif use_batch_size:
                              history = model.fit(X_train, y_train, batch_size=batch_size, validation_data=(X_test, y_test))
                          else:
                              history = model.fit(X_train, y_train, validation_data=(X_test, y_test))

                    # save model's info
                    global_var[8][1][0] = model
                    global_var[8][1][1] = scaler_XA
                    global_var[8][1][2] = scaler_yA
                    # Faire des prédictions sur l'ensemble de test
                    y_pred_test = model.predict(X_test)
                    y_pred_test = y_pred_test.reshape(-1, 1)
                    y_pred_train = model.predict(X_train)
                    y_pred_train = y_pred_train.reshape(-1, 1)


                    # Inverser la normalisation des prédictions et des vraies valeurs
                    y_pred_test = scaler_yA.inverse_transform(y_pred_test)
                    y_test = scaler_yA.inverse_transform(y_test)

                    # Calculer l'erreur quadratique moyenne des prédictions
                    mse_test = mean_squared_error(y_test, y_pred_test)
                    st.write("Mean squared error (test set):", mse_test)

                    # Calculer l'erreur absolue moyenne des prédictions
                    mae_test = mean_absolute_error(y_test, y_pred_test)
                    st.write("Mean Absolute Error (test set):", mae_test)

                    # Calculer le Correlation coefficient R^2
                    R_squared_test = r2_score(y_test, y_pred_test)
                    R_squared_train = r2_score(y_train, y_pred_train)

                    st.write("Correlation coefficient R^2 (test set) : ", R_squared_test)
                    st.write("Correlation coefficient R^2 (training set) : ", R_squared_train)
                
                elif st.button("Cancel", key="cancel_button_g"):
                  # Afficher un message d'annulation
                  st.warning("The operation was canceled !")

              elif model_selection == "Water Noncritical Model":
                st.write("ANN - Water Noncritical Model")
                data= df_jaugeage

                #Convertir la colonne 'START_DATE' en datetime avec le format d'origine
                data['START_DATE'] = pd.to_datetime(data['START_DATE'])
                data['jour'] = data['START_DATE'].dt.day
                data['month'] = data['START_DATE'].dt.month
                data['year'] = data['START_DATE'].dt.year

                 # Checkbox pour Bootstrapping
                doubler_lignes = st.sidebar.checkbox("Bootstrapping")

                if doubler_lignes:
                  # Bootstrapping en ajoutant le DataFrame à lui-même
                  # Double the dataset by duplicating the rows
                  data = pd.concat([data, data], ignore_index=True)
                #   data = data.append(data, ignore_index=True)

                
                    # Gestion des valeurs manquantes
                handle_nan = st.sidebar.checkbox("Handling of missing values NAN", key="checkbox_nanncw")
                if not handle_nan:
                    st.warning("Handling of missing values NAN")
                else:
                    handle_nan = st.sidebar.selectbox("Choose", options=["None", "Delete data point that contain NAN", "Replace with the precedent value", "Replace NAN with avrg"])
                    if handle_nan == "Delete data point that contain NAN":
                        data = data.dropna()
                    elif handle_nan == "Replace with the precedent value":
                        data = data.fillna(method='pad')
                    elif handle_nan == "Replace NAN with avrg":
                        data = data.fillna(data.mean())

                data = data.drop(data[data['FLP'] / data['WHP'] <=  0.75].index)
              
                # Extraire les caractéristiques et la cible
                X = data[['ID','jour', 'month', 'year', 'CHOKE', 'WHP', 'FLP']]
                y = data[['water']]

                # Normaliser les données en utilisant MinMaxScaler
                scaler_XA = MinMaxScaler()
                scaler_yA = MinMaxScaler()
                X = scaler_XA.fit_transform(X)
                y = scaler_yA.fit_transform(y)

                # Séparer le jeu de données en ensembles d'entraînement et de test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

                 # Créer et Model training ANN
                model = Sequential()
                model.add(Dense(num_input_neurons, input_dim=7, activation= input_activation))
                for i in range(num_hidden_layers):
                  model.add(Dense(num_neurons, activation= hidden_activation))
                model.add(Dense(num_output_neurons, activation= output_activation))
               
                model.compile(loss='mean_squared_error', optimizer='adam')

                if st.button("Model training"):
                       # Entraînement du modèle
                    history = None
                    if model is not None:
                          if use_epochs and use_batch_size:
                              history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_test, y_test))
                          elif use_epochs:
                              history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test))
                          elif use_batch_size:
                              history = model.fit(X_train, y_train, batch_size=batch_size, validation_data=(X_test, y_test))
                          else:
                              history = model.fit(X_train, y_train, validation_data=(X_test, y_test))

                    # # save model's info
                    global_var[11][1][0] = model
                    global_var[11][1][1] = scaler_XA
                    global_var[11][1][2] = scaler_yA
                    # Faire des prédictions sur l'ensemble de test
                    y_pred_test = model.predict(X_test)
                    y_pred_test = y_pred_test.reshape(-1, 1)
                    y_pred_train = model.predict(X_train)
                    y_pred_train = y_pred_train.reshape(-1, 1)


                    # Inverser la normalisation des prédictions et des vraies valeurs
                    y_pred_test = scaler_yA.inverse_transform(y_pred_test)
                    y_test = scaler_yA.inverse_transform(y_test)

                    # Calculer l'erreur quadratique moyenne des prédictions
                    mse_test = mean_squared_error(y_test, y_pred_test)
                    st.write("Mean squared error (test set):", mse_test)

                    # Calculer l'erreur absolue moyenne des prédictions
                    mae_test = mean_absolute_error(y_test, y_pred_test)
                    st.write("Mean Absolute Error (test set):", mae_test)

                    # Calculer le Correlation coefficient R^2
                    R_squared_test = r2_score(y_test, y_pred_test)
                    R_squared_train = r2_score(y_train, y_pred_train)

                    st.write("Correlation coefficient R^2 (test set) : ", R_squared_test)
                    st.write("Correlation coefficient R^2 (training set) : ", R_squared_train)
                
                elif st.button("Cancel", key="cancel_button_g"):
                  # Afficher un message d'annulation
                  st.warning("The operation was canceled !")
                 
    elif algorithm == "RFR":
        # Ajoutez le code pour l'algorithme RFR ici
      st.subheader("IA RFR")
      st.write("Type : ", model_type)
      if model_type == "All regim of flow Model ":
         st.sidebar.title("Select Fluid Model")
         model_selection = st.sidebar.selectbox("Select Fluid Model", ("None","2Phase  Model", "Oil Model", "GAS Model", "Water Model"))
         if model_selection == "2Phase  Model":
            st.write("RFR - 2Phase  Model")
            data= df_jaugeage
             #Convertir la colonne 'START_DATE' en datetime avec le format d'origine
            data['START_DATE'] = pd.to_datetime(data['START_DATE'])
            data['jour'] = data['START_DATE'].dt.day
            data['month'] = data['START_DATE'].dt.month
            data['year'] = data['START_DATE'].dt.year

            # Checkbox pour Delete repeted data
            supprimer_doublons = st.sidebar.checkbox("Delete repeted data")
            if supprimer_doublons:
              # Identifier les lignes en double
              doublons = data.duplicated()
              # Supprimer les lignes en double
              data = data[~doublons]
            
             # Gestion des valeurs manquantes
            handle_nan = st.sidebar.checkbox("Handling of missing values NAN", key="checkbox_nan1")
            if not handle_nan:
                st.warning("Handling of missing values NAN")
            else:
                handle_nan = st.sidebar.selectbox("Choose", options=["None", "Delete data point that contain NAN", "Replace with the precedent value", "Replace NAN with zero", "Replace NAN with avrg"])
                if handle_nan == "Delete data point that contain NAN":
                    data = data.dropna()
                elif handle_nan == "Replace with the precedent value":
                    data = data.fillna(method='pad')
                elif handle_nan == "Replace NAN with zero":
                    data = data.fillna(0)
                elif handle_nan == "Replace NAN with avrg":
                    # data = data.select_dtypes(include=[np.number])  # Sélectionner uniquement les colonnes numériques
                    # data.fillna(data.mean(axis=0), inplace=True)
                    data = data.fillna(data.mean())
          
            # Extraire les caractéristiques et la cible
            X = data[[ 'ID','jour', 'month', 'year', 'CHOKE', 'WHP', 'FLP']]
            y = data[['quantiter oil','quantiter gaz']]

            # Normaliser les données
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
            X = scaler_X.fit_transform(X)
            y = scaler_y.fit_transform(y)

            # Séparer les données en ensembles d'entraînement et de test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size, random_state=42)

            # Créer et Model training RFR
            model = RandomForestRegressor(n_estimators= n_estimators, random_state=42)
            is_model_trained = False

            if st.button("Model training"):
               
               model.fit(X_train, y_train)
               # save RANDOM's model infos
               global_var[0][0][0] = model
               global_var[0][0][1] = scaler_X
               global_var[0][0][2] = scaler_y
               is_model_trained = True

               # Faire des prédictions sur l'ensemble de test
               y_pred = model.predict(X_test)
              #  y_pred = y_pred.reshape(-1, 1)
               y_pred2 = model.predict(X_train)
              #  y_pred2 = y_pred2.reshape(-1, 1)

               # Inverser la normalisation des prédictions et des vraies valeurs
               y_pred = scaler_y.inverse_transform(y_pred)
               y_test = scaler_y.inverse_transform(y_test)

              # Calculer les erreurs MSE et MAE pour chaque variable
               mse_oil = mean_squared_error(y_test[:, 0], y_pred[:, 0])
               mae_oil = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
               mse_gaz = mean_squared_error(y_test[:, 1], y_pred[:, 1])
               mae_gaz = mean_absolute_error(y_test[:, 1], y_pred[:, 1])

              # Afficher les résultats
               st.write("MSE oil:", mse_oil)
               st.write("MAE oil:", mae_oil)
               st.write("MSE gaz:", mse_gaz)
               st.write("MAE gaz:", mae_gaz)

              # Calculer le Correlation coefficient R^2
               R_squared = r2_score(y_test, y_pred)
               R_squared2 = r2_score(y_train, y_pred2)
               st.write("Correlation coefficient R^2 pour test : ", R_squared )
               st.write("Correlation coefficient R^2 pour train : ", R_squared2 )
            
            elif st.button("Cancel", key="cancel_button_g"):
               # Afficher un message d'annulation
               st.warning("The operation was canceled !")

         elif model_selection == "Oil Model":
            st.write("RFR - Oil Model")
            data= df_jaugeage

             #Convertir la colonne 'START_DATE' en datetime avec le format d'origine
            data['START_DATE'] = pd.to_datetime(data['START_DATE'])
            data['jour'] = data['START_DATE'].dt.day
            data['month'] = data['START_DATE'].dt.month
            data['year'] = data['START_DATE'].dt.year

            # Checkbox pour Delete repeted data
            supprimer_doublons = st.sidebar.checkbox("Delete repeted data")
            if supprimer_doublons:
              # Identifier les lignes en double
              doublons = data.duplicated()
              # Supprimer les lignes en double
              data = data[~doublons]
            
             # Gestion des valeurs manquantes
            handle_nan = st.sidebar.checkbox("Handling of missing values NAN", key="checkbox_nan2")
            if not handle_nan:
                st.warning("Handling of missing values NAN")
            else:
                handle_nan = st.sidebar.selectbox("Choose", options=["None", "Delete data point that contain NAN", "Replace with the precedent value", "Replace NAN with zero", "Replace NAN with avrg"])
                if handle_nan == "Delete data point that contain NAN":
                    data = data.dropna()
                elif handle_nan == "Replace with the precedent value":
                    data = data.fillna(method='pad')
                elif handle_nan == "Replace NAN with zero":
                    data = data.fillna(0)
                elif handle_nan == "Replace NAN with avrg":
                    data = data.fillna(data.mean())
          
            # Extraire les caractéristiques et la cible
            X = data[[ 'ID','jour', 'month', 'year', 'CHOKE', 'WHP', 'FLP']]
            y = data[['quantiter oil']]

            # Normaliser les données
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
            X = scaler_X.fit_transform(X)
            y = scaler_y.fit_transform(y)

            # Séparer les données en ensembles d'entraînement et de test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            # Créer et Model training RFR
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)

            if st.button("Model training"):

                model.fit(X_train, y_train)
                # save RANDOM's model infos
                global_var[1][0][0] = model
                global_var[1][0][1] = scaler_X
                global_var[1][0][2] = scaler_y

                # Faire des prédictions sur l'ensemble de test
                y_pred = model.predict(X_test)
                y_pred = y_pred.reshape(-1, 1)  # Reshape des prédictions

                y_pred2 = model.predict(X_train)
                y_pred2 = y_pred2.reshape(-1, 1)  # Reshape des prédictions

                  # Inverser la normalisation des prédictions et des vraies valeurs
                y_pred = scaler_y.inverse_transform(y_pred)
                y_test = scaler_y.inverse_transform(y_test)

                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                st.write("MAE:", mae)
                st.write("MSE:", mse)

              
                # Calculer le Correlation coefficient R^2
                R_squared = r2_score(y_test, y_pred)
                R_squared2 = r2_score(y_train, y_pred2)
                st.write("Correlation coefficient R^2 pour test : ", R_squared )
                st.write("Correlation coefficient R^2 pour train : ", R_squared2 )

            elif st.button("Cancel", key="cancel_button_g"):
                  # Afficher un message d'annulation
                  st.warning("The operation was canceled !") 
            
         elif model_selection == "GAS Model":
            st.write("RFR - GAS Model")
            data= df_jaugeage

             #Convertir la colonne 'START_DATE' en datetime avec le format d'origine
            data['START_DATE'] = pd.to_datetime(data['START_DATE'])
            data['jour'] = data['START_DATE'].dt.day
            data['month'] = data['START_DATE'].dt.month
            data['year'] = data['START_DATE'].dt.year
            # data['START_DATE'] = data['START_DATE'].astype(np.int64)/10**9


            # Checkbox pour Delete repeted data
            supprimer_doublons = st.sidebar.checkbox("Delete repeted data")
            if supprimer_doublons:
              # Identifier les lignes en double
              doublons = data.duplicated()
              # Supprimer les lignes en double
              data = data[~doublons]
            
             # Gestion des valeurs manquantes
            handle_nan = st.sidebar.checkbox("Handling of missing values NAN", key="checkbox_nan3")
            if not handle_nan:
                st.warning("Handling of missing values NAN")
            else:
                handle_nan = st.sidebar.selectbox("Choose", options=["None", "Delete data point that contain NAN", "Replace with the precedent value", "Replace NAN with zero", "Replace NAN with avrg"])
                if handle_nan == "Delete data point that contain NAN":
                    data = data.dropna()
                elif handle_nan == "Replace with the precedent value":
                    data = data.fillna(method='pad')
                elif handle_nan == "Replace NAN with zero":
                    data = data.fillna(0)
                elif handle_nan == "Replace NAN with avrg":
                    data = data.fillna(data.mean())

            # Extraire les caractéristiques et la cible
            X = data[[ 'ID','jour', 'month', 'year', 'CHOKE', 'WHP', 'FLP']]
            y = data[[ 'quantiter gaz']]

            # Normaliser les données
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
            X = scaler_X.fit_transform(X)
            y = scaler_y.fit_transform(y)

            # Séparer les données en ensembles d'entraînement et de test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            # Créer et Model training RFR
            model = RandomForestRegressor(n_estimators= n_estimators, random_state=42)


            if st.button("Model training"):

              model.fit(X_train, y_train)
              # save RANDOM's model infos
              global_var[2][0][0] = model
              global_var[2][0][1] = scaler_X
              global_var[2][0][2] = scaler_y

              # Faire des prédictions sur l'ensemble de test
              y_pred = model.predict(X_test)
              y_pred = y_pred.reshape(-1, 1)  # Reshape des prédictions

              y_pred2 = model.predict(X_train)
              y_pred2 = y_pred2.reshape(-1, 1)  # Reshape des prédictions

              # Inverser la normalisation des prédictions et des vraies valeurs
              y_pred = scaler_y.inverse_transform(y_pred)
              y_test = scaler_y.inverse_transform(y_test)

              mae = mean_absolute_error(y_test, y_pred)
              mse = mean_squared_error(y_test, y_pred)
              st.write("MAE:", mae)
              st.write("MSE:", mse)

              
              # Calculer le Correlation coefficient R^2
              R_squared = r2_score(y_test, y_pred)
              R_squared2 = r2_score(y_train, y_pred2)
              st.write("Correlation coefficient R^2 pour test : ", R_squared )
              st.write("Correlation coefficient R^2 pour train : ", R_squared2 )

            elif st.button("Cancel", key="cancel_button_g"):
                  # Afficher un message d'annulation
                  st.warning("The operation was canceled !")

         elif model_selection == "Water Model":
            st.write("RFR - Water Model")
            data= df_jaugeage

             #Convertir la colonne 'START_DATE' en datetime avec le format d'origine
            data['START_DATE'] = pd.to_datetime(data['START_DATE'])
            data['jour'] = data['START_DATE'].dt.day
            data['month'] = data['START_DATE'].dt.month
            data['year'] = data['START_DATE'].dt.year

            # Checkbox pour Bootstrapping
            doubler_lignes = st.sidebar.checkbox("Bootstrapping")

            if doubler_lignes:
              # Bootstrapping en ajoutant le DataFrame à lui-même
              # Double the dataset by duplicating the rows
              data = pd.concat([data, data], ignore_index=True)
              #   data = data.append(data, ignore_index=True)
            
            # Gestion des valeurs manquantes
            handle_nan = st.sidebar.checkbox("Handling of missing values NAN", key="checkbox_nan4")
            if not handle_nan:
                st.warning("Handling of missing values NAN")
            else:
                handle_nan = st.sidebar.selectbox("Choose", options=["None", "Supprimer les lignes qui contient NAN", "Replace with the precedent value", "Replace NAN with avrg"])
                if handle_nan == "Supprimer les lignes qui contient NAN":
                    data = data.dropna()
                elif handle_nan == "Replace with the precedent value":
                    # data = data.fillna(method='ffill')
                    data = data.fillna(method='pad')
                elif handle_nan == "Replace NAN with avrg":
                    data = data.fillna(data.mean())
          
            # Extraire les caractéristiques et la cible
            X = data[[ 'ID','jour', 'month', 'year', 'CHOKE', 'WHP', 'FLP']]
            y = data[[ 'water']]

            # Normaliser les données
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
            X = scaler_X.fit_transform(X)
            y = scaler_y.fit_transform(y)

            # Séparer les données en ensembles d'entraînement et de test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            # Créer et Model training RFR
            model = RandomForestRegressor(n_estimators= n_estimators, random_state=42)


            if st.button("Model training"):

                model.fit(X_train, y_train)
                # # save RANDOM's model infos
                global_var[9][0][0] = model
                global_var[9][0][1] = scaler_X
                global_var[9][0][2] = scaler_y

                # Faire des prédictions sur l'ensemble de test
                y_pred = model.predict(X_test)
                y_pred = y_pred.reshape(-1, 1)  # Reshape des prédictions

                y_pred2 = model.predict(X_train)
                y_pred2 = y_pred2.reshape(-1, 1)  # Reshape des prédictions

                  # Inverser la normalisation des prédictions et des vraies valeurs
                y_pred = scaler_y.inverse_transform(y_pred)
                y_test = scaler_y.inverse_transform(y_test)

                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                st.write("MAE:", mae)
                st.write("MSE:", mse)

              
                # Calculer le Correlation coefficient R^2
                R_squared = r2_score(y_test, y_pred)
                R_squared2 = r2_score(y_train, y_pred2)
                st.write("Correlation coefficient R^2 pour test : ", R_squared )
                st.write("Correlation coefficient R^2 pour train : ", R_squared2 )

            elif st.button("Cancel", key="cancel_button_g"):
                  # Afficher un message d'annulation
                  st.warning("The operation was canceled !") 

      if model_type == "Critical flow Model":
              st.sidebar.title("Select Fluid Model")
              model_selection = st.sidebar.selectbox("Select Fluid Model", ("None","2Phase  Critical Model  ", "Oil Critical Model  ", "Gas Critical Model  ", "Water Critical Model "))
              if model_selection == "2Phase  Critical Model  ":
                st.write("RFR - 2Phase  Critical Model  ")
                data= df_jaugeage
                # data = data.drop(data[data['FLP'] / data['WHP'] > 0.75].index)
                  #Convertir la colonne 'START_DATE' en datetime avec le format d'origine
                data['START_DATE'] = pd.to_datetime(data['START_DATE'])
                data['jour'] = data['START_DATE'].dt.day
                data['month'] = data['START_DATE'].dt.month
                data['year'] = data['START_DATE'].dt.year
                
                # Checkbox pour Delete repeted data
                supprimer_doublons = st.sidebar.checkbox("Delete repeted data")
                if supprimer_doublons:
                  # Identifier les lignes en double
                  doublons = data.duplicated()
                  # Supprimer les lignes en double
                  data = data[~doublons]
                
                  # Gestion des valeurs manquantes
                  handle_nan = st.sidebar.checkbox("Handling of missing values NAN", key="checkbox_nan5")
                  if not handle_nan:
                      st.warning("Handling of missing values NAN")
                  else:
                      handle_nan = st.sidebar.selectbox("Choose", options=["None", "Delete data point that contain NAN", "Replace with the precedent value", "Replace NAN with zero", "Replace NAN with avrg"])
                      if handle_nan == "Delete data point that contain NAN":
                          data = data.dropna()
                      elif handle_nan == "Replace with the precedent value":
                          data = data.fillna(method='pad')
                      elif handle_nan == "Replace NAN with zero":
                          data = data.fillna(0)
                      elif handle_nan == "Replace NAN with avrg":
                          data = data.fillna(data.mean())

                data = data.drop(data[data['FLP'] / data['WHP'] > 0.75].index)
                      
                # Extraire les caractéristiques et la cible
                X = data[[ 'ID','jour', 'month', 'year', 'CHOKE', 'WHP', 'FLP']]
                y = data[['quantiter oil', 'quantiter gaz']]

                # Normalize the data using MinMaxScaler
                scaler_X = MinMaxScaler()
                scaler_y = MinMaxScaler()
                X = scaler_X.fit_transform(X)
                y = scaler_y.fit_transform(y)

                # Split the dataset into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

               # Create and train the RFR model
                model = RandomForestRegressor(n_estimators= n_estimators, random_state=42)

                if st.button("Model training"):

                  model.fit(X_train, y_train)
                  # save RANDOM's model infos
                  global_var[3][0][0] = model
                  global_var[3][0][1] = scaler_X
                  global_var[3][0][2] = scaler_y

                  # Make predictions on the test set
                  y_pred = model.predict(X_test)
                  y_pred2 = model.predict(X_train)

                  # Inverser la normalisation des prédictions et des vraies valeurs
                  y_pred = scaler_y.inverse_transform(y_pred)
                  y_test = scaler_y.inverse_transform(y_test)


                 # Calculer les erreurs MSE et MAE pour chaque variable
                  mse_oil = mean_squared_error(y_test[:, 0], y_pred[:, 0])
                  mae_oil = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
                  mse_gaz = mean_squared_error(y_test[:, 1], y_pred[:, 1])
                  mae_gaz = mean_absolute_error(y_test[:, 1], y_pred[:, 1])

                  # Afficher les résultats
                  st.write("MSE oil:", mse_oil)
                  st.write("MAE oil:", mae_oil)
                  st.write("MSE gaz:", mse_gaz)
                  st.write("MAE gaz:", mae_gaz)
                  # Calculer le Correlation coefficient R^2
                  R_squared_test = r2_score(y_test, y_pred)
                  R_squared_train = r2_score(y_train, y_pred2)

                  st.write("Correlation coefficient R^2 (test set) : ", R_squared_test)
                  st.write("Correlation coefficient R^2 (training set) : ", R_squared_train)
                
                elif st.button("Cancel", key="cancel_button_g"):
                  # Afficher un message d'annulation
                  st.warning("The operation was canceled !")

              elif model_selection == "Oil Critical Model  ":
                st.write("RFR - Oil Critical Model  ")
                data= df_jaugeage
                # data = data.drop(data[data['FLP'] / data['WHP'] > 0.75].index)
                  #Convertir la colonne 'START_DATE' en datetime avec le format d'origine
                data['START_DATE'] = pd.to_datetime(data['START_DATE'])
                data['jour'] = data['START_DATE'].dt.day
                data['month'] = data['START_DATE'].dt.month
                data['year'] = data['START_DATE'].dt.year

                # Checkbox pour Delete repeted data
                supprimer_doublons = st.sidebar.checkbox("Delete repeted data")
                if supprimer_doublons:
                  # Identifier les lignes en double
                  doublons = data.duplicated()
                  # Supprimer les lignes en double
                  data = data[~doublons]

                # Gestion des valeurs manquantes
                handle_nan = st.sidebar.checkbox("Handling of missing values NAN", key="checkbox_nan6")
                if not handle_nan:
                    st.warning("Handling of missing values NAN")
                else:
                    handle_nan = st.sidebar.selectbox("Choose", options=["None", "Delete data point that contain NAN", "Replace with the precedent value", "Replace NAN with zero", "Replace NAN with avrg"])
                    if handle_nan == "Delete data point that contain NAN":
                        data = data.dropna()
                    elif handle_nan == "Replace with the precedent value":
                        data = data.fillna(method='pad')
                    elif handle_nan == "Replace NAN with zero":
                        data = data.fillna(0)
                    elif handle_nan == "Replace NAN with avrg":
                        data = data.fillna(data.mean())
                
                data = data.drop(data[data['FLP'] / data['WHP'] > 0.75].index)
             

                # Split the dataset into input features (X) and target variables (y)
                X = data[[ 'ID','jour', 'month', 'year', 'CHOKE', 'WHP', 'FLP']]
                y = data[['quantiter oil']]

                # Normalize the data using MinMaxScaler
                scaler_X = MinMaxScaler()
                scaler_y = MinMaxScaler()
                X = scaler_X.fit_transform(X)
                y = scaler_y.fit_transform(y)

                # Split the dataset into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size, random_state=42)

               # Create and train the RFR model
                model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)

                if st.button("Model training"):

                  model.fit(X_train, y_train)
                  # save RANDOM's model infos
                  global_var[4][0][0] = model
                  global_var[4][0][1] = scaler_X
                  global_var[4][0][2] = scaler_y

                  # Make predictions on the test set
                  y_pred = model.predict(X_test)
                  y_pred = y_pred.reshape(-1, 1)
                  y_pred2 = model.predict(X_train)
                  y_pred2 = y_pred2.reshape(-1, 1)

                  # Inverser la normalisation des prédictions et des vraies valeurs
                  y_pred = scaler_y.inverse_transform(y_pred)
                  y_test = scaler_y.inverse_transform(y_test)

                  # Calculate the mean squared error of the predictions
                  mse = mean_squared_error(y_test, y_pred)
                  mae = mean_absolute_error(y_test, y_pred)

                  st.write("Mean squared error:", mse)
                  st.write("Mean Absolute Error:", mae)

                # Calculer le Correlation coefficient R^2
                  R_squared_test = r2_score(y_test, y_pred)
                  R_squared_train = r2_score(y_train, y_pred2)

                  st.write("Correlation coefficient R^2 (test set) : ", R_squared_test)
                  st.write("Correlation coefficient R^2 (training set) : ", R_squared_train)
                
                elif st.button("Cancel", key="cancel_button_g"):
                  # Afficher un message d'annulation
                  st.warning("The operation was canceled !")

              elif model_selection == "Gas Critical Model  ":
                st.write("RFR - Gas Critical Model  ")
                data= df_jaugeage
                # data = data.drop(data[data['FLP'] / data['WHP'] > 0.75].index)
                  #Convertir la colonne 'START_DATE' en datetime avec le format d'origine
                data['START_DATE'] = pd.to_datetime(data['START_DATE'])
                data['jour'] = data['START_DATE'].dt.day
                data['month'] = data['START_DATE'].dt.month
                data['year'] = data['START_DATE'].dt.year

                # Checkbox pour Delete repeted data
                supprimer_doublons = st.sidebar.checkbox("Delete repeted data")
                if supprimer_doublons:
                  # Identifier les lignes en double
                  doublons = data.duplicated()
                  # Supprimer les lignes en double
                  data = data[~doublons]


                # Gestion des valeurs manquantes
                handle_nan = st.sidebar.checkbox("Handling of missing values NAN", key="checkbox_nan7")
                if not handle_nan:
                    st.warning("Handling of missing values NAN")
                else:
                    handle_nan = st.sidebar.selectbox("Choose", options=["None", "Delete data point that contain NAN", "Replace with the precedent value", "Replace NAN with zero", "Replace NAN with avrg"])
                    if handle_nan == "Delete data point that contain NAN":
                        data = data.dropna()
                    elif handle_nan == "Replace with the precedent value":
                        data = data.fillna(method='pad')
                    elif handle_nan == "Replace NAN with zero":
                        data = data.fillna(0)
                    elif handle_nan == "Replace NAN with avrg":
                        data = data.fillna(data.mean())

                data = data.drop(data[data['FLP'] / data['WHP'] > 0.75].index)

                # Split the dataset into input features (X) and target variables (y)
                X = data[['ID','jour', 'month', 'year', 'CHOKE', 'WHP', 'FLP']]
                y = data[['quantiter gaz']]

                # Normalize the data using MinMaxScaler
                scaler_X = MinMaxScaler()
                scaler_y = MinMaxScaler()
                X = scaler_X.fit_transform(X)
                y = scaler_y.fit_transform(y)

                # Split the dataset into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

               # Create and train the RFR model
                model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)

                if st.button("Model training"):

                  model.fit(X_train, y_train)
                  # save RANDOM's model infos
                  global_var[5][0][0] = model
                  global_var[5][0][1] = scaler_X
                  global_var[5][0][2] = scaler_y

                  # Make predictions on the test set
                  y_pred = model.predict(X_test)
                  y_pred = y_pred.reshape(-1, 1)
                  y_pred2 = model.predict(X_train)
                  y_pred2 = y_pred2.reshape(-1, 1)

                  # Inverser la normalisation des prédictions et des vraies valeurs
                  y_pred = scaler_y.inverse_transform(y_pred)
                  y_test = scaler_y.inverse_transform(y_test)

                  # Calculate the mean squared error of the predictions
                  mse = mean_squared_error(y_test, y_pred)
                  mae = mean_absolute_error(y_test, y_pred)

                  st.write("Mean squared error:", mse)
                  st.write("Mean Absolute Error:", mae)
                # Calculer le Correlation coefficient R^2
                  R_squared_test = r2_score(y_test, y_pred)
                  R_squared_train = r2_score(y_train, y_pred2)

                  st.write("Correlation coefficient R^2 (test set) : ", R_squared_test)
                  st.write("Correlation coefficient R^2 (training set) : ", R_squared_train)

                elif st.button("Cancel", key="cancel_button_g"):
                  # Afficher un message d'annulation
                  st.warning("The operation was canceled !")

              elif model_selection == "Water Critical Model ":
                st.write("RFR - Water Critical Model ")
                data= df_jaugeage

                #Convertir la colonne 'START_DATE' en datetime avec le format d'origine
                data['START_DATE'] = pd.to_datetime(data['START_DATE'])
                data['jour'] = data['START_DATE'].dt.day
                data['month'] = data['START_DATE'].dt.month
                data['year'] = data['START_DATE'].dt.year

                  # Checkbox pour Bootstrapping
                doubler_lignes = st.sidebar.checkbox("Bootstrapping")

                if doubler_lignes:
                  # Bootstrapping en ajoutant le DataFrame à lui-même
                  # Double the dataset by duplicating the rows
                  data = pd.concat([data, data], ignore_index=True)
                #   data = data.append(data, ignore_index=True)



                # Gestion des valeurs manquantes
                handle_nan = st.sidebar.checkbox("Handling of missing values NAN", key="checkbox_nan8")
                if not handle_nan:
                    st.warning("Handling of missing values NAN")
                else:
                    handle_nan = st.sidebar.selectbox("Choose", options=["None", "Delete data point that contain NAN", "Replace with the precedent value", "Replace NAN with avrg"])
                    if handle_nan == "Delete data point that contain NAN":
                        data = data.dropna()
                    elif handle_nan == "Replace with the precedent value":
                        data = data.fillna(method='pad')
                    elif handle_nan == "Replace NAN with avrg":
                        data = data.fillna(data.mean())

                data = data.drop(data[data['FLP'] / data['WHP'] > 0.75].index)

                
                # Extraire les caractéristiques et la cible
                X = data[['ID','jour', 'month', 'year', 'CHOKE', 'WHP', 'FLP']]
                y = data[['water']]

                
                # Normalize the data using MinMaxScaler
                scaler_X = MinMaxScaler()
                scaler_y = MinMaxScaler()
                X = scaler_X.fit_transform(X)
                y = scaler_y.fit_transform(y)

                # Split the dataset into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size, random_state=42)

               # Create and train the RFR model
                model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)

                if st.button("Model training"):

                  model.fit(X_train, y_train)
                  # save RANDOM's model infos
                  global_var[10][0][0] = model
                  global_var[10][0][1] = scaler_X
                  global_var[10][0][2] = scaler_y

                  # Make predictions on the test set
                  y_pred = model.predict(X_test)
                  y_pred = y_pred.reshape(-1, 1)
                  y_pred2 = model.predict(X_train)
                  y_pred2 = y_pred2.reshape(-1, 1)

                  # Inverser la normalisation des prédictions et des vraies valeurs
                  y_pred = scaler_y.inverse_transform(y_pred)
                  y_test = scaler_y.inverse_transform(y_test)

                  # Calculate the mean squared error of the predictions
                  mse = mean_squared_error(y_test, y_pred)
                  mae = mean_absolute_error(y_test, y_pred)

                  st.write("Mean squared error:", mse)
                  st.write("Mean Absolute Error:", mae)

                # Calculer le Correlation coefficient R^2
                  R_squared_test = r2_score(y_test, y_pred)
                  R_squared_train = r2_score(y_train, y_pred2)

                  st.write("Correlation coefficient R^2 (test set) : ", R_squared_test)
                  st.write("Correlation coefficient R^2 (training set) : ", R_squared_train)
                
                elif st.button("Cancel", key="cancel_button_g"):
                  # Afficher un message d'annulation
                  st.warning("The operation was canceled !")
         
      if model_type == "NonCritical flow Model ":
              st.sidebar.title("Select Fluid Model")
              model_selection = st.sidebar.selectbox("Select Fluid Model", ("None","2Phase  Noncritical Model", "Oil Noncritical Model", "Gas Noncritical Model", "Water Noncritical Model"))
              if model_selection == "2Phase  Noncritical Model":
                st.write("RFR - 2Phase  Noncritical Model")
                data= df_jaugeage
                # data = data.drop(data[data['FLP'] / data['WHP'] > 0.75].index)
                #Convertir la colonne 'START_DATE' en datetime avec le format d'origine
                data['START_DATE'] = pd.to_datetime(data['START_DATE'])
                data['jour'] = data['START_DATE'].dt.day
                data['month'] = data['START_DATE'].dt.month
                data['year'] = data['START_DATE'].dt.year

                # Checkbox pour Delete repeted data
                supprimer_doublons = st.sidebar.checkbox("Delete repeted data")
                if supprimer_doublons:
                  # Identifier les lignes en double
                  doublons = data.duplicated()
                  # Supprimer les lignes en double
                  data = data[~doublons]
                
                # Gestion des valeurs manquantes
                handle_nan = st.sidebar.checkbox("Handling of missing values NAN", key="checkbox_nan9")
                if not handle_nan:
                    st.warning("Handling of missing values NAN")
                else:
                    handle_nan = st.sidebar.selectbox("Choose", options=["None", "Delete data point that contain NAN", "Replace with the precedent value", "Replace NAN with zero", "Replace NAN with avrg"])
                    if handle_nan == "Delete data point that contain NAN":
                        data = data.dropna()
                    elif handle_nan == "Replace with the precedent value":
                        data = data.fillna(method='pad')
                    elif handle_nan == "Replace NAN with zero":
                        data = data.fillna(0)
                    elif handle_nan == "Replace NAN with avrg":
                        data = data.fillna(data.mean())

                data = data.drop(data[data['FLP'] / data['WHP'] <=  0.75].index)
              
                # Split the dataset into input features (X) and target variables (y)
                X = data[['ID', 'jour', 'month', 'year', 'CHOKE', 'WHP', 'FLP']]
                y = data[['quantiter oil', 'quantiter gaz']]

                # Normalize the data using MinMaxScaler
                scaler_X = MinMaxScaler()
                scaler_y = MinMaxScaler()
                X = scaler_X.fit_transform(X)
                y = scaler_y.fit_transform(y)

                # Split the dataset into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

               # Create and train the RFR model
                model = RandomForestRegressor(n_estimators=100, random_state=42)

                if st.button("Model training"):

                  model.fit(X_train, y_train)
                  # save RANDOM's model infos
                  global_var[6][0][0] = model
                  global_var[6][0][1] = scaler_X
                  global_var[6][0][2] = scaler_y
                  # Make predictions on the test set
                  y_pred = model.predict(X_test)
                  y_pred2 = model.predict(X_train)

                  # Inverser la normalisation des prédictions et des vraies valeurs
                  y_pred = scaler_y.inverse_transform(y_pred)
                  y_test = scaler_y.inverse_transform(y_test)

                # Calculer les erreurs MSE et MAE pour chaque variable
                  mse_oil = mean_squared_error(y_test[:, 0], y_pred[:, 0])
                  mae_oil = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
                  mse_gaz = mean_squared_error(y_test[:, 1], y_pred[:, 1])
                  mae_gaz = mean_absolute_error(y_test[:, 1], y_pred[:, 1])

                # Afficher les résultats
                  st.write("MSE oil:", mse_oil)
                  st.write("MAE oil:", mae_oil)
                  st.write("MSE gaz:", mse_gaz)
                  st.write("MAE gaz:", mae_gaz)
                
                  # Calculer le Correlation coefficient R^2
                  R_squared_test = r2_score(y_test, y_pred)
                  R_squared_train = r2_score(y_train, y_pred2)

                  st.write("Correlation coefficient R^2 (test set) : ", R_squared_test)
                  st.write("Correlation coefficient R^2 (training set) : ", R_squared_train)

                elif st.button("Cancel", key="cancel_button_g"):
                  # Afficher un message d'annulation
                  st.warning("The operation was canceled !")
               
              elif model_selection == "Oil Noncritical Model":
                st.write("RFR - Oil Noncritical Model")
                data= df_jaugeage
                # data = data.drop(data[data['FLP'] / data['WHP'] > 0.75].index)
                #Convertir la colonne 'START_DATE' en datetime avec le format d'origine
                data['START_DATE'] = pd.to_datetime(data['START_DATE'])
                data['jour'] = data['START_DATE'].dt.day
                data['month'] = data['START_DATE'].dt.month
                data['year'] = data['START_DATE'].dt.year

                # Checkbox pour Delete repeted data
                supprimer_doublons = st.sidebar.checkbox("Delete repeted data")
                if supprimer_doublons:
                  # Identifier les lignes en double
                  doublons = data.duplicated()
                  # Supprimer les lignes en double
                  data = data[~doublons]
                
                  # Gestion des valeurs manquantes
                handle_nan = st.sidebar.checkbox("Handling of missing values NAN", key="checkbox_nanr")
                if not handle_nan:
                    st.warning("Handling of missing values NAN")
                else:
                    handle_nan = st.sidebar.selectbox("Choose", options=["None", "Delete data point that contain NAN", "Replace with the precedent value", "Replace NAN with zero", "Replace NAN with avrg"])
                    if handle_nan == "Delete data point that contain NAN":
                        data = data.dropna()
                    elif handle_nan == "Replace with the precedent value":
                        data = data.fillna(method='pad')
                    elif handle_nan == "Replace NAN with zero":
                        data = data.fillna(0)
                    elif handle_nan == "Replace NAN with avrg":
                        data = data.fillna(data.mean())
              
                data = data.drop(data[data['FLP'] / data['WHP'] <=  0.75].index)

                # Split the dataset into input features (X) and target variables (y)
                X = data[['ID','jour', 'month', 'year', 'CHOKE', 'WHP', 'FLP']]
                y = data[['quantiter oil']]

                # Normalize the data using MinMaxScaler
                scaler_X = MinMaxScaler()
                scaler_y = MinMaxScaler()
                X = scaler_X.fit_transform(X)
                y = scaler_y.fit_transform(y)

                # Split the dataset into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

               # Create and train the RFR model
                model = RandomForestRegressor(n_estimators=100, random_state=42)

                if st.button("Model training"):

                  model.fit(X_train, y_train)
                  # save RANDOM's model infos
                  global_var[7][0][0] = model
                  global_var[7][0][1] = scaler_X
                  global_var[7][0][2] = scaler_y
                  # Make predictions on the test set
                  y_pred = model.predict(X_test)
                  y_pred = y_pred.reshape(-1, 1)
                  y_pred2 = model.predict(X_train)
                  y_pred2 = y_pred2.reshape(-1, 1)

                  # Inverser la normalisation des prédictions et des vraies valeurs
                  y_pred = scaler_y.inverse_transform(y_pred)
                  y_test = scaler_y.inverse_transform(y_test)

                  # Calculate the mean squared error of the predictions
                  mse = mean_squared_error(y_test, y_pred)
                  mae = mean_absolute_error(y_test, y_pred)

                  st.write("Mean squared error:", mse)
                  st.write("Mean Absolute Error:", mae)

                # Calculer le Correlation coefficient R^2
                  R_squared_test = r2_score(y_test, y_pred)
                  R_squared_train = r2_score(y_train, y_pred2)

                  st.write("Correlation coefficient R^2 (test set) : ", R_squared_test)
                  st.write("Correlation coefficient R^2 (training set) : ", R_squared_train)

                elif st.button("Cancel", key="cancel_button_g"):
                  # Afficher un message d'annulation
                  st.warning("The operation was canceled !")
        
              elif model_selection == "Gas Noncritical Model":
                st.write("RFR - Gas Noncritical Model")
                data= df_jaugeage
                # data = data.drop(data[data['FLP'] / data['WHP'] > 0.75].index)
                #Convertir la colonne 'START_DATE' en datetime avec le format d'origine
                data['START_DATE'] = pd.to_datetime(data['START_DATE'])
                data['jour'] = data['START_DATE'].dt.day
                data['month'] = data['START_DATE'].dt.month
                data['year'] = data['START_DATE'].dt.year

                # Checkbox pour Delete repeted data
                supprimer_doublons = st.sidebar.checkbox("Delete repeted data")
                if supprimer_doublons:
                  # Identifier les lignes en double
                  doublons = data.duplicated()
                  # Supprimer les lignes en double
                  data = data[~doublons]
                
                # Gestion des valeurs manquantes
                handle_nan = st.sidebar.checkbox("Handling of missing values NAN", key="checkbox_nanf")
                if not handle_nan:
                    st.warning("Handling of missing values NAN")
                else:
                    handle_nan = st.sidebar.selectbox("Choose", options=["None", "Delete data point that contain NAN", "Replace with the precedent value", "Replace NAN with zero", "Replace NAN with avrg"])
                    if handle_nan == "Delete data point that contain NAN":
                        data = data.dropna()
                    elif handle_nan == "Replace with the precedent value":
                        data = data.fillna(method='pad')
                    elif handle_nan == "Replace NAN with zero":
                        data = data.fillna(0)
                    elif handle_nan == "Replace NAN with avrg":
                        data = data.fillna(data.mean())


                data = data.drop(data[data['FLP'] / data['WHP'] <=  0.75].index)

                # Split the dataset into input features (X) and target variables (y)
                X = data[['ID','jour', 'month', 'year', 'CHOKE', 'WHP', 'FLP']]
                y = data[['quantiter gaz']]

                # Normalize the data using MinMaxScaler
                scaler_X = MinMaxScaler()
                scaler_y = MinMaxScaler()
                X = scaler_X.fit_transform(X)
                y = scaler_y.fit_transform(y)

                # Split the dataset into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

               # Create and train the RFR model
                model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
                                
                if st.button("Model training"):

                  model.fit(X_train, y_train)
                  # save RANDOM's model infos
                  global_var[8][0][0] = model
                  global_var[8][0][1] = scaler_X
                  global_var[8][0][2] = scaler_y
                  # Make predictions on the test set
                  y_pred = model.predict(X_test)
                  y_pred = y_pred.reshape(-1, 1)
                  y_pred2 = model.predict(X_train)
                  y_pred2 = y_pred2.reshape(-1, 1)

                  # Inverser la normalisation des prédictions et des vraies valeurs
                  y_pred = scaler_y.inverse_transform(y_pred)
                  y_test = scaler_y.inverse_transform(y_test)

                  # Calculate the mean squared error of the predictions
                  mse = mean_squared_error(y_test, y_pred)
                  mae = mean_absolute_error(y_test, y_pred)

                  st.write("Mean squared error:", mse)
                  st.write("Mean Absolute Error:", mae)
                # Calculer le Correlation coefficient R^2
                  R_squared_test = r2_score(y_test, y_pred)
                  R_squared_train = r2_score(y_train, y_pred2)

                  st.write("Correlation coefficient R^2 (test set) : ", R_squared_test)
                  st.write("Correlation coefficient R^2 (training set) : ", R_squared_train)

                elif st.button("Cancel", key="cancel_button_g"):
                  # Afficher un message d'annulation
                  st.warning("The operation was canceled !")

              elif model_selection == "Water Noncritical Model":
                st.write("RFR  - Water Noncritical Model")
                data= df_jaugeage

                #Convertir la colonne 'START_DATE' en datetime avec le format d'origine
                data['START_DATE'] = pd.to_datetime(data['START_DATE'])
                data['jour'] = data['START_DATE'].dt.day
                data['month'] = data['START_DATE'].dt.month
                data['year'] = data['START_DATE'].dt.year

                 # Checkbox pour Bootstrapping
                doubler_lignes = st.sidebar.checkbox("Bootstrapping")

                if doubler_lignes:
                  # Bootstrapping en ajoutant le DataFrame à lui-même
                  # Double the dataset by duplicating the rows
                  data = pd.concat([data, data], ignore_index=True)
                #   data = data.append(data, ignore_index=True)

                
                   # Gestion des valeurs manquantes
                  handle_nan = st.sidebar.checkbox("Handling of missing values NAN", key="checkbox_nanb")
                  if not handle_nan:
                      st.warning("Handling of missing values NAN")
                  else:
                      handle_nan = st.sidebar.selectbox("Choose", options=["None", "Delete data point that contain NAN", "Replace with the precedent value", "Replace NAN with avrg"])
                      if handle_nan == "Delete data point that contain NAN":
                          data = data.dropna()
                      elif handle_nan == "Replace with the precedent value":
                          data = data.fillna(method='pad')
                      elif handle_nan == "Replace NAN with avrg":
                          data = data.fillna(data.mean())

                data = data.drop(data[data['FLP'] / data['WHP'] <=  0.75].index)
              
                # Extraire les caractéristiques et la cible
                X = data[['ID','jour', 'month', 'year', 'CHOKE', 'WHP', 'FLP']]
                y = data[['water']]

                
                # Normalize the data using MinMaxScaler
                scaler_X = MinMaxScaler()
                scaler_y = MinMaxScaler()
                X = scaler_X.fit_transform(X)
                y = scaler_y.fit_transform(y)

                # Split the dataset into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size, random_state=42)

               # Create and train the RFR model
                model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)

                if st.button("Model training"):

                  model.fit(X_train, y_train)
                  # save RANDOM's model infos
                  global_var[11][0][0] = model
                  global_var[11][0][1] = scaler_X
                  global_var[11][0][2] = scaler_y

                  # Make predictions on the test set
                  y_pred = model.predict(X_test)
                  y_pred = y_pred.reshape(-1, 1)
                  y_pred2 = model.predict(X_train)
                  y_pred2 = y_pred2.reshape(-1, 1)

                  # Inverser la normalisation des prédictions et des vraies valeurs
                  y_pred = scaler_y.inverse_transform(y_pred)
                  y_test = scaler_y.inverse_transform(y_test)

                  # Calculate the mean squared error of the predictions
                  mse = mean_squared_error(y_test, y_pred)
                  mae = mean_absolute_error(y_test, y_pred)

                  st.write("Mean squared error:", mse)
                  st.write("Mean Absolute Error:", mae)

                # Calculer le Correlation coefficient R^2
                  R_squared_test = r2_score(y_test, y_pred)
                  R_squared_train = r2_score(y_train, y_pred2)

                  st.write("Correlation coefficient R^2 (test set) : ", R_squared_test)
                  st.write("Correlation coefficient R^2 (training set) : ", R_squared_train)
                
                elif st.button("Cancel", key="cancel_button_g"):
                  # Afficher un message d'annulation
                  st.warning("The operation was canceled !")
                                            
    elif algorithm == "RNN":
        # Ajoutez le code pour l'algorithme RNN ici
        st.subheader("AI RNN")
        st.write(" Type : ", model_type)
        if model_type == "All regim of flow Model ":
            st.sidebar.title("Select Fluid Model")
            model_selection = st.sidebar.selectbox("Select Fluid Model", ("None","2Phase  Model", "Oil Model", "GAS Model", "Water Model"))
            if model_selection == "2Phase  Model":
                st.write("RNN - 2Phase  Model")
                data= df_jaugeage
                # data = data.drop(data[data['FLP'] / data['WHP'] > 0.75].index)

                #Convertir la colonne 'START_DATE' en datetime avec le format d'origine
                data['START_DATE'] = pd.to_datetime(data['START_DATE'])
                data['jour'] = data['START_DATE'].dt.day
                data['month'] = data['START_DATE'].dt.month
                data['year'] = data['START_DATE'].dt.year

                # Checkbox pour Delete repeted data
                supprimer_doublons = st.sidebar.checkbox("Delete repeted data")
                if supprimer_doublons:
                  # Identifier les lignes en double
                  doublons = data.duplicated()
                  # Supprimer les lignes en double
                  data = data[~doublons]
                
                # Gestion des valeurs manquantes
                handle_nan = st.sidebar.checkbox("Handling of missing values NAN", key="checkbox_nanm")
                if not handle_nan:
                    st.warning("Handling of missing values NAN")
                else:
                    handle_nan = st.sidebar.selectbox("Choose", options=["None", "Delete data point that contain NAN", "Replace with the precedent value", "Replace NAN with zero", "Replace NAN with avrg"])
                    if handle_nan == "Delete data point that contain NAN":
                        data = data.dropna()
                    elif handle_nan == "Replace with the precedent value":
                        data = data.fillna(method='pad')
                    elif handle_nan == "Replace NAN with zero":
                        data = data.fillna(0)
                    elif handle_nan == "Replace NAN with avrg":
                        data = data.fillna(data.mean())
              
                # Split the dataset into input features (X) and target variables (y)
                X = data[['ID', 'jour', 'month', 'year', 'CHOKE', 'WHP', 'FLP']]
                y = data[['quantiter oil', 'quantiter gaz']]

                # Normaliser les données en utilisant MinMaxScaler
                scaler_XR = MinMaxScaler()
                scaler_yR= MinMaxScaler()
                X = scaler_XR.fit_transform(X)
                y = scaler_yR.fit_transform(y)

                # Séparer le jeu de données en ensembles d'entraînement et de test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size, random_state=42)

               # Reshape des données pour les rendre compatibles avec RNN
                X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
                X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

               # Créer et Model training RNN
                model = Sequential()
                model.add(SimpleRNN(num_input_neurons, input_shape=(1, 7), activation= input_activation))
                for i in range(num_hidden_layers):
                  model.add(Dense(num_neurons, activation= hidden_activation))
                # model.add(Dense(32, activation='relu'))
                model.add(Dense(num_output_neurons, activation= output_activation))
                model.compile(loss='mean_squared_error', optimizer='adam')


                if st.button("Model training"):
                     # Entraînement du modèle
                  history = None
                  if model is not None:
                          if use_epochs and use_batch_size:
                              history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_test, y_test))
                          elif use_epochs:
                              history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test))
                          elif use_batch_size:
                              history = model.fit(X_train, y_train, batch_size=batch_size, validation_data=(X_test, y_test))
                          else:
                              history = model.fit(X_train, y_train, validation_data=(X_test, y_test))

                  # save RNN's infos
                  global_var[0][2][0] = model
                  global_var[0][2][1] = scaler_XR
                  global_var[0][2][2] = scaler_yR
                  # Faire des prédictions sur l'ensemble de test
                  y_pred_test = model.predict(X_test)
                  y_pred_train = model.predict(X_train)


                  # Inverser la normalisation des prédictions et des vraies valeurs
                  y_pred_test = scaler_yR.inverse_transform(y_pred_test)
                  y_test = scaler_yR.inverse_transform(y_test)


                  # Calculer les erreurs MSE et MAE pour chaque variable
                  mse_oil = mean_squared_error(y_test[:, 0], y_pred_test[:, 0])
                  mae_oil = mean_absolute_error(y_test[:, 0], y_pred_test[:, 0])
                  mse_gaz = mean_squared_error(y_test[:, 1], y_pred_test[:, 1])
                  mae_gaz = mean_absolute_error(y_test[:, 1], y_pred_test[:, 1])

                  # Afficher les résultats
                  st.write("MSE oil:", mse_oil)
                  st.write("MAE oil:", mae_oil)
                  st.write("MSE gaz:", mse_gaz)
                  st.write("MAE gaz:", mae_gaz)
                  # Calculer le Correlation coefficient R^2
                  R_squared_test = r2_score(y_test, y_pred_test)
                  R_squared_train = r2_score(y_train, y_pred_train)

                  st.write("Correlation coefficient R^2 (test set) : ", R_squared_test)
                  st.write("Correlation coefficient R^2 (training set) : ", R_squared_train)

                elif st.button("Cancel", key="cancel_button_g"):
                  # Afficher un message d'annulation
                  st.warning("The operation was canceled !")
             
            elif model_selection == "Oil Model":
                st.write("RNN - Oil Model")
                # Charger les données
                data= df_jaugeage
                # data = data.drop(data[data['FLP'] / data['WHP'] > 0.75].index)
                #Convertir la colonne 'START_DATE' en datetime avec le format d'origine
                data['START_DATE'] = pd.to_datetime(data['START_DATE'])
                data['jour'] = data['START_DATE'].dt.day
                data['month'] = data['START_DATE'].dt.month
                data['year'] = data['START_DATE'].dt.year

                # Checkbox pour Delete repeted data
                supprimer_doublons = st.sidebar.checkbox("Delete repeted data")
                if supprimer_doublons:
                  # Identifier les lignes en double
                  doublons = data.duplicated()
                  # Supprimer les lignes en double
                  data = data[~doublons]
                
                 # Gestion des valeurs manquantes
                handle_nan = st.sidebar.checkbox("Handling of missing values NAN", key="checkbox_nanv")
                if not handle_nan:
                    st.warning("Handling of missing values NAN")
                else:
                    handle_nan = st.sidebar.selectbox("Choose", options=["None", "Delete data point that contain NAN", "Replace with the precedent value", "Replace NAN with zero", "Replace NAN with avrg"])
                    if handle_nan == "Delete data point that contain NAN":
                        data = data.dropna()
                    elif handle_nan == "Replace with the precedent value":
                        data = data.fillna(method='pad')
                    elif handle_nan == "Replace NAN with zero":
                        data = data.fillna(0)
                    elif handle_nan == "Replace NAN with avrg":
                        data = data.fillna(data.mean())
                

                # Split the dataset into input features (X) and target variables (y)
                X = data[['ID','jour', 'month', 'year', 'CHOKE', 'WHP', 'FLP']]
                y = data[['quantiter oil']]

                # Normaliser les données en utilisant MinMaxScaler
                scaler_XR = MinMaxScaler()
                scaler_yR= MinMaxScaler()
                X = scaler_XR.fit_transform(X)
                y = scaler_yR.fit_transform(y)

                 # Séparer le jeu de données en ensembles d'entraînement et de test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size, random_state=42)

               # Reshape des données pour les rendre compatibles avec RNN
                X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
                X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

               # Créer et Model training RNN
                model = Sequential()
                model.add(SimpleRNN(num_input_neurons, input_shape=(1, 7), activation= input_activation))
                for i in range(num_hidden_layers):
                  model.add(Dense(num_neurons, activation= hidden_activation))
                model.add(Dense(num_output_neurons, activation= output_activation))
                model.compile(loss='mean_squared_error', optimizer='adam')

                if st.button("Model training"):
                  # Entraînement du modèle
                  history = None
                  if model is not None:
                          if use_epochs and use_batch_size:
                              history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_test, y_test))
                          elif use_epochs:
                              history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test))
                          elif use_batch_size:
                              history = model.fit(X_train, y_train, batch_size=batch_size, validation_data=(X_test, y_test))
                          else:
                              history = model.fit(X_train, y_train, validation_data=(X_test, y_test))

                  # save RNN's infos
                  global_var[1][2][0] = model
                  global_var[1][2][1] = scaler_XR
                  global_var[1][2][2] = scaler_yR
                # Faire des prédictions sur l'ensemble de test
                  y_pred_test = model.predict(X_test)
                  y_pred_train = model.predict(X_train)

                  # Inverser la normalisation des prédictions et des vraies valeurs
                  y_pred_test = scaler_yR.inverse_transform(y_pred_test)
                  y_test = scaler_yR.inverse_transform(y_test)

                # Calculer l'erreur quadratique moyenne des prédictions
                  mse_test = mean_squared_error(y_test, y_pred_test)

                  st.write("Mean squared error (test set):", mse_test)

                # Calculer l'erreur absolue moyenne des prédictions
                  mae_test = mean_absolute_error(y_test, y_pred_test)

                  st.write("Mean Absolute Error (test set):", mae_test)
                  # Calculer le Correlation coefficient R^2
                  R_squared_test = r2_score(y_test, y_pred_test)
                  R_squared_train = r2_score(y_train, y_pred_train)

                  st.write("Correlation coefficient R^2 (test set) : ", R_squared_test)
                  st.write("Correlation coefficient R^2 (training set) : ", R_squared_train)

                elif st.button("Cancel", key="cancel_button_g"):
                  # Afficher un message d'annulation
                  st.warning("The operation was canceled !")

            elif model_selection == "GAS Model":
                st.write("RNN - GAS Model")
                data= df_jaugeage
                # data = data.drop(data[data['FLP'] / data['WHP'] > 0.75].index)
                #Convertir la colonne 'START_DATE' en datetime avec le format d'origine
                data['START_DATE'] = pd.to_datetime(data['START_DATE'])
                data['jour'] = data['START_DATE'].dt.day
                data['month'] = data['START_DATE'].dt.month
                data['year'] = data['START_DATE'].dt.year
                # Checkbox pour Delete repeted data
                supprimer_doublons = st.sidebar.checkbox("Delete repeted data")
                if supprimer_doublons:
                  # Identifier les lignes en double
                  doublons = data.duplicated()
                  # Supprimer les lignes en double
                  data = data[~doublons]
                
                 # Gestion des valeurs manquantes
                handle_nan = st.sidebar.checkbox("Handling of missing values NAN", key="checkbox_nanx")
                if not handle_nan:
                    st.warning("Handling of missing values NAN")
                else:
                    handle_nan = st.sidebar.selectbox("Choose", options=["None", "Delete data point that contain NAN", "Replace with the precedent value", "Replace NAN with zero", "Replace NAN with avrg"])
                    if handle_nan == "Delete data point that contain NAN":
                        data = data.dropna()
                    elif handle_nan == "Replace with the precedent value":
                        data = data.fillna(method='pad')
                    elif handle_nan == "Replace NAN with zero":
                        data = data.fillna(0)
                    elif handle_nan == "Replace NAN with avrg":
                        data = data.fillna(data.mean())

                # Split the dataset into input features (X) and target variables (y)
                X = data[['ID','jour', 'month', 'year', 'CHOKE', 'WHP', 'FLP']]
                y = data[['quantiter gaz']]

                # Normaliser les données en utilisant MinMaxScaler
                scaler_XR = MinMaxScaler()
                scaler_yR= MinMaxScaler()
                X = scaler_XR.fit_transform(X)
                y = scaler_yR.fit_transform(y)

                 # Séparer le jeu de données en ensembles d'entraînement et de test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size, random_state=42)

               # Reshape des données pour les rendre compatibles avec RNN
                X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
                X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

               # Créer et Model training RNN
                model = Sequential()
                model.add(SimpleRNN(num_input_neurons, input_shape=(1, 7), activation= input_activation))
                for i in range(num_hidden_layers):
                  model.add(Dense(num_neurons, activation= hidden_activation))
                # model.add(Dense(32, activation='relu'))
                model.add(Dense(num_output_neurons, activation= output_activation))
                model.compile(loss='mean_squared_error', optimizer='adam')

                if st.button("Model training"):
                     # Entraînement du modèle
                  history = None
                  if model is not None:
                          if use_epochs and use_batch_size:
                              history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_test, y_test))
                          elif use_epochs:
                              history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test))
                          elif use_batch_size:
                              history = model.fit(X_train, y_train, batch_size=batch_size, validation_data=(X_test, y_test))
                          else:
                              history = model.fit(X_train, y_train, validation_data=(X_test, y_test))

                  # save RNN's infos
                  global_var[2][2][0] = model
                  global_var[2][2][1] = scaler_XR
                  global_var[2][2][2] = scaler_yR
                  # Faire des prédictions sur l'ensemble de test
                  y_pred_test = model.predict(X_test)
                  y_pred_train = model.predict(X_train)
                  # Inverser la normalisation des prédictions et des vraies valeurs
                  y_pred_test = scaler_yR.inverse_transform(y_pred_test)
                  y_test = scaler_yR.inverse_transform(y_test)

                  # Calculer l'erreur quadratique moyenne des prédictions
                  mse_test = mean_squared_error(y_test, y_pred_test)

                  st.write("Mean squared error (test set):", mse_test)

                # Calculer l'erreur absolue moyenne des prédictions
                  mae_test = mean_absolute_error(y_test, y_pred_test)

                  st.write("Mean Absolute Error (test set):", mae_test)

                

                  # # Afficher les valeurs de y_test et y_pred dans une boucle
                  # for i in range(len(y_test)):
                  #   y_test_inverse = scaler_yR.inverse_transform([y_test[i]])
                  #   y_pred_inverse = scaler_yR.inverse_transform([y_pred_test[i]])
                  #   st.write("y_test:", y_test_inverse)
                  #   st.write("y_pred:", y_pred_inverse)
                  #   st.write()

                  # Calculer le Correlation coefficient R^2
                  R_squared_test = r2_score(y_test, y_pred_test)
                  R_squared_train = r2_score(y_train, y_pred_train)

                  st.write("Correlation coefficient R^2 (test set) : ", R_squared_test)
                  st.write("Correlation coefficient R^2 (training set) : ", R_squared_train)

                elif st.button("Cancel", key="cancel_button_g"):
                  # Afficher un message d'annulation
                  st.warning("The operation was canceled !")

            elif model_selection == "Water Model":
                st.write("RNN - Water Model")
                data= df_jaugeage

                #Convertir la colonne 'START_DATE' en datetime avec le format d'origine
                data['START_DATE'] = pd.to_datetime(data['START_DATE'])
                data['jour'] = data['START_DATE'].dt.day
                data['month'] = data['START_DATE'].dt.month
                data['year'] = data['START_DATE'].dt.year

                # Checkbox pour Bootstrapping
                doubler_lignes = st.sidebar.checkbox("Bootstrapping")

                if doubler_lignes:
                  # Double the dataset by duplicating the rows
                  data = pd.concat([data, data], ignore_index=True)
                  #   data = data.append(data, ignore_index=True)
                
                 # Gestion des valeurs manquantes
                handle_nan = st.sidebar.checkbox("Handling of missing values NAN", key="checkbox_nanz")
                if not handle_nan:
                    st.warning("Handling of missing values NAN")
                else:
                    handle_nan = st.sidebar.selectbox("Choose", options=["None", "Delete data point that contain NAN", "Replace with the precedent value", "Replace NAN with avrg"])
                    if handle_nan == "Delete data point that contain NAN":
                        data = data.dropna()
                    elif handle_nan == "Replace with the precedent value":
                        data = data.fillna(method='pad')
                    elif handle_nan == "Replace NAN with avrg":
                        data = data.fillna(data.mean())
            
                # Extraire les caractéristiques et la cible
                X = data[[ 'ID','jour', 'month', 'year', 'CHOKE', 'WHP', 'FLP']]
                y = data[[ 'water']]

                # Normaliser les données en utilisant MinMaxScaler
                scaler_XR = MinMaxScaler()
                scaler_yR= MinMaxScaler()
                X = scaler_XR.fit_transform(X)
                y = scaler_yR.fit_transform(y)

                 # Séparer le jeu de données en ensembles d'entraînement et de test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size, random_state=42)

               # Reshape des données pour les rendre compatibles avec RNN
                X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
                X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

               # Créer et Model training RNN
                model = Sequential()
                model.add(SimpleRNN(num_input_neurons, input_shape=(1, 7), activation= input_activation))
                for i in range(num_hidden_layers):
                  model.add(Dense(num_neurons, activation= hidden_activation))
                # model.add(Dense(32, activation='relu'))
                model.add(Dense(num_output_neurons, activation= output_activation))
                model.compile(loss='mean_squared_error', optimizer='adam')

                if st.button("Model training"):
                     # Entraînement du modèle
                  history = None
                  if model is not None:
                          if use_epochs and use_batch_size:
                              history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_test, y_test))
                          elif use_epochs:
                              history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test))
                          elif use_batch_size:
                              history = model.fit(X_train, y_train, batch_size=batch_size, validation_data=(X_test, y_test))
                          else:
                              history = model.fit(X_train, y_train, validation_data=(X_test, y_test))

                  # save RNN's infos
                  global_var[9][2][0] = model
                  global_var[9][2][1] = scaler_XR
                  global_var[9][2][2] = scaler_yR
                  # Faire des prédictions sur l'ensemble de test
                  y_pred_test = model.predict(X_test)
                  y_pred_train = model.predict(X_train)
                  # Inverser la normalisation des prédictions et des vraies valeurs
                  y_pred_test = scaler_yR.inverse_transform(y_pred_test)
                  y_test = scaler_yR.inverse_transform(y_test)

                  # Calculer l'erreur quadratique moyenne des prédictions
                  mse_test = mean_squared_error(y_test, y_pred_test)

                  st.write("Mean squared error (test set):", mse_test)

                # Calculer l'erreur absolue moyenne des prédictions
                  mae_test = mean_absolute_error(y_test, y_pred_test)

                  st.write("Mean Absolute Error (test set):", mae_test)

                

                  # # Afficher les valeurs de y_test et y_pred dans une boucle
                  # for i in range(len(y_test)):
                  #   y_test_inverse = scaler_yR.inverse_transform([y_test[i]])
                  #   y_pred_inverse = scaler_yR.inverse_transform([y_pred_test[i]])
                  #   st.write("y_test:", y_test_inverse)
                  #   st.write("y_pred:", y_pred_inverse)
                  #   st.write()

                  # Calculer le Correlation coefficient R^2
                  R_squared_test = r2_score(y_test, y_pred_test)
                  R_squared_train = r2_score(y_train, y_pred_train)

                  st.write("Correlation coefficient R^2 (test set) : ", R_squared_test)
                  st.write("Correlation coefficient R^2 (training set) : ", R_squared_train)

                elif st.button("Cancel", key="cancel_button_g"):
                  # Afficher un message d'annulation
                  st.warning("The operation was canceled !")
                    
        if model_type == "Critical flow Model":
              st.sidebar.title("Select Fluid Model")
              model_selection = st.sidebar.selectbox("Select Fluid Model", ("None","2Phase  Critical Model  ", "Oil Critical Model  ", "Gas Critical Model  ", "Water Critical Model "))
              if model_selection == "2Phase  Critical Model  ":
                st.write("RNN- 2Phase  Critical Model  ")
                data= df_jaugeage
                #Convertir la colonne 'START_DATE' en datetime avec le format d'origine
                data['START_DATE'] = pd.to_datetime(data['START_DATE'])
                data['jour'] = data['START_DATE'].dt.day
                data['month'] = data['START_DATE'].dt.month
                data['year'] = data['START_DATE'].dt.year

                # Checkbox pour Delete repeted data
                supprimer_doublons = st.sidebar.checkbox("Delete repeted data")
                if supprimer_doublons:
                  # Identifier les lignes en double
                  doublons = data.duplicated()
                  # Supprimer les lignes en double
                  data = data[~doublons]
                
                 # Gestion des valeurs manquantes
                handle_nan = st.sidebar.checkbox("Handling of missing values NAN", key="checkbox_nany")
                if not handle_nan:
                    st.warning("Handling of missing values NAN")
                else:
                    handle_nan = st.sidebar.selectbox("Choose", options=["None", "Delete data point that contain NAN", "Replace with the precedent value", "Replace NAN with zero", "Replace NAN with avrg"])
                    if handle_nan == "Delete data point that contain NAN":
                        data = data.dropna()
                    elif handle_nan == "Replace with the precedent value":
                        data = data.fillna(method='pad')
                    elif handle_nan == "Replace NAN with zero":
                        data = data.fillna(0)
                    elif handle_nan == "Replace NAN with avrg":
                        data = data.fillna(data.mean())

                data = data.drop(data[data['FLP'] / data['WHP'] <=  0.75].index)
                

                # Split the dataset into input features (X) and target variables (y)
                X = data[['ID', 'jour', 'month', 'year', 'CHOKE', 'WHP', 'FLP']]
                y = data[['quantiter oil','quantiter gaz']]

                # Normaliser les données en utilisant MinMaxScaler
                scaler_XR = MinMaxScaler()
                scaler_yR= MinMaxScaler()
                X = scaler_XR.fit_transform(X)
                y = scaler_yR.fit_transform(y)

                 # Séparer le jeu de données en ensembles d'entraînement et de test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size, random_state=42)

               # Reshape des données pour les rendre compatibles avec RNN
                X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
                X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

               # Créer et Model training RNN
                model = Sequential()
                model.add(SimpleRNN(num_input_neurons, input_shape=(1, 7), activation= input_activation))
                for i in range(num_hidden_layers):
                  model.add(Dense(num_neurons, activation= hidden_activation))
                # model.add(Dense(32, activation='relu'))
                model.add(Dense(num_output_neurons, activation= output_activation))
                model.compile(loss='mean_squared_error', optimizer='adam')

                if st.button("Model training"):

                  history = None
                  if model is not None:
                          if use_epochs and use_batch_size:
                              history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_test, y_test))
                          elif use_epochs:
                              history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test))
                          elif use_batch_size:
                              history = model.fit(X_train, y_train, batch_size=batch_size, validation_data=(X_test, y_test))
                          else:
                              history = model.fit(X_train, y_train, validation_data=(X_test, y_test))
                  # save RNN's infos
                  global_var[3][2][0] = model
                  global_var[3][2][1] = scaler_XR
                  global_var[3][2][2] = scaler_yR
                # Faire des prédictions sur l'ensemble de test
                  y_pred_test = model.predict(X_test)
                  y_pred_train = model.predict(X_train)

                  # Inverser la normalisation des prédictions et des vraies valeurs
                  y_pred_test = scaler_yR.inverse_transform(y_pred_test)
                  y_test = scaler_yR.inverse_transform(y_test)


                # Calculer les erreurs MSE et MAE pour chaque variable
                  mse_oil = mean_squared_error(y_test[:, 0], y_pred_test[:, 0])
                  mae_oil = mean_absolute_error(y_test[:, 0], y_pred_test[:, 0])
                  mse_gaz = mean_squared_error(y_test[:, 1], y_pred_test[:, 1])
                  mae_gaz = mean_absolute_error(y_test[:, 1], y_pred_test[:, 1])

                # Afficher les résultats
                  st.write("MSE oil:", mse_oil)
                  st.write("MAE oil:", mae_oil)
                  st.write("MSE gaz:", mse_gaz)
                  st.write("MAE gaz:", mae_gaz)

                  # Calculer le Correlation coefficient R^2
                  R_squared_test = r2_score(y_test, y_pred_test)
                  R_squared_train = r2_score(y_train, y_pred_train)

                  st.write("Correlation coefficient R^2 (test set) : ", R_squared_test)
                  st.write("Correlation coefficient R^2 (training set) : ", R_squared_train)

                elif st.button("Cancel", key="cancel_button_g"):
                  # Afficher un message d'annulation
                  st.warning("The operation was canceled !")

              elif model_selection == "Oil Critical Model  ":
                st.write("RNN - Oil Critical Model  ")
                data= df_jaugeage

                #Convertir la colonne 'START_DATE' en datetime avec le format d'origine
                data['START_DATE'] = pd.to_datetime(data['START_DATE'])
                data['jour'] = data['START_DATE'].dt.day
                data['month'] = data['START_DATE'].dt.month
                data['year'] = data['START_DATE'].dt.year

                # Checkbox pour Delete repeted data
                supprimer_doublons = st.sidebar.checkbox("Delete repeted data")
                if supprimer_doublons:
                  # Identifier les lignes en double
                  doublons = data.duplicated()
                  # Supprimer les lignes en double
                  data = data[~doublons]
                
                 # Gestion des valeurs manquantes
                handle_nan = st.sidebar.checkbox("Handling of missing values NAN", key="checkbox_nanl")
                if not handle_nan:
                    st.warning("Handling of missing values NAN")
                else:
                    handle_nan = st.sidebar.selectbox("Choose", options=["None", "Delete data point that contain NAN", "Replace with the precedent value", "Replace NAN with zero", "Replace NAN with avrg"])
                    if handle_nan == "Delete data point that contain NAN":
                        data = data.dropna()
                    elif handle_nan == "Replace with the precedent value":
                        data = data.fillna(method='pad')
                    elif handle_nan == "Replace NAN with zero":
                        data = data.fillna(0)
                    elif handle_nan == "Replace NAN with avrg":
                        data = data.fillna(data.mean())

                data = data.drop(data[data['FLP'] / data['WHP'] <=  0.75].index)

                # Split the dataset into input features (X) and target variables (y)
                X = data[['ID','jour', 'month', 'year', 'CHOKE', 'WHP', 'FLP']]
                y = data[['quantiter oil']]

                # Normaliser les données en utilisant MinMaxScaler
                scaler_XR = MinMaxScaler()
                scaler_yR= MinMaxScaler()
                X = scaler_XR.fit_transform(X)
                y = scaler_yR.fit_transform(y)

                 # Séparer le jeu de données en ensembles d'entraînement et de test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size, random_state=42)

               # Reshape des données pour les rendre compatibles avec RNN
                X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
                X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

               # Créer et Model training RNN
                model = Sequential()
                model.add(SimpleRNN(num_input_neurons, input_shape=(1, 7), activation= input_activation))
                for i in range(num_hidden_layers):
                  model.add(Dense(num_neurons, activation= hidden_activation))
                # model.add(Dense(32, activation='relu'))
                model.add(Dense(num_output_neurons, activation= output_activation))
                model.compile(loss='mean_squared_error', optimizer='adam')

                if st.button("Model training"):

                  history = None
                  if model is not None:
                          if use_epochs and use_batch_size:
                              history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_test, y_test))
                          elif use_epochs:
                              history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test))
                          elif use_batch_size:
                              history = model.fit(X_train, y_train, batch_size=batch_size, validation_data=(X_test, y_test))
                          else:
                              history = model.fit(X_train, y_train, validation_data=(X_test, y_test))
                  # save RNN's infos
                  global_var[4][2][0] = model
                  global_var[4][2][1] = scaler_XR
                  global_var[4][2][2] = scaler_yR
                  # Faire des prédictions sur l'ensemble de test
                  y_pred_test = model.predict(X_test)
                  y_pred_train = model.predict(X_train)

                  # Inverser la normalisation des prédictions et des vraies valeurs
                  y_pred_test = scaler_yR.inverse_transform(y_pred_test)
                  y_test = scaler_yR.inverse_transform(y_test)

                # Calculer l'erreur quadratique moyenne des prédictions
                  mse_test = mean_squared_error(y_test, y_pred_test)

                  st.write("Mean squared error (test set):", mse_test)

                # Calculer l'erreur absolue moyenne des prédictions
                  mae_test = mean_absolute_error(y_test, y_pred_test)

                  st.write("Mean Absolute Error (test set):", mae_test)

                  # Calculer le Correlation coefficient R^2
                  R_squared_test = r2_score(y_test, y_pred_test)
                  R_squared_train = r2_score(y_train, y_pred_train)

                  st.write("Correlation coefficient R^2 (test set) : ", R_squared_test)
                  st.write("Correlation coefficient R^2 (training set) : ", R_squared_train)

                elif st.button("Cancel", key="cancel_button_g"):
                  # Afficher un message d'annulation
                  st.warning("The operation was canceled !")

              elif model_selection == "Gas Critical Model  ":
                st.write("RNN - Gas Critical Model  ")
                data= df_jaugeage

                #Convertir la colonne 'START_DATE' en datetime avec le format d'origine
                data['START_DATE'] = pd.to_datetime(data['START_DATE'])
                data['jour'] = data['START_DATE'].dt.day
                data['month'] = data['START_DATE'].dt.month
                data['year'] = data['START_DATE'].dt.year

                # Checkbox pour Delete repeted data
                supprimer_doublons = st.sidebar.checkbox("Delete repeted data")
                if supprimer_doublons:
                  # Identifier les lignes en double
                  doublons = data.duplicated()
                  # Supprimer les lignes en double
                  data = data[~doublons]
                
                 # Gestion des valeurs manquantes
                handle_nan = st.sidebar.checkbox("Handling of missing values NAN", key="checkbox_nane")
                if not handle_nan:
                    st.warning("Handling of missing values NAN")
                else:
                    handle_nan = st.sidebar.selectbox("Choose", options=["None", "Delete data point that contain NAN", "Replace with the precedent value", "Replace NAN with zero", "Replace NAN with avrg"])
                    if handle_nan == "Delete data point that contain NAN":
                        data = data.dropna()
                    elif handle_nan == "Replace with the precedent value":
                        data = data.fillna(method='pad')
                    elif handle_nan == "Replace NAN with zero":
                        data = data.fillna(0)
                    elif handle_nan == "Replace NAN with avrg":
                        data = data.fillna(data.mean())
                
                data = data.drop(data[data['FLP'] / data['WHP'] <=  0.75].index)

                # Split the dataset into input features (X) and target variables (y)
                X = data[['ID', 'jour', 'month', 'year', 'CHOKE', 'WHP', 'FLP']]
                y = data[['quantiter gaz']]

                # Normaliser les données en utilisant MinMaxScaler
                scaler_XR = MinMaxScaler()
                scaler_yR= MinMaxScaler()
                X = scaler_XR.fit_transform(X)
                y = scaler_yR.fit_transform(y)

                 # Séparer le jeu de données en ensembles d'entraînement et de test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size, random_state=42)

               # Reshape des données pour les rendre compatibles avec RNN
                X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
                X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

               # Créer et Model training RNN
                model = Sequential()
                model.add(SimpleRNN(num_input_neurons, input_shape=(1, 7), activation= input_activation))
                for i in range(num_hidden_layers):
                  model.add(Dense(num_neurons, activation= hidden_activation))
                # model.add(Dense(32, activation='relu'))
                model.add(Dense(num_output_neurons, activation= output_activation))
                model.compile(loss='mean_squared_error', optimizer='adam')

                if st.button("Model training"):

                  history = None
                  if model is not None:
                          if use_epochs and use_batch_size:
                              history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_test, y_test))
                          elif use_epochs:
                              history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test))
                          elif use_batch_size:
                              history = model.fit(X_train, y_train, batch_size=batch_size, validation_data=(X_test, y_test))
                          else:
                              history = model.fit(X_train, y_train, validation_data=(X_test, y_test))

                  # save RNN's infos
                  global_var[5][2][0] = model
                  global_var[5][2][1] = scaler_XR
                  global_var[5][2][2] = scaler_yR
                  # Faire des prédictions sur l'ensemble de test
                  y_pred_test = model.predict(X_test)
                  y_pred_train = model.predict(X_train)

                  # Inverser la normalisation des prédictions et des vraies valeurs
                  y_pred_test = scaler_yR.inverse_transform(y_pred_test)
                  y_test = scaler_yR.inverse_transform(y_test)

                # Calculer l'erreur quadratique moyenne des prédictions
                  mse_test = mean_squared_error(y_test, y_pred_test)

                  st.write("Mean squared error (test set):", mse_test)

                # Calculer l'erreur absolue moyenne des prédictions
                  mae_test = mean_absolute_error(y_test, y_pred_test)

                  st.write("Mean Absolute Error (test set):", mae_test)
                  # Calculer le Correlation coefficient R^2
                  R_squared_test = r2_score(y_test, y_pred_test)
                  R_squared_train = r2_score(y_train, y_pred_train)

                  st.write("Correlation coefficient R^2 (test set) : ", R_squared_test)
                  st.write("Correlation coefficient R^2 (training set) : ", R_squared_train)

                elif st.button("Cancel", key="cancel_button_g"):
                  # Afficher un message d'annulation
                  st.warning("The operation was canceled !")

              elif model_selection == "Water Critical Model ":
                st.write("RNN - Water Critical Model ")
                data= df_jaugeage

                #Convertir la colonne 'START_DATE' en datetime avec le format d'origine
                data['START_DATE'] = pd.to_datetime(data['START_DATE'])
                data['jour'] = data['START_DATE'].dt.day
                data['month'] = data['START_DATE'].dt.month
                data['year'] = data['START_DATE'].dt.year


                  # Checkbox pour Bootstrapping
                doubler_lignes = st.sidebar.checkbox("Bootstrapping")

                if doubler_lignes:
                  # Bootstrapping en ajoutant le DataFrame à lui-même
                  # Double the dataset by duplicating the rows
                  data = pd.concat([data, data], ignore_index=True)
                #   data = data.append(data, ignore_index=True)


                # Gestion des valeurs manquantes
                handle_nan = st.sidebar.checkbox("Handling of missing values NAN", key="checkbox_nana")
                if not handle_nan:
                    st.warning("Handling of missing values NAN")
                else:
                    handle_nan = st.sidebar.selectbox("Choose", options=["None", "Delete data point that contain NAN", "Replace with the precedent value", "Replace NAN with avrg"])
                    if handle_nan == "Delete data point that contain NAN":
                        data = data.dropna()
                    elif handle_nan == "Replace with the precedent value":
                        data = data.fillna(method='pad')
                    elif handle_nan == "Replace NAN with avrg":
                        data = data.fillna(data.mean())

                data = data.drop(data[data['FLP'] / data['WHP'] <=  0.75].index)
              
                # Extraire les caractéristiques et la cible
                X = data[['ID','jour', 'month', 'year', 'CHOKE', 'WHP', 'FLP']]
                y = data[['water']]

                # Normaliser les données en utilisant MinMaxScaler
                scaler_XR = MinMaxScaler()
                scaler_yR= MinMaxScaler()
                X = scaler_XR.fit_transform(X)
                y = scaler_yR.fit_transform(y)

                 # Séparer le jeu de données en ensembles d'entraînement et de test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size, random_state=42)

               # Reshape des données pour les rendre compatibles avec RNN
                X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
                X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

               # Créer et Model training RNN
                model = Sequential()
                model.add(SimpleRNN(num_input_neurons, input_shape=(1, 7), activation= input_activation))
                for i in range(num_hidden_layers):
                  model.add(Dense(num_neurons, activation= hidden_activation))
                # model.add(Dense(32, activation='relu'))
                model.add(Dense(num_output_neurons, activation= output_activation))
                model.compile(loss='mean_squared_error', optimizer='adam')

                if st.button("Model training"):
                     # Entraînement du modèle
                  history = None
                  if model is not None:
                          if use_epochs and use_batch_size:
                              history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_test, y_test))
                          elif use_epochs:
                              history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test))
                          elif use_batch_size:
                              history = model.fit(X_train, y_train, batch_size=batch_size, validation_data=(X_test, y_test))
                          else:
                              history = model.fit(X_train, y_train, validation_data=(X_test, y_test))

                  # save RNN's infos
                  global_var[10][2][0] = model
                  global_var[10][2][1] = scaler_XR
                  global_var[10][2][2] = scaler_yR
                  # Faire des prédictions sur l'ensemble de test
                  y_pred_test = model.predict(X_test)
                  y_pred_train = model.predict(X_train)
                  # Inverser la normalisation des prédictions et des vraies valeurs
                  y_pred_test = scaler_yR.inverse_transform(y_pred_test)
                  y_test = scaler_yR.inverse_transform(y_test)

                  # Calculer l'erreur quadratique moyenne des prédictions
                  mse_test = mean_squared_error(y_test, y_pred_test)

                  st.write("Mean squared error (test set):", mse_test)

                # Calculer l'erreur absolue moyenne des prédictions
                  mae_test = mean_absolute_error(y_test, y_pred_test)

                  st.write("Mean Absolute Error (test set):", mae_test)

                

                  # # Afficher les valeurs de y_test et y_pred dans une boucle
                  # for i in range(len(y_test)):
                  #   y_test_inverse = scaler_yR.inverse_transform([y_test[i]])
                  #   y_pred_inverse = scaler_yR.inverse_transform([y_pred_test[i]])
                  #   st.write("y_test:", y_test_inverse)
                  #   st.write("y_pred:", y_pred_inverse)
                  #   st.write()

                  # Calculer le Correlation coefficient R^2
                  R_squared_test = r2_score(y_test, y_pred_test)
                  R_squared_train = r2_score(y_train, y_pred_train)

                  st.write("Correlation coefficient R^2 (test set) : ", R_squared_test)
                  st.write("Correlation coefficient R^2 (training set) : ", R_squared_train)

                elif st.button("Cancel", key="cancel_button_g"):
                  # Afficher un message d'annulation
                  st.warning("The operation was canceled !")
                    
        if model_type == "NonCritical flow Model ":
              st.sidebar.title("Select Fluid Model")
              model_selection = st.sidebar.selectbox("Select Fluid Model", ("None","2Phase  Noncritical Model", "Oil Noncritical Model", "Gas Noncritical Model", "Water Noncritical Model"))
              if model_selection == "2Phase  Noncritical Model":
                st.write("RNN- 2Phase  Noncritical Model")
                data= df_jaugeage

                #Convertir la colonne 'START_DATE' en datetime avec le format d'origine
                data['START_DATE'] = pd.to_datetime(data['START_DATE'])
                data['jour'] = data['START_DATE'].dt.day
                data['month'] = data['START_DATE'].dt.month
                data['year'] = data['START_DATE'].dt.year

                # Checkbox pour Delete repeted data
                supprimer_doublons = st.sidebar.checkbox("Delete repeted data")
                if supprimer_doublons:
                  # Identifier les lignes en double
                  doublons = data.duplicated()
                  # Supprimer les lignes en double
                  data = data[~doublons]
                
                 # Gestion des valeurs manquantes
                handle_nan = st.sidebar.checkbox("Handling of missing values NAN", key="checkbox_nani")
                if not handle_nan:
                    st.warning("Handling of missing values NAN")
                else:
                    handle_nan = st.sidebar.selectbox("Choose", options=["None", "Delete data point that contain NAN", "Replace with the precedent value", "Replace NAN with zero", "Replace NAN with avrg"])
                    if handle_nan == "Delete data point that contain NAN":
                        data = data.dropna()
                    elif handle_nan == "Replace with the precedent value":
                        data = data.fillna(method='pad')
                    elif handle_nan == "Replace NAN with zero":
                        data = data.fillna(0)
                    elif handle_nan == "Replace NAN with avrg":
                        data = data.fillna(data.mean())
                
                data = data.drop(data[data['FLP'] / data['WHP'] <= 0.75].index)

                # Split the dataset into input features (X) and target variables (y)
                X = data[['ID', 'jour', 'month', 'year', 'CHOKE', 'WHP', 'FLP']]
                y = data[['quantiter oil','quantiter gaz']]

                # Normaliser les données en utilisant MinMaxScaler
                scaler_XR = MinMaxScaler()
                scaler_yR= MinMaxScaler()
                X = scaler_XR.fit_transform(X)
                y = scaler_yR.fit_transform(y)

                 # Séparer le jeu de données en ensembles d'entraînement et de test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size, random_state=42)

               # Reshape des données pour les rendre compatibles avec RNN
                X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
                X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

               # Créer et Model training RNN
                model = Sequential()
                model.add(SimpleRNN(num_input_neurons, input_shape=(1, 7), activation= input_activation))
                for i in range(num_hidden_layers):
                  model.add(Dense(num_neurons, activation= hidden_activation))
                # model.add(Dense(32, activation='relu'))
                model.add(Dense(num_output_neurons, activation= output_activation))
                model.compile(loss='mean_squared_error', optimizer='adam')
                                
                if st.button("Model training"):

                  history = None
                  if model is not None:
                          if use_epochs and use_batch_size:
                              history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_test, y_test))
                          elif use_epochs:
                              history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test))
                          elif use_batch_size:
                              history = model.fit(X_train, y_train, batch_size=batch_size, validation_data=(X_test, y_test))
                          else:
                              history = model.fit(X_train, y_train, validation_data=(X_test, y_test))

                  # save RNN's infos
                  global_var[6][2][0] = model
                  global_var[6][2][1] = scaler_XR
                  global_var[6][2][2] = scaler_yR
                # Faire des prédictions sur l'ensemble de test
                  y_pred_test = model.predict(X_test)
                  y_pred_train = model.predict(X_train)

                  # Inverser la normalisation des prédictions et des vraies valeurs
                  y_pred_test = scaler_yR.inverse_transform(y_pred_test)
                  y_test = scaler_yR.inverse_transform(y_test)


                # Calculer les erreurs MSE et MAE pour chaque variable
                  mse_oil = mean_squared_error(y_test[:, 0], y_pred_test[:, 0])
                  mae_oil = mean_absolute_error(y_test[:, 0], y_pred_test[:, 0])
                  mse_gaz = mean_squared_error(y_test[:, 1], y_pred_test[:, 1])
                  mae_gaz = mean_absolute_error(y_test[:, 1], y_pred_test[:, 1])

                # Afficher les résultats
                  st.write("MSE oil:", mse_oil)
                  st.write("MAE oil:", mae_oil)
                  st.write("MSE gaz:", mse_gaz)
                  st.write("MAE gaz:", mae_gaz)
                  # Calculer le Correlation coefficient R^2
                  R_squared_test = r2_score(y_test, y_pred_test)
                  R_squared_train = r2_score(y_train, y_pred_train)

                  st.write("Correlation coefficient R^2 (test set) : ", R_squared_test)
                  st.write("Correlation coefficient R^2 (training set) : ", R_squared_train)

                elif st.button("Cancel", key="cancel_button_g"):
                  # Afficher un message d'annulation
                  st.warning("The operation was canceled !")

              elif model_selection == "Oil Noncritical Model":
                st.write("RNN - Oil Noncritical Model")
                data= df_jaugeage
                #Convertir la colonne 'START_DATE' en datetime avec le format d'origine
                data['START_DATE'] = pd.to_datetime(data['START_DATE'])
                data['jour'] = data['START_DATE'].dt.day
                data['month'] = data['START_DATE'].dt.month
                data['year'] = data['START_DATE'].dt.year

                # Checkbox pour Delete repeted data
                supprimer_doublons = st.sidebar.checkbox("Delete repeted data")
                if supprimer_doublons:
                  # Identifier les lignes en double
                  doublons = data.duplicated()
                  # Supprimer les lignes en double
                  data = data[~doublons]
                
                 # Gestion des valeurs manquantes
                handle_nan = st.sidebar.checkbox("Handling of missing values NAN", key="checkbox_nank")
                if not handle_nan:
                    st.warning("Handling of missing values NAN")
                else:
                    handle_nan = st.sidebar.selectbox("Choose", options=["None", "Delete data point that contain NAN", "Replace with the precedent value", "Replace NAN with zero", "Replace NAN with avrg"])
                    if handle_nan == "Delete data point that contain NAN":
                        data = data.dropna()
                    elif handle_nan == "Replace with the precedent value":
                        data = data.fillna(method='pad')
                    elif handle_nan == "Replace NAN with zero":
                        data = data.fillna(0)
                    elif handle_nan == "Replace NAN with avrg":
                        data = data.fillna(data.mean())

                data = data.drop(data[data['FLP'] / data['WHP'] <= 0.75].index)
                
                # Split the dataset into input features (X) and target variables (y)
                X = data[['ID','jour', 'month', 'year', 'CHOKE', 'WHP', 'FLP']]
                y = data[['quantiter oil']]

                # Normaliser les données en utilisant MinMaxScaler
                scaler_XR = MinMaxScaler()
                scaler_yR= MinMaxScaler()
                X = scaler_XR.fit_transform(X)
                y = scaler_yR.fit_transform(y)

                 # Séparer le jeu de données en ensembles d'entraînement et de test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size, random_state=42)

               # Reshape des données pour les rendre compatibles avec RNN
                X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
                X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

               # Créer et Model training RNN
                model = Sequential()
                model.add(SimpleRNN(num_input_neurons, input_shape=(1, 7), activation= input_activation))
                for i in range(num_hidden_layers):
                  model.add(Dense(num_neurons, activation= hidden_activation))
                # model.add(Dense(32, activation='relu'))
                model.add(Dense(num_output_neurons, activation= output_activation))
                model.compile(loss='mean_squared_error', optimizer='adam')

                if st.button("Model training"):
                  history = None
                  if model is not None:
                          if use_epochs and use_batch_size:
                              history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_test, y_test))
                          elif use_epochs:
                              history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test))
                          elif use_batch_size:
                              history = model.fit(X_train, y_train, batch_size=batch_size, validation_data=(X_test, y_test))
                          else:
                              history = model.fit(X_train, y_train, validation_data=(X_test, y_test))

                  # save RNN's infos
                  global_var[7][2][0] = model
                  global_var[7][2][1] = scaler_XR
                  global_var[7][2][2] = scaler_yR
                 # Faire des prédictions sur l'ensemble de test
                  y_pred_test = model.predict(X_test)
                  y_pred_test = y_pred_test.reshape(-1, 1)
                  y_pred_train = model.predict(X_train)
                  y_pred_train  = y_pred_train.reshape(-1, 1)
                  # Inverser la normalisation des prédictions et des vraies valeurs
                  y_pred_test = scaler_yR.inverse_transform(y_pred_test)
                  y_test = scaler_yR.inverse_transform(y_test)

                 # Calculer l'erreur quadratique moyenne des prédictions
                  mse_test = mean_squared_error(y_test, y_pred_test)
                  st.write("Mean squared error (test set):", mse_test)

                  # Calculer l'erreur absolue moyenne des prédictions
                  mae_test = mean_absolute_error(y_test, y_pred_test)
                  st.write("Mean Absolute Error (test set):", mae_test)
                  # Calculer le Correlation coefficient R^2
                  R_squared_test = r2_score(y_test, y_pred_test)
                  R_squared_train = r2_score(y_train, y_pred_train)

                  st.write("Correlation coefficient R^2 (test set) : ", R_squared_test)
                  st.write("Correlation coefficient R^2 (training set) : ", R_squared_train)

                elif st.button("Cancel", key="cancel_button_g"):
                  # Afficher un message d'annulation
                  st.warning("The operation was canceled !")

              elif model_selection == "Gas Noncritical Model":
                st.write("RNN - Gas Noncritical Model")
                # st.write("RNN - 2Phase  Model")
                data= df_jaugeage
                #Convertir la colonne 'START_DATE' en datetime avec le format d'origine
                data['START_DATE'] = pd.to_datetime(data['START_DATE'])
                data['jour'] = data['START_DATE'].dt.day
                data['month'] = data['START_DATE'].dt.month
                data['year'] = data['START_DATE'].dt.year

                # Checkbox pour Delete repeted data
                supprimer_doublons = st.sidebar.checkbox("Delete repeted data")
                if supprimer_doublons:
                  # Identifier les lignes en double
                  doublons = data.duplicated()
                  # Supprimer les lignes en double
                  data = data[~doublons]
                
                 # Gestion des valeurs manquantes
                handle_nan = st.sidebar.checkbox("Handling of missing values NAN", key="checkbox_nanj")
                if not handle_nan:
                    st.warning("Handling of missing values NAN")
                else:
                    handle_nan = st.sidebar.selectbox("Choose", options=["None", "Delete data point that contain NAN", "Replace with the precedent value", "Replace NAN with zero", "Replace NAN with avrg"])
                    if handle_nan == "Delete data point that contain NAN":
                        data = data.dropna()
                    elif handle_nan == "Replace with the precedent value":
                        data = data.fillna(method='pad')
                    elif handle_nan == "Replace NAN with zero":
                        data = data.fillna(0)
                    elif handle_nan == "Replace NAN with avrg":
                        data = data.fillna(data.mean())

                data = data.drop(data[data['FLP'] / data['WHP'] <=  0.75].index)
                
                # Split the dataset into input features (X) and target variables (y)
                X = data[['ID','jour', 'month', 'year', 'CHOKE', 'WHP', 'FLP']]
                y = data[['quantiter gaz']]

                # Normaliser les données en utilisant MinMaxScaler
                scaler_XR = MinMaxScaler()
                scaler_yR= MinMaxScaler()
                X = scaler_XR.fit_transform(X)
                y = scaler_yR.fit_transform(y)

                 # Séparer le jeu de données en ensembles d'entraînement et de test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size, random_state=42)

               # Reshape des données pour les rendre compatibles avec RNN
                X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
                X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

               # Créer et Model training RNN
                model = Sequential()
                model.add(SimpleRNN(num_input_neurons, input_shape=(1, 7), activation= input_activation))
                for i in range(num_hidden_layers):
                  model.add(Dense(num_neurons, activation= hidden_activation))
                # model.add(Dense(32, activation='relu'))
                model.add(Dense(num_output_neurons, activation= output_activation))
                model.compile(loss='mean_squared_error', optimizer='adam')

                if st.button("Model training"):

                  history = None
                  if model is not None:
                          if use_epochs and use_batch_size:
                              history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_test, y_test))
                          elif use_epochs:
                              history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test))
                          elif use_batch_size:
                              history = model.fit(X_train, y_train, batch_size=batch_size, validation_data=(X_test, y_test))
                          else:
                              history = model.fit(X_train, y_train, validation_data=(X_test, y_test))

                  # save RNN's infos
                  global_var[8][2][0] = model
                  global_var[8][2][1] = scaler_XR
                  global_var[8][2][2] = scaler_yR
                # Faire des prédictions sur l'ensemble de test
                  y_pred_test = model.predict(X_test)
                  y_pred_test = y_pred_test.reshape(-1, 1)
                  y_pred_train = model.predict(X_train)
                  y_pred_train  = y_pred_train.reshape(-1, 1)

                  # Inverser la normalisation des prédictions et des vraies valeurs
                  y_pred_test = scaler_yR.inverse_transform(y_pred_test)
                  y_test = scaler_yR.inverse_transform(y_test)

                # Calculer l'erreur quadratique moyenne des prédictions
                  mse_test = mean_squared_error(y_test, y_pred_test)

                  st.write("Mean squared error (test set):", mse_test)

                # Calculer l'erreur absolue moyenne des prédictions
                  mae_test = mean_absolute_error(y_test, y_pred_test)

                  st.write("Mean Absolute Error (test set):", mae_test)

                  # Calculer le Correlation coefficient R^2
                  R_squared_test = r2_score(y_test, y_pred_test)
                  R_squared_train = r2_score(y_train, y_pred_train)

                  st.write("Correlation coefficient R^2 (test set) : ", R_squared_test)
                  st.write("Correlation coefficient R^2 (training set) : ", R_squared_train)

                elif st.button("Cancel", key="cancel_button_g"):
                  # Afficher un message d'annulation
                  st.warning("The operation was canceled !")

              elif model_selection == "Water Noncritical Model":
                st.write("RNN  - Water Noncritical Model")
                data= df_jaugeage
                #Convertir la colonne 'START_DATE' en datetime avec le format d'origine
                data['START_DATE'] = pd.to_datetime(data['START_DATE'])
                data['jour'] = data['START_DATE'].dt.day
                data['month'] = data['START_DATE'].dt.month
                data['year'] = data['START_DATE'].dt.year

                 # Checkbox pour Bootstrapping
                doubler_lignes = st.sidebar.checkbox("Bootstrapping")

                if doubler_lignes:
                  # Bootstrapping en ajoutant le DataFrame à lui-même
                  # Double the dataset by duplicating the rows
                  data = pd.concat([data, data], ignore_index=True)
                #   data = data.append(data, ignore_index=True)

                
                 # Gestion des valeurs manquantes
                handle_nan = st.sidebar.checkbox("Handling of missing values NAN", key="checkbox_nanf")
                if not handle_nan:
                    st.warning("Handling of missing values NAN")
                else:
                    handle_nan = st.sidebar.selectbox("Choose", options=["None", "Delete data point that contain NAN", "Replace with the precedent value", "Replace NAN with avrg"])
                    if handle_nan == "Delete data point that contain NAN":
                        data = data.dropna()
                    elif handle_nan == "Replace with the precedent value":
                        data = data.fillna(method='pad')
                    
                    elif handle_nan == "Replace NAN with avrg":
                        data = data.fillna(data.mean())

                data = data.drop(data[data['FLP'] / data['WHP'] <=  0.75].index)
              
                # Extraire les caractéristiques et la cible
                X = data[['ID','jour', 'month', 'year', 'CHOKE', 'WHP', 'FLP']]
                y = data[['water']]

                
                # Normaliser les données en utilisant MinMaxScaler
                scaler_XR = MinMaxScaler()
                scaler_yR= MinMaxScaler()
                X = scaler_XR.fit_transform(X)
                y = scaler_yR.fit_transform(y)

                 # Séparer le jeu de données en ensembles d'entraînement et de test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size, random_state=42)

               # Reshape des données pour les rendre compatibles avec RNN
                X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
                X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

               # Créer et Model training RNN
                model = Sequential()
                model.add(SimpleRNN(num_input_neurons, input_shape=(1, 7), activation= input_activation))
                for i in range(num_hidden_layers):
                  model.add(Dense(num_neurons, activation= hidden_activation))
                # model.add(Dense(32, activation='relu'))
                model.add(Dense(num_output_neurons, activation= output_activation))
                model.compile(loss='mean_squared_error', optimizer='adam')

                if st.button("Model training"):

                  history = None
                  if model is not None:
                          if use_epochs and use_batch_size:
                              history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_test, y_test))
                          elif use_epochs:
                              history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test))
                          elif use_batch_size:
                              history = model.fit(X_train, y_train, batch_size=batch_size, validation_data=(X_test, y_test))
                          else:
                              history = model.fit(X_train, y_train, validation_data=(X_test, y_test))

                  # save RNN's infos
                  global_var[11][2][0] = model
                  global_var[11][2][1] = scaler_XR
                  global_var[11][2][2] = scaler_yR
                # Faire des prédictions sur l'ensemble de test
                  y_pred_test = model.predict(X_test)
                  y_pred_test = y_pred_test.reshape(-1, 1)
                  y_pred_train = model.predict(X_train)
                  y_pred_train  = y_pred_train.reshape(-1, 1)

                  # Inverser la normalisation des prédictions et des vraies valeurs
                  y_pred_test = scaler_yR.inverse_transform(y_pred_test)
                  y_test = scaler_yR.inverse_transform(y_test)

                # Calculer l'erreur quadratique moyenne des prédictions
                  mse_test = mean_squared_error(y_test, y_pred_test)

                  st.write("Mean squared error (test set):", mse_test)

                # Calculer l'erreur absolue moyenne des prédictions
                  mae_test = mean_absolute_error(y_test, y_pred_test)

                  st.write("Mean Absolute Error (test set):", mae_test)

                  # Calculer le Correlation coefficient R^2
                  R_squared_test = r2_score(y_test, y_pred_test)
                  R_squared_train = r2_score(y_train, y_pred_train)

                  st.write("Correlation coefficient R^2 (test set) : ", R_squared_test)
                  st.write("Correlation coefficient R^2 (training set) : ", R_squared_train)

                elif st.button("Cancel", key="cancel_button_g"):
                  # Afficher un message d'annulation
                  st.warning("The operation was canceled !")


  # estimation_button = st.sidebar.button("Make a new estimation")
    # if estimation_button:

     # Autres contenus de votre application Streamlit
    title_style = """
        <style>
        h1 {
            color: orange;
        }
        </style>
    """

    # Ajouter le titre avec la couleur orange
    st.markdown("<p style=' font-weight: bold; text-align:center; font-size: 30px; color: orange;'>Upload daily measurement data file.</p>", unsafe_allow_html=True)
    # st.subheader("Apload the daily data file :")
    def process_uploaded_file(file):
        # Process the uploaded CSV file here
        st.write('File was read')
        uploaded_file_daily = pd.read_csv(file)
        model_general(uploaded_file_daily, global_var, dj)
    # Create a session state variable
    session_state = st.session_state
    # Initialize the flag to track file upload status
    file_uploaded = False
    # Check if a file is already uploaded in the session state
    if 'uploaded_file' in session_state:
        uploaded_file = session_state['uploaded_file']
        file_uploaded = True
    # File uploader
    uploaded_file = st.file_uploader("Daily measurement data is required to execute IA models that are fitted on gauging data", type="csv")
    # Check if a file is uploaded
    if uploaded_file is not None:
        # Save the uploaded file in the session state
        session_state['uploaded_file'] = uploaded_file
        file_uploaded = True
    # Check if the file is uploaded and proceed accordingly
    if file_uploaded:
        st.write("CSV file has been uploaded!")
        uploaded_file = session_state['uploaded_file']
        process_uploaded_file(uploaded_file)
    else:
        st.write("No CSV file uploaded yet.")