## RAC Prediction App
# Copyright (c) 2024 (Amira Ahmed, Wu Jin, Mosaad Ali ). All rights reserved.

# You may not copy, modify, publish, transmit, distribute, perform, display, or sell any part of this software without the prior written permission of the authors.
#
# Corresponding Author: Amira Hamdy Ali Ahmed
# Email: amira672012@yahoo.com

import streamlit as st
import pandas as pd
import joblib

# Function to load models and scalers based on the selected model type
def load_model_scaler(selected_model):
    try:
        if selected_model == 'RAC-unconfined':
            scaler = joblib.load("scaler_CS.pkl")
            model = joblib.load("best_model_CS.pkl")
            return model, scaler
        elif selected_model == 'FRP-confined RAC':
            scaler = joblib.load("scaler_Fcc_Scc.pkl")
            model_fcc = joblib.load("best_model_fcc.pkl")
            model_scc = joblib.load("best_model_scc.pkl")
            return model_fcc, model_scc, scaler
    except FileNotFoundError:
        st.error(f"Required files for {selected_model} model not found.")
        return None, None, None if selected_model == 'FRP-confined RAC' else (None, None)

# Function to predict using CSV file
def predict_from_csv(df, selected_model):
    try:
        if selected_model == 'RAC-unconfined':
            required_columns = ['C', 'W', 'NFA', 'NCA', 'RFA', 'RCA', 'SF', 'FA', 'Age']
            model, scaler = load_model_scaler(selected_model)
            X_scaled = scaler.transform(df[required_columns])
            predictions = model.predict(X_scaled)
            df['Predicted CS (MPa)'] = predictions
        elif selected_model == 'FRP-confined RAC':
            required_columns = ['AT', '%RA', 'MSA', '%W/C', 'H', 'Efrp', '%Rfrp', 'fco', '%Sco', '%RS']
            model_fcc, model_scc, scaler = load_model_scaler(selected_model)
            if not (model_fcc and model_scc and scaler):
                return df  # Skip processing if models/scaler aren't loaded
            X_scaled = scaler.transform(df[required_columns])
            predictions_fcc = model_fcc.predict(X_scaled)
            predictions_scc = model_scc.predict(X_scaled)

            # Calculate strength and strain improvement
            strength_improvement = (predictions_fcc / df['fco']) * 100
            strain_improvement = (predictions_scc / df['%Sco']) * 100

            df['Predicted Fcc (MPa)'] = predictions_fcc
            df['Predicted %Scc'] = predictions_scc
            df['Strength Improvement (%)'] = strength_improvement
            df['Strain Improvement (%)'] = strain_improvement
        
        return df
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return df

# Function to handle manual input predictions
def predict_from_manual_input(selected_model, inputs):
    try:
        if selected_model == 'RAC-unconfined':
            input_data = pd.DataFrame([inputs])
            model, scaler = load_model_scaler(selected_model)
            if not (model and scaler):
                return  # Skip processing if models/scaler aren't loaded
            X_scaled = scaler.transform(input_data)
            prediction = model.predict(X_scaled)
            st.write(f"**Predicted Compressive Strength (CS): {prediction[0]:.2f} MPa**")
        elif selected_model == 'FRP-confined RAC':
            input_data = pd.DataFrame([inputs])
            model_fcc, model_scc, scaler = load_model_scaler(selected_model)
            if not (model_fcc and model_scc and scaler):
                return  # Skip processing if models/scaler aren't loaded
            X_scaled = scaler.transform(input_data)
            prediction_fcc = model_fcc.predict(X_scaled)
            prediction_scc = model_scc.predict(X_scaled)

            # Calculating Strength and Strain Improvement
            strength_improvement = (prediction_fcc[0] / inputs['fco']) * 100
            strain_improvement = (prediction_scc[0] / inputs['%Sco']) * 100

            st.write(f"**Predicted Ultimate Compressive Strength (Fcc): {prediction_fcc[0]:.2f} MPa**")
            st.write(f"**Predicted Ultimate Axial Strain (%Scc): {prediction_scc[0]:.2f}%**")
            st.write(f"**Strength Improvement for RAC: {strength_improvement:.2f}%**")
            st.write(f"**Strain Improvement for RAC: {strain_improvement:.2f}%**")
    except Exception as e:
        st.error(f"An error occurred during manual prediction: {e}")

# Main App Interface
st.title("RAC Prediction App")
st.markdown("Predict the mechanical properties of unconfined and FRP-confined Recycled Aggregate Concrete (RAC).")

# Dropdown menu for model selection
model_choice = st.selectbox("Select Model Type:", ['Select a model', 'RAC-unconfined', 'FRP-confined RAC'])

# Upload CSV file or enter data manually
if model_choice != 'Select a model':
    st.markdown("### Input Data")
    upload_method = st.radio("Choose how to input data:", ('Upload CSV File', 'Enter Manually'))

    # For CSV input
    if upload_method == 'Upload CSV File':
        uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                result_df = predict_from_csv(df, model_choice)
                st.dataframe(result_df)

                # Allow the user to download the prediction results
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name='prediction_results.csv',
                    mime='text/csv',
                )

            except Exception as e:
                st.error(f"An error occurred: {e}")

    # For manual input
    elif upload_method == 'Enter Manually':
        if model_choice == 'RAC-unconfined':
            inputs = {
                'C': st.number_input('Cement Content (C) [kg/m³]', min_value=0.0),
                'W': st.number_input('Water Content (W) [kg/m³]', min_value=0.0),
                'NFA': st.number_input('Natural Fine Aggregate (NFA) [kg/m³]', min_value=0.0),
                'NCA': st.number_input('Natural Coarse Aggregate (NCA) [kg/m³]', min_value=0.0),
                'RFA': st.number_input('Recycled Fine Aggregate (RFA) [kg/m³]', min_value=0.0),
                'RCA': st.number_input('Recycled Coarse Aggregate (RCA) [kg/m³]', min_value=0.0),
                'SF': st.number_input('Silica Fume Content (SF) [kg/m³]', min_value=0.0),
                'FA': st.number_input('Fly Ash Content (FA) [kg/m³]', min_value=0.0),
                'Age': st.number_input('Age (days)', min_value=0.0)
            }
            if st.button("Predict"):
                predict_from_manual_input(model_choice, inputs)

        elif model_choice == 'FRP-confined RAC':
            inputs = {
                'AT': st.selectbox('Aggregate Type (AT)', options=[0, 1, 2, 3], format_func=lambda x: ['NA', 'RCA', 'RCL', 'RBA'][x]),
                '%RA': st.number_input('Recycled Aggregate Replacement Ratio (%RA) [%]', min_value=0.0),
                'MSA': st.number_input('Maximum Size of Aggregate (MSA) [mm]', min_value=0.0),
                '%W/C': st.number_input('Effective Water-to-Cement Ratio (%W/C)', min_value=0.0),
                'H': st.number_input('Column Height (H) [mm]', min_value=0.0),
                'Efrp': st.number_input('Elastic Modulus of FRP (Efrp) [GPa]', min_value=0.0),
                '%Rfrp': st.number_input('FRP Reinforcement Ratio (%Rfrp) [%]', min_value=0.0),
                'fco': st.number_input('Compressive Strength of Plain Concrete (fco) [MPa]', min_value=0.0),
                '%Sco': st.number_input('Peak Strain of Plain Concrete (%Sco) [%]', min_value=0.0),
                '%RS': st.number_input('FRP Rupture Strain (%RS) [%]', min_value=0.0)
            }
            if st.button("Predict"):
                predict_from_manual_input(model_choice, inputs)

# Footer
st.markdown("---")
st.markdown("© 2024 (Amira Ahmed, Wu Jin, Mosaad Ali ). All rights reserved.")
st.markdown("Developed by Amira Ahmed. Contact: amira672012@yahoo.com")
