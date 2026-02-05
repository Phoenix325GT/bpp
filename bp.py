import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import os

# 1. Handle File Paths for the Cloud
base_path = os.path.dirname(__file__)
file_path = os.path.join(base_path, "bp_data.csv")

st.title("🌡️ Organic Boiling Point Predictor")
st.markdown("Predicting properties using **RDKit** and **Random Forest**.")

# 2. Cache the Training (So it doesn't take 5 minutes to load)
@st.cache_resource
def train_model():
    df = pd.read_csv(file_path)
    SMILES_COL = "SMILES"
    TARGET_COL = "boiling_point"
    
    df["mol"] = df[SMILES_COL].apply(Chem.MolFromSmiles)
    df = df[df["mol"].notnull()]
    
    descriptor_fns = [
        Descriptors.MolWt, Descriptors.MolLogP, 
        Descriptors.TPSA, Descriptors.NumRotatableBonds, Descriptors.RingCount
    ]
    
    X = np.array([[fn(m) for fn in descriptor_fns] for m in df["mol"]])
    y = df[TARGET_COL].values
    
    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X, y)
    return model, descriptor_fns

# Run the training
with st.spinner("Training model... please wait."):
    model, descriptor_fns = train_model()

# 3. Replace input() with a text box
user_input = st.text_input("Enter SMILES string:", "CCO")

if st.button("Predict Boiling Point"):
    mol = Chem.MolFromSmiles(user_input)
    if mol:
        features = np.array([[fn(mol) for fn in descriptor_fns]])
        prediction = model.predict(features)[0]
        st.success(f"Predicted BP: **{prediction:.2f} °C**")
        
        # Display the molecule structure
        img = Chem.Draw.MolToImage(mol)
        st.image(img, caption="Molecular Structure")
    else:
        st.error("Invalid SMILES. Try 'CCCC' or 'c1ccccc1'")
