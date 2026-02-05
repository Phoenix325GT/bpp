import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("bp_data.csv")
print("Columns:", df.columns)

SMILES_COL = "SMILES"
TARGET_COL = "boiling_point"

# -----------------------------
# Convert SMILES to molecules
# -----------------------------
df["mol"] = df[SMILES_COL].apply(Chem.MolFromSmiles)
df = df[df["mol"].notnull()].reset_index(drop=True)

# -----------------------------
# Descriptor set (chemically meaningful)
# -----------------------------
descriptor_fns = [
    Descriptors.MolWt,              # size
    Descriptors.HeavyAtomCount,     # size (better behaved)
    Descriptors.MolLogP,            # dispersion + polarity proxy
    Descriptors.TPSA,               # polarity / H-bond surface
    Descriptors.NumHDonors,         # H-bonding
    Descriptors.NumHAcceptors,      # H-bonding
    Descriptors.NumRotatableBonds,  # entropy / flexibility
    Descriptors.RingCount,          # rigidity
    Descriptors.FractionCSP3        # branching vs flatness
]

X = np.array([[fn(m) for fn in descriptor_fns] for m in df["mol"]])

# log-transform target (important)
y = np.log(df[TARGET_COL].values)

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Model
# -----------------------------
model = RandomForestRegressor(
    n_estimators=500,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Evaluation
# -----------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(
    np.exp(y_test),
    np.exp(y_pred)
)

print("MAE (Kelvin):", mae)

# -----------------------------
# Predict new molecule
# -----------------------------
j = input("Enter SMILES string of the molecule: ")
m = Chem.MolFromSmiles(j)

if m is None:
    print("Invalid SMILES string.")
else:
    m_features = np.array([[fn(m) for fn in descriptor_fns]])
    pred_bp = np.exp(model.predict(m_features)[0])
    print("Predicted boiling point:", pred_bp, "K")
