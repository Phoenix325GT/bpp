import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# ---------------------------------
# 1. Load dataset
# ---------------------------------
df = pd.read_csv("bp_data.csv")
print("Columns:", df.columns)

SMILES_COL = "SMILES"
TARGET_COL = "boiling_point"

# ---------------------------------
# 2. Convert SMILES → RDKit molecules
# ---------------------------------
df["mol"] = df[SMILES_COL].apply(Chem.MolFromSmiles)
df = df[df["mol"].notnull()].reset_index(drop=True)

# ---------------------------------
# 3. Morgan fingerprint generator (modern API)
# ---------------------------------
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048
)

def morgan_fp(mol):
    fp = morgan_gen.GetFingerprint(mol)
    arr = np.zeros((fp.GetNumBits(),), dtype=int)
    ConvertToNumpyArray(fp, arr)
    return arr

# ---------------------------------
# 4. Build feature matrix
# ---------------------------------
X = np.array([morgan_fp(m) for m in df["mol"]])

# ---------------------------------
# 5. Target (log boiling point)
# ---------------------------------
y = np.log(df[TARGET_COL].values)

# ---------------------------------
# 6. Train–test split
# ---------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------
# 7. Random Forest model
# ---------------------------------
model = RandomForestRegressor(
    n_estimators=600,
    min_samples_leaf=2,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42
)

# ---------------------------------
# 8. Train
# ---------------------------------
model.fit(X_train, y_train)

# ---------------------------------
# 9. Test evaluation
# ---------------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(
    np.exp(y_test),
    np.exp(y_pred)
)

print("Test MAE (K):", mae)

# ---------------------------------
# 10. Cross-validation
# ---------------------------------
cv = KFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

print("CV MAE (K):", -cv_scores.mean())
print("CV Std:", cv_scores.std())

# ---------------------------------
# 11. Save trained model
# ---------------------------------
joblib.dump(model, "bp_fingerprint_model.pkl")
print("Model saved as bp_fingerprint_model.pkl")

# ---------------------------------
# 12. Predict new molecule
# ---------------------------------
j = input("Enter SMILES string: ")
m = Chem.MolFromSmiles(j)

if m is None:
    print("Invalid SMILES string.")
else:
    m_fp = morgan_fp(m).reshape(1, -1)
    bp_pred = np.exp(model.predict(m_fp)[0])
    print("Predicted boiling point (K):", bp_pred)
