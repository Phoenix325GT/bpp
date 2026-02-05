import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load data
df = pd.read_csv("bp_data.csv")

# Check columns (debug safety)
print("Columns:", df.columns)

# yeh woh cheez hai jaha hum apna target aur features define karte hain
SMILES_COL = "SMILES"
TARGET_COL = "boiling_point"

# SMILES se better Mol hai humare liye kyunki descriptors nikalne mein asaani hoti hai
df["mol"] = df[SMILES_COL].apply(Chem.MolFromSmiles)
df = df[df["mol"].notnull()]

# yeh hai descriptors jo hum use karenge
descriptor_fns = [
    Descriptors.MolWt,
    Descriptors.MolLogP,
    Descriptors.TPSA,
    Descriptors.NumRotatableBonds,
    Descriptors.RingCount,
]

X = np.array([[fn(m) for fn in descriptor_fns] for m in df["mol"]])
y = df[TARGET_COL].values

# yeh hume help karega data ko train aur test mein split karne mein, split ka matlab hai kuch data model ko sikhane ke liye aur kuch uski performance check karne ke liye
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# yaha hum model banate hain aur usse train karte hain ek regressor algorithm ke saath jo basic regression problems ke liye acha hai
model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# yaha hum model ki performance check karte hain using Mean Absolute Error (MAE)
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))

j = input("Enter SMILES string of the molecule to predict its boiling point: ")
m = Chem.MolFromSmiles(j)
m_features= np.array([[fn(m) for fn in descriptor_fns]])
print("Predicted BP of given molecular sample is: ", model.predict(m_features)[0])
#Ab tu koisa aise molecule ka boiling point pata kar sakta hai bas SMILES string deke, which is not even present in the original dataset! isnt it amazing?