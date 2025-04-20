import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("Dataset/Fertilizer Prediction.csv")

# Strip column names of any whitespace
df.columns = df.columns.str.strip()

# Encode 'Soil Type' and 'Crop Type'
le_soil = LabelEncoder()
df['Soil Type'] = le_soil.fit_transform(df['Soil Type'])

le_crop = LabelEncoder()
df['Crop Type'] = le_crop.fit_transform(df['Crop Type'])

# Features and Target
X = df.drop(columns=['Fertilizer Name'])
y = df['Fertilizer Name']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
with open('fertilizer_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save encoders
with open('le_soil.pkl', 'wb') as f:
    pickle.dump(le_soil, f)

with open('le_crop.pkl', 'wb') as f:
    pickle.dump(le_crop, f)

print("✅ Fertilizer model and encoders saved successfully!")
