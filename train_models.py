import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# ======= Crop Model =======
crop_df = pd.read_csv("Dataset/Crop_recommendation.csv")

X_crop = crop_df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y_crop = crop_df['label']

Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_crop, y_crop, test_size=0.2, random_state=42)

crop_model = RandomForestClassifier()
crop_model.fit(Xc_train, yc_train)

with open("crop_model.pkl", "wb") as f:
    pickle.dump(crop_model, f)

# ======= Fertilizer Model =======
fert_df = pd.read_csv("Dataset/Fertilizer Prediction.csv")

# Encode 'Soil Type' and 'Crop Type'
le_soil = LabelEncoder()
le_crop = LabelEncoder()

fert_df['Soil Type'] = le_soil.fit_transform(fert_df['Soil Type'])
fert_df['Crop Type'] = le_crop.fit_transform(fert_df['Crop Type'])

X_fert = fert_df[['Temparature', 'Humidity ', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Phosphorous', 'Potassium']]
y_fert = fert_df['Fertilizer Name']

Xf_train, Xf_test, yf_train, yf_test = train_test_split(X_fert, y_fert, test_size=0.2, random_state=42)

fert_model = RandomForestClassifier()
fert_model.fit(Xf_train, yf_train)

# Save fertilizer model and encoders
with open("fertilizer_model.pkl", "wb") as f:
    pickle.dump(fert_model, f)

with open("soil_encoder.pkl", "wb") as f:
    pickle.dump(le_soil, f)

with open("crop_encoder.pkl", "wb") as f:
    pickle.dump(le_crop, f)
