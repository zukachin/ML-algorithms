from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.utils import resample

# Define the original dataset
data = {
    'Character Name': ['Tanjiro Kamado', 'Muzan Kibutsuji', 'Giyu Tomioka', 'Zenitsu Agatsuma', 'Inosuke Hashibira', 'Nezuko Kamado'],
    'Age': [15, 1000, 19, 16, 18, 14],
    'Breathing Style': [1, 0, 1, 1, 1, 0],  # 1 for Yes, 0 for No
    'Weapon Type': [1, 0, 1, 1, 1, 0],      # 1 for Nichirin Sword, 0 for Other
    'Demon Slayer': [1, 0, 1, 1, 1, 0]      # 1 for Yes, 0 for No
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Create 3 bootstrapped datasets
bootstrap_1 = resample(df, replace=True, n_samples=len(df), random_state=1)
bootstrap_2 = resample(df, replace=True, n_samples=len(df), random_state=2)
bootstrap_3 = resample(df, replace=True, n_samples=len(df), random_state=3)

# Prepare features and targets for each bootstrapped dataset
X1, y1 = bootstrap_1[['Age', 'Breathing Style', 'Weapon Type']], bootstrap_1['Demon Slayer']
X2, y2 = bootstrap_2[['Age', 'Breathing Style', 'Weapon Type']], bootstrap_2['Demon Slayer']
X3, y3 = bootstrap_3[['Age', 'Breathing Style', 'Weapon Type']], bootstrap_3['Demon Slayer']

# Initialize the RandomForestClassifier
rf = RandomForestClassifier(n_estimators=10, random_state=42)

# Train the model on each bootstrapped dataset
rf.fit(X1, y1)
predictions_1 = rf.predict(df[['Age', 'Breathing Style', 'Weapon Type']])

rf.fit(X2, y2)
predictions_2 = rf.predict(df[['Age', 'Breathing Style', 'Weapon Type']])

rf.fit(X3, y3)
predictions_3 = rf.predict(df[['Age', 'Breathing Style', 'Weapon Type']])

# Combine the predictions from the three bootstrapped models
aggregated_predictions = np.round((predictions_1 + predictions_2 + predictions_3) / 3).astype(int)

# Add aggregated predictions to the original DataFrame
df['Predicted'] = aggregated_predictions

# Display the DataFrame with predictions
print(df[['Character Name', 'Demon Slayer', 'Predicted']])
