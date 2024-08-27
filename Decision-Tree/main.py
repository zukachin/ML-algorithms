import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

# Original data
data = [
    [1, "Sunny", "Hot", "N"],
    [2, "Sunny", "Hot", "N"],
    [3, "Overcast", "Hot", "Y"],
    [4, "Rain", "Mild", "Y"],
    [5, "Rain", "Cool", "Y"],
    [6, "Rain", "Cool", "N"],
    [7, "Overcast", "Cool", "Y"],
    [8, "Sunny", "Mild", "N"],
    [9, "Sunny", "Cool", "Y"],
    [10, "Rain", "Mild", "Y"],
    [11, "Sunny", "Mild", "Y"],
    [12, "Overcast", "Mild", "Y"],
    [13, "Overcast", "Hot", "N"],
    [14, "Rain", "Mild", "N"]
]

# Convert data to a NumPy array
data_np = np.array(data)

# Extract columns for outlook, temperature, and play
outlook = data_np[:, 1]
temperature = data_np[:, 2]
play = data_np[:, 3]

# Encode categorical data
label_encoder_outlook = LabelEncoder()
label_encoder_temperature = LabelEncoder()
label_encoder_play = LabelEncoder()

# Transform categorical values to numerical values
outlook_encoded = label_encoder_outlook.fit_transform(outlook)
temperature_encoded = label_encoder_temperature.fit_transform(temperature)
play_encoded = label_encoder_play.fit_transform(play)

# Combine the encoded columns into a single dataset for training
X = np.column_stack((outlook_encoded, temperature_encoded))
y = play_encoded

# Create and train the decision tree classifier
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X, y)

# Plot the decision tree
tree.plot_tree(clf)
