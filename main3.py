from PIL import Image
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np

# Load the digits dataset
digits = load_digits()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2)

# Train an SVM classifier on the training data
clf = SVC()
clf.fit(X_train, y_train)

# Load the image file
img = Image.open("C:/Users/elvis/OneDrive/Desktop/Handwriting/images/handwritten_digit.png").convert("L")

# Resize the image to 8x8 pixels
img = img.resize((8, 8))

# Convert the image to a numpy array
image = np.array(img)

# Flatten the image into a 64-element array
image = image.reshape((64,))

# Predict the digit using the classifier
prediction = clf.predict([image])[0]

# Print the predicted digit
print("Predicted digit:", prediction)

# Test the classifier on the testing data and print the accuracy
accuracy = clf.score(X_test, y_test)
print("Accuracy:", accuracy)
