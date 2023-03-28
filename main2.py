from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load the digits dataset
digits = load_digits()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2)

# Train an SVM classifier on the training data
clf = SVC()
clf.fit(X_train, y_train)

# Take input from the user
user_input = input("Enter a handwritten digit image (as a 64-element list of integers, separated by spaces): ")
image = [int(x) for x in user_input.split()]

# Predict the digit using the classifier
prediction = clf.predict([image])[0]

# Print the predicted digit
print("Predicted digit:", prediction)

# Test the classifier on the testing data and print the accuracy
accuracy = clf.score(X_test, y_test)
print("Accuracy:", accuracy)
