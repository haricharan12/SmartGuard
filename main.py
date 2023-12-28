from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os

# Expanded dataset with more diverse examples
emails = [
    "Dear user, your account has been compromised, please reset your password",
    "You have won a lottery! Click here to claim",
    "Please confirm your email address for account update",
    "This is to inform you of a scheduled server maintenance",
    "Urgent: Your account will be closed unless action is taken",
    "Meeting scheduled at 3 PM today",
    "Your package delivery status",
    "Alert: Unusual sign-in activity detected on your account",
    "Congratulations! You've been selected for a prize",
    "Important: Verify your account details immediately",
    "Notice of policy change from your bank",
    "Free vacation tickets for exclusive customers",
    "Security alert: New login from unknown device",
    "Team meeting agenda and notes",
    "Invoice for your recent purchase",
    "Update your payment information"
]
labels = [1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]  # 1 for phishing, 0 for legitimate

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(emails, labels, test_size=0.2, random_state=42)

# Feature extraction using Bag-of-Words model
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Training a Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(X_train_counts, y_train)

# Predicting and evaluating the model
y_pred = model.predict(X_test_counts)
report = classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing'])

print(report)


# Expanded dataset
emails = [
    "Dear user, your account has been compromised, please reset your password",
    "You have won a lottery! Click here to claim",
    "Please confirm your email address for account update",
    "This is to inform you of a scheduled server maintenance",
    "Urgent: Your account will be closed unless action is taken",
    "Meeting scheduled at 3 PM today",
    "Your package delivery status",
    "Alert: Unusual sign-in activity detected on your account",
    "Congratulations! You've been selected for a prize",
    "Important: Verify your account details immediately",
    "Notice of policy change from your bank",
    "Free vacation tickets for exclusive customers",
    "Security alert: New login from unknown device",
    "Team meeting agenda and notes",
    "Invoice for your recent purchase",
    "Update your payment information"
]
labels = [1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]  # 1 for phishing, 0 for legitimate

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(emails, labels, test_size=0.2, random_state=42)

# Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Training a Multinomial Naive Bayes model
model_tfidf = MultinomialNB()
model_tfidf.fit(X_train_tfidf, y_train)

# Predicting and evaluating the model
y_pred_tfidf = model_tfidf.predict(X_test_tfidf)
report_tfidf = classification_report(y_test, y_pred_tfidf, target_names=['Legitimate', 'Phishing'])

print(report_tfidf)

# Simulated dataset with more variety
emails = [
    "Urgent: Confirm your account details to avoid suspension",
    "Exclusive offer just for you, claim your prize now",
    "Reminder: Your subscription is about to expire",
    "Schedule for the upcoming project meeting",
    "Security notice: New device logged into your account",
    "Your invoice for recent cloud services purchase",
    "Action required: Unusual activity detected in your account",
    "Happy holidays: Special discount on your next purchase",
    "Critical security patch for your operating system",
    "Your order has been shipped, track it online",
    "Suspicious sign-in attempt, was this you?",
    "Updated privacy policy and terms of service",
    "Win a free trip to Hawaii, enter the contest now",
    "Your feedback is important, complete this survey",
    "Payment successfully processed for your recent order",
    "Warning: Potential malware detected in your system"
]
labels = [1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1]  # 1 for phishing, 0 for legitimate

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(emails, labels, test_size=0.2, random_state=42)

# Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Training a Support Vector Machine (SVM) model
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train_tfidf, y_train)

# Predicting and evaluating the model
y_pred_svm = svm_model.predict(X_test_tfidf)
report_svm = classification_report(y_test, y_pred_svm, target_names=['Legitimate', 'Phishing'])

print(report_svm)

# Accuracies for each model
models = ["Logistic Regression\n(Bag-of-Words)", "Multinomial Naive Bayes\n(TF-IDF)", "SVM\n(TF-IDF)"]
accuracies = [0.50, 0.50, 0.75]  # Accuracies for each model

# Creating the line graph
plt.figure(figsize=(10, 6))
plt.plot(models, accuracies, marker='o', color='b', linestyle='-', linewidth=2, markersize=8)
plt.xlabel('Model and Feature Extraction Technique')
plt.ylabel('Accuracy')
plt.title('Comparison of Phishing Detection Model Accuracies Over Time')
plt.ylim([0, 1])  # Setting y-axis limit from 0 to 1 (0% to 100%)
plt.grid(True)

# Show the graph
print(plt.show())

def classify_email_from_file(file_path, vectorizer, model):
    """
    Reads an email from a text file, preprocesses it, and classifies it as phishing or legitimate.

    Parameters:
    - file_path: Path to the text file containing the email.
    - vectorizer: The TF-IDF vectorizer used to transform the text data.
    - model: The trained SVM model for classification.

    Returns:
    - Classification result as a string.
    """

    # Check if the file exists
    if not os.path.exists(file_path):
        return "Error: File does not exist."

    # Read the content of the file
    with open(file_path, 'r') as file:
        email_content = file.read()

    # Transform the email content using the trained TF-IDF vectorizer
    email_tfidf = vectorizer.transform([email_content])

    # Predict using the trained model
    prediction = model.predict(email_tfidf)

    # Return the result
    return "Phishing Email" if prediction[0] == 1 else "Legitimate Email"

file_path = "/content/sample_email.txt"

# Call the function with the file path, TF-IDF vectorizer, and SVM model
result = classify_email_from_file(file_path, tfidf_vectorizer, svm_model)
print(result)
