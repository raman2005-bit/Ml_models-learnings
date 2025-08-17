from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Data
data = [
    ("ham", "Hey, how are you doing today?"),
    ("ham", "Don't forget our meeting at 5pm."),
    ("ham", "Can you call me when you are free?"),
    ("ham", "Let's have lunch tomorrow."),
    ("ham", "I'm going to the gym now."),
    ("spam", "Congratulations! You've won a $1000 gift card. Click here."),
    ("spam", "You have been selected for a free vacation. Reply YES to claim."),
    ("spam", "URGENT! Your account has been compromised. Send details now."),
    ("spam", "Win a free iPhone, just visit this link."),
    ("spam", "Get cheap loans at 0% interest. Apply now!"),
]

# Split messages & labels
texts = [msg for label, msg in data]     # sirf messages
labels = [1 if label == "spam" else 0 for label, msg in data]  # spam = 1, ham = 0

# CountVectorizer
v = CountVectorizer()
X = v.fit_transform(texts)   # Text → Numbers

# Train model
model = MultinomialNB()
model.fit(X, labels)

# User input
user = input("Enter your SMS: ").lower()
uv = v.transform([user])  # Convert user msg

prediction = model.predict(uv)

if prediction[0] == 1:
    print("SMS is SPAM 🚨")
else:
    print("SMS is NOT spam ✅")
