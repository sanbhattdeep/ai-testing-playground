import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

X = np.random.rand(100, 2)
y = np.random.randint(0, 2, 100)

num_runs = 5

accuracy_measures = []
f1_scores = []

for run in range(num_runs):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=run
    )
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    accuracy_measures.append(accuracy)
    f1_scores.append(f1)

avg_accuracy = np.mean(accuracy_measures)
avg_f1 = np.mean(f1_scores)

print("Accuracy Measure (average):", avg_accuracy)
print("F1 Score (average):", avg_f1)