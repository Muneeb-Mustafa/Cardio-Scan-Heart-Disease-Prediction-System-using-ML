import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


df = pd.read_csv('./data/heart.csv')
pickle.dump(df, open('./models/data.pkl', 'wb'))

X = df.drop('target', axis=1)
y = 1 - df['target']   

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

params = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
}

grid = GridSearchCV(RandomForestClassifier(random_state=42), params, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
accuracy = best_model.score(X_test, y_test)

print(f"Best params: {grid.best_params_}")
print(f"Model accuracy: {accuracy:.2%}")

pickle.dump(best_model, open('./models/model.pkl', 'wb'))
print("Model saved successfully to ./models/model.pkl")


y_prob = best_model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], '--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Model Performance (ROC Curve)")
plt.legend()

plt.show()
plt.savefig("performance.png")
