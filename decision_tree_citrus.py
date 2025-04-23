import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib

df = pd.read_csv('citrus.csv')  

def show_data_overview(df):
    print("-- Data Head --")
    print(df.head())
    print("\n-- Data Info --")
    print(df.info())
    print("\n-- Descriptive Statistics --")
    print(df.describe())
    print("\n-- Class Distribution --")
    print(df['name'].value_counts())
    print("\n-- Missing Values --")
    print(df.isnull().sum())

show_data_overview(df)

df[['diameter','weight','red','green','blue']].hist(figsize=(10,8))
plt.tight_layout()
plt.show()

colors = {'orange': 'orange', 'grapefruit': 'purple'}  # pastikan key sesuai label
groups = df['name'].unique()
plt.figure(figsize=(6,6))
for grp in groups:
    mask = df['name'] == grp
    plt.scatter(df.loc[mask, 'diameter'], df.loc[mask, 'weight'],
                label=grp, color=colors.get(grp, 'gray'), alpha=0.6)
plt.xlabel('Diameter')
plt.ylabel('Weight')
plt.title('Scatter Plot: Diameter vs Weight')
plt.legend()
plt.show()

le = LabelEncoder()
df['label'] = le.fit_transform(df['name'])
X = df[['diameter','weight','red','green','blue']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
dt = DecisionTreeClassifier(random_state=42)
grid = GridSearchCV(dt, param_grid, cv=5, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)
best = grid.best_estimator_
print("\nBest Params:", grid.best_params_)

y_pred = best.predict(X_test)
y_proba = best.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

auc = roc_auc_score(y_test, y_proba)
print(f"\nROC AUC Score: {auc:.4f}")

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], '--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

print("\nFeature Importances:")
for name, score in zip(X.columns, best.feature_importances_):
    print(f"{name}: {score:.4f}")

plt.figure(figsize=(12, 8))
plot_tree(
    best,
    feature_names=X.columns,
    class_names=le.classes_,
    filled=True,
    fontsize=8
)
plt.show()

joblib.dump(best, 'decision_tree_citrus_model.pkl')
print("Model tersimpan di 'decision_tree_citrus_model.pkl'")
