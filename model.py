import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

crops = pd.read_csv("soil_measures.csv")

# "N": Nitrogen content ratio in the soil
# "P": Phosphorous content ratio in the soil
# "K": Potassium content ratio in the soil
# "pH" value of the soil
# "crop": categorical values that contain various crops (target variable)

print(crops.head())

X = crops.drop('crop', axis=1)
y = crops['crop']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_predicted = logreg.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_predicted)

print(f'Accuracy = {accuracy}')

coefficients = abs(logreg.coef_[0])
print(f'Features coefficients: {coefficients}')

best_index = list(coefficients).index(max(coefficients))
best_predictive_feature = {X.columns[best_index]: coefficients[best_index]}
print(f'Best predictive feature: {best_predictive_feature}')

X_best_feature = X_scaled[:, best_index].reshape(-1, 1)

(X_train_minimized, X_test_minimized,
 y_train_minimized, y_test_minimized) = train_test_split(X_best_feature, y, test_size=0.2, random_state=42)

logreg_minimized = LogisticRegression()
logreg_minimized.fit(X_train_minimized, y_train_minimized)

y_predicted_minimized = logreg_minimized.predict(X_test_minimized)

accuracy_minimized = metrics.accuracy_score(y_test_minimized, y_predicted_minimized)

print(f'Minimized accuracy = {accuracy_minimized}')

coefficients_minimized = abs(logreg_minimized.coef_[0])
print(f'Features coefficients: {coefficients_minimized}')
