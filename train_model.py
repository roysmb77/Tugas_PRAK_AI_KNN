# train_model.py
import joblib
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import os

# Load dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model KNN
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Evaluasi akurasi
accuracy = model.score(X_test, y_test)

# Simpan model, scaler, dan akurasi
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/knn_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

# Simpan akurasi ke file
with open('models/accuracy.txt', 'w') as f:
    f.write(str(round(accuracy * 100, 2)))  # Contoh: 96.67

print("Model dan scaler berhasil disimpan. Akurasi:", accuracy)

