import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train(n_estimators, max_depth):
    # 1. Load dataset hasil preprocessing laptop
    df = pd.read_csv('laptop_sales_preprocessed.csv')
    
    # 2. Pembersihan Akhir (PENTING): Menghapus kolom teks yang menyebabkan Error
    # Kita hapus kolom yang tidak bisa dikonversi ke float secara otomatis
    cols_to_drop = ['price_category', 'laptop_ID', 'Product', 'ScreenResolution', 'Memory']
    
    # Menghapus kolom hanya jika kolom tersebut ada di dataframe
    X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    y = df['price_category']
    
    # 3. Split Dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run():
        # 4. Inisialisasi dan Training Model
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # 5. Evaluasi
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        # 6. Logging secara manual untuk syarat poin Advance
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)
        
        # Log model ke MLflow
        mlflow.sklearn.log_model(model, "model")
        
        print(f"✅ Success! Model trained with Accuracy: {acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()
    train(args.n_estimators, args.max_depth)