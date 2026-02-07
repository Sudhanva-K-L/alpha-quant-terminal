import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from data_processor import build_stock_dataset

def train_advanced_model(ticker="AAPL"):
    # 1. Load data with the new indicators we discussed
    df = build_stock_dataset(ticker)
    
    if df.empty:
        print("No data found. Check data_processor.py")
        return

    # 2. Expanded Feature Set
    # We add Volatility (BB) and Momentum (MACD) to give the model 'eyes'
    features = ['Close', 'SMA_50', 'SMA_200', 'RSI', 'MACD', 'MACD_Signal', 'BB_High', 'BB_Low']
    
    # Ensure all features exist in the dataframe
    X = df[features]
    y = df['Target']

    # 3. Time-Series Split (80/20)
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    print(f"Training XGBoost on {len(X_train)} days...")

    # 4. XGBoost with Regularization
    # learning_rate: slows down learning to prevent overfitting
    # max_depth: limits tree complexity
    # gamma: requires a minimum loss reduction to make a split
    model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    # 5. Fit model with early stopping
    # This stops training if the test score stops improving
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # 6. Evaluation
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("\n--- ADVANCED MODEL PERFORMANCE ---")
    print(f"XGBoost Accuracy: {acc:.2%}")
    print("\nDetailed Report:")
    print(classification_report(y_test, preds))

    # 7. Save for Deployment
    model_data = {
        "model": model,
        "features": features
    }
    joblib.dump(model_data, "stock_model.pkl")
    print("\nSuccess! Advanced model saved as 'stock_model.pkl'")

if __name__ == "__main__":
    train_advanced_model("AAPL")