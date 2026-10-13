"""
Titanic Masterclass - Script 03: Machine Learning Modeling & Submission
=======================================================================
สคริปต์นี้สาธิตกระบวนการสร้างและประเมินโมเดล Machine Learning อย่างเป็นระบบ:
1. Baseline Model (Heuristic: ผู้หญิงรอดทุกคน)
2. Logistic Regression (Linear Model + Scaling)
3. Random Forest Classifier (Tree Ensemble)
4. Stratified 5-Fold Cross Validation
5. การอ่าน Classification Report & Feature Importance
6. การ Export ผลลัพธ์เป็น submission.csv
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

import importlib.util
import os

# โหลดฟังก์ชันเตรียมข้อมูลจาก 02_feature_engineering.py
spec = importlib.util.spec_from_file_location("fe", "02_feature_engineering.py")
fe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fe)

def main():
    print("=" * 65)
    print("🤖 TITANIC MACHINE LEARNING & EVALUATION MASTERCLASS")
    print("=" * 65)

    # 1. โหลดข้อมูล
    train_raw = pd.read_csv("data/train.csv")
    test_raw = pd.read_csv("data/test.csv")

    X_train, y_train = fe.prepare_features(train_raw, is_train=True)
    X_test, passenger_ids = fe.prepare_features(test_raw, is_train=False)

    # ตรวจสอบว่าคอลัมน์ใน Test ตรงกับ Train ครบถ้วน
    missing_cols = set(X_train.columns) - set(X_test.columns)
    for c in missing_cols:
        X_test[c] = 0
    X_test = X_test[X_train.columns]

    print(f"\n[1] ขนาดข้อมูลหลัง Feature Engineering:")
    print(f"    - Training Features: {X_train.shape}")
    print(f"    - Test Features:     {X_test.shape}")

    # 2. Heuristic Baseline: "ผู้หญิงทุกคนรอด ผู้ชายทุกคนเสียชีวิต"
    baseline_pred = (train_raw['Sex'] == 'female').astype(int)
    baseline_acc = (baseline_pred == y_train).mean() * 100
    print(f"\n[2] Heuristic Baseline (ทายว่าผู้หญิงรอดทุกคน):")
    print(f"    - Accuracy: {baseline_acc:.2f}% (โมเดล ML เราต้องชนะตัวเลขนี้!)")

    # 3. Stratified 5-Fold Cross-Validation Setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 4. โมเดลที่ 1: Logistic Regression
    lr_pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    lr_scores = cross_val_score(lr_pipe, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"\n[3] Model 1: Logistic Regression")
    print(f"    - 5-Fold CV Accuracy: {lr_scores.mean() * 100:.2f}% (± {lr_scores.std() * 100:.2f}%)")

    # 5. โมเดลที่ 2: Random Forest
    rf_model = RandomForestClassifier(n_estimators=150, max_depth=5, min_samples_split=4, random_state=42)
    rf_scores = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"\n[4] Model 2: Random Forest (Depth=5)")
    print(f"    - 5-Fold CV Accuracy: {rf_scores.mean() * 100:.2f}% (± {rf_scores.std() * 100:.2f}%)")

    # 6. เจาะลึกผลการประเมิน Random Forest (Confusion Matrix & Classification Report)
    y_pred_cv = cross_val_predict(rf_model, X_train, y_train, cv=cv)
    cm = confusion_matrix(y_train, y_pred_cv)
    
    print("\n[5] Confusion Matrix (Random Forest):")
    print(f"               ทำนายว่า 0 (ตาย)    ทำนายว่า 1 (รอด)")
    print(f"    ของจริง 0 :     {cm[0,0]:<15}      {cm[0,1]} (False Positive)")
    print(f"    ของจริง 1 :     {cm[1,0]:<15}      {cm[1,1]} (True Positive)")

    print("\n[6] Classification Report:")
    print(classification_report(y_train, y_pred_cv, target_names=['Died (0)', 'Survived (1)']))

    # 7. เทรนโมเดลบน Train ทั้งหมด และวิเคราะห์ Feature Importance
    rf_model.fit(X_train, y_train)
    importances = pd.Series(rf_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print("\n[7] Top 7 ฟีเจอร์ที่ส่งผลต่อการทำนายมากที่สุด (Feature Importance):")
    for feat, imp in importances.head(7).items():
        print(f"    - {feat:<15}: {imp * 100:.1f}%")

    # 8. ทำนายชุด Test และสร้างไฟล์ Submission
    test_preds = rf_model.predict(X_test)
    sub = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': test_preds})
    sub.to_csv("submission.csv", index=False)
    print(f"\n[8] Kaggle Submission File พร้อมแล้ว!")
    print(f"    - บันทึกไว้ที่: {os.path.abspath('submission.csv')}")
    print(f"    - จำนวนแถวทำนาย: {len(sub)} (ต้องเท่ากับ 418)")
    print(f"    - สัดส่วนที่ทายว่ารอด: {(sub['Survived'] == 1).mean() * 100:.2f}%")

if __name__ == "__main__":
    main()
