"""
Titanic Masterclass - Script 02: Feature Engineering & Preprocessing
====================================================================
สคริปต์นี้สาธิตเทคนิคการสร้างฟีเจอร์ใหม่ (Feature Engineering)
และการทำความสะอาดข้อมูลที่ขาดหายไปอย่างชาญฉลาด (Smart Imputation)
"""

import pandas as pd
import numpy as np

def prepare_features(df, is_train=True):
    # ทำสำเนาข้อมูลเพื่อป้องกัน SettingWithCopyWarning
    data = df.copy()

    # 1. Feature Extraction: สกัด Title (คำนำหน้าชื่อ) จาก Name
    data['Title'] = data['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
    data['Title'] = data['Title'].replace(rare_titles, 'Rare')
    data['Title'] = data['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})

    # 2. Smart Imputation: เติม Age ตามมัธยฐานของแต่ละ Title
    # สังเกตว่า 'Master' คือเด็กชาย (มัธยฐาน 3.5 ขวบ) ไม่ควรเติม 30 ปีเหมือน 'Mr'!
    title_age_median = {'Master': 3.5, 'Miss': 21.0, 'Mr': 30.0, 'Mrs': 35.0, 'Rare': 48.5}
    for title, med_age in title_age_median.items():
        data.loc[(data['Age'].isnull()) & (data['Title'] == title), 'Age'] = med_age
    # เผื่อกรณีหลุด
    data['Age'] = data['Age'].fillna(data['Age'].median())

    # 3. เติม Embarked (Missing ใน train 2 คน) ด้วยฐานนิยม (Mode) คือ 'S'
    data['Embarked'] = data['Embarked'].fillna('S')

    # 4. เติม Fare (Missing ใน test 1 คน) ด้วยมัธยฐานของชั้น 3
    data['Fare'] = data['Fare'].fillna(data[data['Pclass'] == 3]['Fare'].median())

    # 5. Feature Creation: สร้าง FamilySize และ IsAlone
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int)

    # 6. Feature Creation: สกัด Deck จากตัวอักษรแรกของ Cabin (M = Missing)
    data['Deck'] = data['Cabin'].apply(lambda x: str(x)[0] if pd.notna(x) else 'M')

    # 7. Log Transform Fare เพื่อลด Skewness (เบ้ขวาจัด)
    data['LogFare'] = np.log1p(data['Fare'])

    # 8. Categorical Encoding (แปลงข้อความเป็นตัวเลข)
    # Sex: male=0, female=1
    data['Sex_Code'] = data['Sex'].map({'male': 0, 'female': 1}).astype(int)

    # One-Hot Encoding สำหรับ Embarked และ Title
    data = pd.get_dummies(data, columns=['Embarked', 'Title'], drop_first=True, dtype=int)

    # เลือกเฉพาะฟีเจอร์ที่เป็นตัวเลขพร้อมเข้าโมเดล
    feature_cols = [
        'Pclass', 'Sex_Code', 'Age', 'SibSp', 'Parch',
        'LogFare', 'FamilySize', 'IsAlone'
    ] + [c for c in data.columns if c.startswith(('Embarked_', 'Title_'))]

    if is_train:
        return data[feature_cols], data['Survived']
    else:
        return data[feature_cols], data['PassengerId']

if __name__ == "__main__":
    train_df = pd.read_csv("data/train.csv")
    X_train, y_train = prepare_features(train_df, is_train=True)
    print("=" * 60)
    print("✨ FEATURE ENGINEERING COMPLETED")
    print("=" * 60)
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print("\nรายชื่อ Features ทั้งหมด:")
    for i, col in enumerate(X_train.columns, 1):
        print(f"  {i:2d}. {col}")
    print("\nตัวอย่าง 3 แถวแรกของ Features ที่เตรียมเสร็จ:")
    print(X_train.head(3))
