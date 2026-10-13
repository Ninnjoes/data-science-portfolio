"""
Titanic Masterclass - Script 01: Exploratory Data Analysis (EDA)
================================================================
สคริปต์นี้ใช้สำหรับโหลดข้อมูล Titanic และทำการวิเคราะห์เชิงสำรวจอย่างละเอียด
"""

import pandas as pd
import numpy as np

def run_eda():
    print("=" * 60)
    print("🚢 TITANIC DATA EXPLORATION (EDA)")
    print("=" * 60)

    # 1. โหลดข้อมูล
    train_path = "data/train.csv"
    df = pd.read_csv(train_path)

    # 2. ดูขนาดและโครงสร้างพื้นฐาน
    print(f"\n[1] ข้อมูลมีทั้งหมด: {df.shape[0]} แถว, {df.shape[1]} คอลัมน์")
    print("\n--- ตัวอย่างข้อมูล 3 แถวแรก ---")
    print(df.head(3))

    # 3. ตรวจสอบ Missing Values (ข้อมูลที่หายไป)
    print("\n[2] สรุป Missing Values:")
    null_counts = df.isnull().sum()
    null_percent = (df.isnull().sum() / len(df)) * 100
    missing_table = pd.DataFrame({'Missing Count': null_counts, 'Percent (%)': null_percent})
    print(missing_table[missing_table['Missing Count'] > 0])

    # 4. อัตราการรอดชีวิตภาพรวม (Base Rate)
    overall_survival = df['Survived'].value_counts(normalize=True) * 100
    print(f"\n[3] อัตราการรอดชีวิตภาพรวม:")
    print(f"   - เสียชีวิต (0): {overall_survival[0]:.2f}%")
    print(f"   - รอดชีวิต   (1): {overall_survival[1]:.2f}%")

    # 5. วิเคราะห์แยกตามเพศ (Sex)
    print("\n[4] อัตราการรอดชีวิตแยกตามเพศ:")
    sex_analysis = df.groupby('Sex')['Survived'].agg(
        Total='count',
        Survived='sum',
        SurvivalRate=lambda x: x.mean() * 100
    )
    print(sex_analysis)

    # 6. วิเคราะห์แยกตามชั้นโดยสาร (Pclass)
    print("\n[5] อัตราการรอดชีวิตแยกตามชั้นตั๋วโดยสาร (Pclass):")
    pclass_analysis = df.groupby('Pclass')['Survived'].agg(
        Total='count',
        Survived='sum',
        SurvivalRate=lambda x: x.mean() * 100
    )
    print(pclass_analysis)

    # 7. วิเคราะห์ไขว้ 2 มิติ (Sex x Pclass) ด้วย Pivot Table
    print("\n[6] อัตราการรอดชีวิตไขว้ระหว่าง เพศ (Sex) x ชั้นตั๋ว (Pclass) (%):")
    pivot = df.pivot_table(
        index='Pclass',
        columns='Sex',
        values='Survived',
        aggfunc=lambda x: x.mean() * 100
    )
    print(pivot.round(2))

    # 8. วิเคราะห์เรื่องอายุและเด็ก (Age & Children)
    print("\n[7] เด็ก (อายุ < 10 ปี) รอดชีวิตกี่ % เทียบกับผู้ใหญ่?")
    df['IsChild'] = df['Age'] < 10
    child_analysis = df.groupby('IsChild')['Survived'].agg(
        Total='count',
        SurvivalRate=lambda x: x.mean() * 100
    )
    print(child_analysis)

if __name__ == "__main__":
    run_eda()
