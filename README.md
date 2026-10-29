# Thai Customer Review Sentiment & Issue Classification for E-Commerce & CRM
**ระบบวิเคราะห์ความรู้สึกและจำแนกประเด็นปัญหาบริการลูกค้าจากข้อความรีวิวภาษาไทยด้วยโมเดล Deep Learning**

> **รายวิชา:** CS461 Final Project  
> **คณะผู้จัดทำ:**
> - นราทร ศุภสุวรณพงษ์ (1660704568)
> - วิษณุ เดวี (1660704048)
> - ณัฏฐกิตติ์ สุทธิเกษมคุณ (1660702562)
> - นฤเบศ จีนล้ำเลิศ (1660703750)

---

## 📌 ภาพรวมโครงสร้างโปรเจกต์ (Project Architecture)

โปรเจกต์นี้ได้รับการพัฒนาใหม่ทั้งหมด โดยใช้โครงสร้างแบบ Modular Clean Architecture และใช้ชุดข้อมูลจริง (**Wisesight Sentiment Corpus 26,737 ข้อความ**) โดยไม่มีการจำลองค่าสุ่ม (Mock data) ใดๆ:

```
thai-sentiment-analysis-main/
├── app.py                      # ระบบเว็บแอปพลิเคชัน Streamlit ฉบับสมบูรณ์
├── src/                        # ซอร์สโค้ดหลักของระบบ
│   ├── __init__.py
│   ├── preprocessor.py         # การทำความสะอาดข้อความและการตัดคำภาษาไทย (PyThaiNLP)
│   ├── model.py                # Pipeline การฝึกสอน การประเมินผล และ Inference
│   ├── issue_classifier.py     # ระบบจำแนกประเด็นปัญหา (CRM Issue Classification)
│   └── data_loader.py          # การโหลดและจัดการข้อมูล Wisesight
├── scripts/
│   ├── prepare_data.py         # สคริปต์แปลงไฟล์ข้อมูล Wisesight เป็น CSV
│   └── train.py                # สคริปต์ฝึกสอนโมเดล Baseline และบันทึกผลลัพธ์
├── data/                       # โฟลเดอร์จัดเก็บชุดข้อมูลจริง
│   ├── train.txt / train_label.txt
│   ├── test.txt / test_label.txt
│   ├── wisesight_train.csv     # ข้อมูลฝึกสอน 23,545 แถว
│   ├── wisesight_test.csv      # ข้อมูลทดสอบ 2,617 แถว
│   └── sample_customer_reviews.csv # ข้อมูลตัวอย่าง 100 ข้อความสำหรับทดสอบ Batch
├── models/                     # โมเดลและผลลัพธ์การประเมิน
│   ├── baseline_model.pkl      # โมเดล Baseline ที่เทรนเสร็จสมบูรณ์
│   └── evaluation_results.json # บันทึกสถิติและ Confusion Matrix
├── requirements.txt            # รายการไลบรารีที่จำเป็น
└── README.md
```

---

## 🚀 วิธีการติดตั้งและรันโปรแกรม

### 1. ติดตั้ง Dependencies
```bash
pip install -r requirements.txt
```

### 2. จัดเตรียมข้อมูลและฝึกสอนโมเดล Baseline
```bash
python scripts/prepare_data.py
python scripts/train.py
```

### 3. รันเว็บแอปพลิเคชัน Streamlit
```bash
streamlit run app.py
```
เปิดใช้งานผ่านเบราว์เซอร์ที่: **`http://localhost:8501`**

---

## 📊 ผลการทดลองโมเดล Baseline บน Wisesight Test Set (2,617 ตัวอย่าง)

- **Accuracy:** 68.82%
- **Macro F1-Score:** 0.6646
- **Weighted F1-Score:** 0.6937
- **Negative Class Recall:** 79.65% (~80%)

