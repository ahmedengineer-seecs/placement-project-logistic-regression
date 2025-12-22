# 🎓 End-to-End Placement Prediction

## 📌 Overview
This project predicts whether a student will get a placement based on academic performance and other features.  
It demonstrates a complete **end-to-end ML pipeline** including:
- Data preprocessing
- Model training
- Evaluation
- Saving & loading model (`model.pkl`)
-  Deployment with Flask

---

## 📂 Project Structure
```
End-to-End-Placement-Prediction/
│── data/placement.csv
│── notebooks/end_to_end_ml.ipynb
│── models/model.pkl
│── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── predict.py
│── app.py
│── requirements.txt
│── README.md
│── .gitignore
```

---

## ⚙️ Installation
```bash
git clone https://github.com/yourusername/End-to-End-Placement-Prediction.git
cd End-to-End-Placement-Prediction
pip install -r requirements.txt
```

---

## ▶️ Usage
Train the model:
```bash
python src/train_model.py
```

Make predictions:
```bash
python src/predict.py
```

Run Flask API:
```bash
python app.py
```

---

## 📊 Results
- Model: Logistic Regression  
- Accuracy: ~88% (from notebook results)  

---

## 🌐 Deployment
Run `python app.py` → access at `http://127.0.0.1:5000`  
Send POST request:  
```json
{"features": [70, 80, 8.5]}
```

---

## ✨ Author
Ahmed Hussain  
