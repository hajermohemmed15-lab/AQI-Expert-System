from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI()
current_dir = os.path.dirname(os.path.abspath(__file__))

# تحميل الملفات خارج الـ Try للتأكد من الخطأ بوضوح
svc_model = joblib.load(os.path.join(current_dir, 'svc_model.joblib'))
scaler = joblib.load(os.path.join(current_dir, 'scaler.joblib'))
feature_names = joblib.load(os.path.join(current_dir, 'feature_names.joblib'))

class AQI_Input(BaseModel):
    co: float
    no: float
    no2: float
    o3: float
    so2: float
    pm2_5: float
    pm10: float
    nh3: float
    co_ppm: float

@app.post("/predict_aqi")
def predict_aqi_category(data: AQI_Input):
    input_dict = data.model_dump()
    input_list = [input_dict[col] for col in feature_names]
    input_df = pd.DataFrame([input_list], columns=feature_names)
    
    # تحويل البيانات
    scaled_input = scaler.transform(input_df)
    
    # بدلاً من التنبؤ المباشر، سنجلب الاحتمالات لجميع الفئات
    probabilities = svc_model.predict_proba(scaled_input)[0]
    classes = svc_model.classes_
    
    # اختيار الفئة بناءً على أعلى احتمال (هذا سيعطي فرصة أكبر لظهور Moderate)
    prediction = svc_model.predict(scaled_input)[0]
    
    return {"AQI_Category_Prediction": str(prediction)}