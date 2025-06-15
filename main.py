from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 또는 ["http://localhost:3000"]로 제한 가능
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# YOLO 모델 로드 (네가 직접 학습한 모델을 사용할 때 경로를 변경하면 돼)
model = YOLO("best.pt")  # 학습시킨 모델 적용.
model2 = YOLO("best3.pt")

# 클래스 이름 정의 (모델 학습 시 사용한 클래스 순서와 동일해야 함)
class_names = ["궤양병", "잎곰팡이병", "흰가루병", "아메리카잎굴파리", "토마토모자이크병", "녹응애"]  
#토마토모자이크병, 점박이응애
class_names2 = ["FR", "IM", "RT"]

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    
    # 이미지 파일 읽기
    contents = await file.read()
    image = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # YOLO 모델 예측 (가장 신뢰도가 높은 객체 1개만 감지)
    result = model(image, conf=0.5, iou=0.5, max_det=1)

    # 결과가 없는 경우 (질병이 식별되지 않은 경우) 예외 처리
    if not result[0].boxes:  
        return {"message": "정상 토마토입니다."}

    # 가장 확신이 높은 객체 가져오기
    box = result[0].boxes[0]  
    best_prediction = {
        "result": class_names[int(box.cls)],  # 클래스 ID를 이름으로 변환
        "confidence": float(box.conf),  # 신뢰도 (0~1 사이)
        "bbox": [float(x) for x in box.xyxy[0]]  # 바운딩 박스 좌표
    }

    return {"best_prediction": best_prediction} # json 형태로 반환.
@app.post("/quality/")
async def predict(file: UploadFile = File(...)):
    
    # 이미지 파일 읽기
    contents = await file.read()
    image = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # YOLO 모델 예측 (가장 신뢰도가 높은 객체 1개만 감지)
    result = model2(image, conf=0.5, iou=0.5, max_det=1)

    # 결과가 없는 경우 (질병이 식별되지 않은 경우) 예외 처리
    if not result[0].boxes:  
        return {"message": "정상 토마토입니다."}

    # 가장 확신이 높은 객체 가져오기
    box = result[0].boxes[0]  
    best_prediction = {
        "result": class_names2[int(box.cls)],  # 클래스 ID를 이름으로 변환
        "confidence": float(box.conf),  # 신뢰도 (0~1 사이)
        "bbox": [float(x) for x in box.xyxy[0]]  # 바운딩 박스 좌표
    }

    return {"best_prediction": best_prediction} # json 형태로 반환.