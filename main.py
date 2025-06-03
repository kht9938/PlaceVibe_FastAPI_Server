from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi import Request
import os
import shutil
import requests
import torch
import numpy as np
import librosa
import wave


# 경로 및 설정
UPLOAD_DIR = "uploads"
MODEL_PATH = "model.pth"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Kakao API Key
KAKAO_REST_API_KEY = os.getenv("KAKAO_REST_API_KEY")

# 클래스 목록 (tram 제외)
TAU_CLASSES = [
    'airport', 'bus', 'metro', 'metro_station', 'park',
    'public_square', 'shopping_mall', 'street_pedestrian',
    'street_traffic'
]
label_map = {label: idx for idx, label in enumerate(TAU_CLASSES)}
inv_label_map = {v: k for k, v in label_map.items()}

# 최종 분류
CATEGORY_GROUPS = {
    "교통": [
        "airport", "bus", "metro", "metro_station", "지하철역", "주차장", "주유소", "공항"
    ],
    "상점": [
        "shopping_mall", "대형마트", "편의점", "중개업소", "은행"
    ],
    "음식": [
        "음식점", "카페"
    ],
    "여가": [
        "park", "public_square", "문화시설", "관광명소", "숙박"
    ],
    "교육/공공기관": [
        "학교", "유치원", "학원", "공공기관"
    ],
    "의료": [
        "병원", "약국"
    ],
    "거리": [
        "street_pedestrian", "street_traffic"
    ]
}

def remap_place(place: str) -> str:
    for group, items in CATEGORY_GROUPS.items():
        if place in items:
            print(f"[DEBUG] 리맵핑: '{place}' → '{group}'")  # 디버깅 로그
            return group
    print(f"[DEBUG] 리맵핑 실패: '{place}' → '기타'")
    return "기타"


# 모델 정의 (SimpleCNN)
class SimpleCNN(torch.nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# 모델 로딩
model = SimpleCNN(num_classes=len(label_map))
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()


# 오디오 → Mel-spectrogram 전처리
def preprocess(filepath, max_len=173):
    y, sr = librosa.load(filepath, sr=22050)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = mel_db[:, :max_len]
    if mel_db.shape[1] < max_len:
        pad_width = max_len - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
    mel_db = mel_db[np.newaxis, np.newaxis, :, :]  # [1, 1, 128, max_len]
    return torch.tensor(mel_db, dtype=torch.float32)


# 예측 함수
def predict(filepath):
    x = preprocess(filepath)
    with torch.no_grad():
        output = model(x)
        pred = output.argmax(dim=1).item()
        return inv_label_map[pred]


# Kakao API 호출 함수 통합
def search_all_categories(lat, lon):
    url = "https://dapi.kakao.com/v2/local/search/category.json"
    headers = {
        "Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"
    }

    category_codes = {
        "MT1": "대형마트", "CS2": "편의점", "PS3": "유치원", "SC4": "학교",
        "AC5": "학원", "PK6": "주차장", "OL7": "주유소", "SW8": "지하철역",
        "BK9": "은행", "CT1": "문화시설", "AG2": "중개업소", "PO3": "공공기관",
        "AT4": "관광명소", "AD5": "숙박", "FD6": "음식점", "CE7": "카페",
        "HP8": "병원", "PM9": "약국"
    }

    nearest = None

    for code, name in category_codes.items():
        params = {
            "category_group_code": code,
            "x": lon,
            "y": lat,
            "radius": 40,
            "size": 1
        }

        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            docs = response.json().get("documents", [])
            if docs:
                doc = docs[0]
                distance = int(doc["distance"])
                if nearest is None or distance < nearest["distance"]:
                    nearest = {"category": name, "distance": distance}

    if nearest:
        print(f"가장 가까운 카테고리: {nearest['category']} ({nearest['distance']}m 이내)")
        return nearest["category"]
    else:
        print("반경 내에 장소 없음")
        return None


# FastAPI 시작
app = FastAPI()


# 1. 오디오 기반 예측
@app.post("/predict_place_audio")
async def predict_place_audio(audio_file: UploadFile = File(...)):
    # PCM 파일 저장
    pcm_path = os.path.join(UPLOAD_DIR, audio_file.filename)
    with open(pcm_path, "wb") as f:
        shutil.copyfileobj(audio_file.file, f)

    # PCM → WAV 변환 (필수 파라미터: 채널 수, 샘플폭, 샘플레이트)
    wav_path = pcm_path.replace(".pcm", ".wav")
    with wave.open(wav_path, 'wb') as wf:
        wf.setnchannels(1)               # mono
        wf.setsampwidth(2)               # 16-bit = 2 bytes
        wf.setframerate(16000)           # sample rate
        with open(pcm_path, 'rb') as pf:
            wf.writeframes(pf.read())

    # 원래 예측
    raw_place = predict(wav_path)

    # 리맵핑 적용
    place = remap_place(raw_place)

    # 결과 반환
    return JSONResponse({"place": place})


# 좌표 기반 장소 예측
@app.post("/predict_place_kakao")
async def predict_place_kakao(
    request: Request,
    lat: float = Form(...),
    lng: float = Form(...)
):
    category = search_all_categories(lat, lng)

    if category is None:
        print("카카오 장소 없음 → 오디오로 대체")

        # 같은 요청에 파일 포함돼 있을 경우를 위해 처리
        form = await request.form()
        audio_file = form.get("audio_file")

        if audio_file:
            save_path = os.path.join(UPLOAD_DIR, audio_file.filename)
            with open(save_path, "wb") as f:
                shutil.copyfileobj(audio_file.file, f)

            # PCM → WAV
            wav_path = save_path.replace(".pcm", ".wav")
            with wave.open(wav_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                with open(save_path, 'rb') as pf:
                    wf.writeframes(pf.read())

            raw_place = predict(wav_path)
            category = remap_place(raw_place)
    else:
        category = remap_place(category)

    return JSONResponse({"place": category})