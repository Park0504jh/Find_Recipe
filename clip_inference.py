import torch
import open_clip
import requests
from PIL import Image
from io import BytesIO
import streamlit as st

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"
MODEL_PATH = "clip_finetuned.pt"

# 음식 후보 키워드 (Recipe1M+ 기반)
FOOD_CANDIDATES = [
    "pasta", "chicken", "salad", "soup", "pizza",
    "steak", "sushi", "burger", "sandwich", "cake",
    "cookie", "bread", "rice", "noodle", "fish",
    "shrimp", "beef", "pork", "tofu", "curry",
    "taco", "burrito", "pancake", "waffle", "muffin",
    "brownie", "pie", "cheesecake", "ice cream", "smoothie"
]

@st.cache_resource
def load_clip_model():
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED
    )
    # 학습된 가중치 로드
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )
    model = model.to(DEVICE)
    model.eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    return model, preprocess, tokenizer


def classify_food(image: Image.Image) -> str:
    """이미지에서 음식 이름을 분류하여 반환"""

    model, preprocess, tokenizer = load_clip_model()

    # 이미지 전처리
    image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    # 텍스트 후보 토크나이징
    texts = tokenizer(
        [f"a photo of {food}" for food in FOOD_CANDIDATES]
    ).to(DEVICE)

    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        text_features  = model.encode_text(texts)

        image_features = image_features / image_features.norm(
            dim=-1, keepdim=True
        )
        text_features  = text_features / text_features.norm(
            dim=-1, keepdim=True
        )

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    # 가장 높은 유사도의 음식
    best_idx   = similarity[0].argmax().item()
    best_food  = FOOD_CANDIDATES[best_idx]
    confidence = similarity[0][best_idx].item()

    return best_food, confidence