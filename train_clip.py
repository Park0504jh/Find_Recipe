import os
import json
import torch
import open_clip
import requests
from PIL import Image
from io import BytesIO
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from tqdm import tqdm

# ------------------------
# 설정
# ------------------------
LAYER1_PATH = r"C:\Users\DS\Desktop\recipe1M_layers\layer1.json"
LAYER2_PATH = r"C:\Users\DS\Desktop\recipe1M_layers\layer2.json"
MODEL_NAME  = "ViT-B-32"
PRETRAINED  = "openai"
BATCH_SIZE  = 32
EPOCHS      = 5
LR          = 1e-5
MAX_SAMPLES = 10000   # 학습에 사용할 최대 샘플 수
SAVE_PATH   = "clip_finetuned.pt"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

print(f"사용 디바이스: {DEVICE}")

# ------------------------
# 데이터 준비
# ------------------------
def load_data(layer1_path, layer2_path, max_samples=MAX_SAMPLES):
    """layer1.json(텍스트)과 layer2.json(이미지 URL)을 합쳐서 반환"""

    print("layer2.json 로딩 중...")
    with open(layer2_path, "r", encoding="utf-8") as f:
        layer2 = json.load(f)

    # ID → 이미지 URL 딕셔너리
    image_dict = {}
    for item in layer2:
        images = item.get("images", [])
        if images:
            image_dict[item["id"]] = images[0]["url"]

    print("layer1.json 로딩 중...")
    pairs = []

    with open(layer1_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line in ["[", "]"]:
                continue
            if line.endswith(","):
                line = line[:-1]
            try:
                recipe = json.loads(line)
            except:
                continue

            recipe_id = recipe.get("id")
            title     = recipe.get("title", "")

            # 이미지 URL이 있는 레시피만 사용
            if recipe_id in image_dict and title:
                pairs.append({
                    "image_url": image_dict[recipe_id],
                    "text": title
                })

            if len(pairs) >= max_samples:
                break

    print(f"총 {len(pairs)}개 쌍 준비 완료")
    return pairs


# ------------------------
# Dataset
# ------------------------
class RecipeDataset(Dataset):

    def __init__(self, pairs, preprocess, tokenizer):
        self.pairs      = pairs
        self.preprocess = preprocess
        self.tokenizer  = tokenizer

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        # 이미지 URL에서 다운로드
        try:
            response = requests.get(
                pair["image_url"],
                timeout=5
            )
            image = Image.open(
                BytesIO(response.content)
            ).convert("RGB")
            image = self.preprocess(image)

        except:
            # 이미지 로드 실패 시 빈 이미지
            image = torch.zeros(3, 224, 224)

        # 텍스트 토크나이징
        text = self.tokenizer([pair["text"]])[0]

        return image, text


# ------------------------
# 학습
# ------------------------
def train():

    # 모델 로드
    print("CLIP 모델 로딩 중...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED
    )
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model     = model.to(DEVICE)

    # 데이터 로드
    pairs   = load_data(LAYER1_PATH, LAYER2_PATH)
    dataset = RecipeDataset(pairs, preprocess, tokenizer)
    loader  = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    # 옵티마이저
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR
    )

    # Loss (Contrastive)
    loss_fn = nn.CrossEntropyLoss()

    print("학습 시작!")

    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0

        for batch_idx, (images, texts) in enumerate(
            tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        ):
            images = images.to(DEVICE)
            texts  = texts.to(DEVICE)

            # 인코딩
            image_features = model.encode_image(images)
            text_features  = model.encode_text(texts)

            # 정규화
            image_features = image_features / image_features.norm(
                dim=-1, keepdim=True
            )
            text_features  = text_features / text_features.norm(
                dim=-1, keepdim=True
            )

            # 유사도 계산
            logit_scale = model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text  = logits_per_image.T

            # Loss 계산
            labels = torch.arange(
                len(images),
                device=DEVICE
            )
            loss = (
                loss_fn(logits_per_image, labels) +
                loss_fn(logits_per_text, labels)
            ) / 2

            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} 완료 | Loss: {avg_loss:.4f}")

        # 에폭마다 저장
        torch.save(
            model.state_dict(),
            f"clip_epoch{epoch+1}.pt"
        )

    # 최종 모델 저장
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"학습 완료! 모델 저장: {SAVE_PATH}")


if __name__ == "__main__":
    train()