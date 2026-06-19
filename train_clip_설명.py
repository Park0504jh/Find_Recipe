import os
import json
import random
import torch
import open_clip
import requests
from PIL import Image
from io import BytesIO
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from tqdm import tqdm

# ============================================================
# 1. 기본 설정
# ============================================================

# 현재 train_clip.py 파일이 있는 위치
# 지금은 실제 데이터셋 경로를 하드코딩해서 사용하고 있지만,
# 제출용으로는 BASE_DIR 기준 상대경로로 바꾸는 것이 더 안전함
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Recipe1M+ 데이터셋 경로
# layer1.json: 레시피 제목, 재료, 조리법 등 텍스트 정보
# layer2.json: 레시피 ID별 이미지 URL 정보
LAYER1_PATH = r"C:\Users\DS\Desktop\recipe1M_layers\layer1.json"
LAYER2_PATH = r"C:\Users\DS\Desktop\recipe1M_layers\layer2.json"

# 사용할 CLIP 모델 설정
# ViT-B-32: Vision Transformer 기반 CLIP 모델
# pretrained="openai": OpenAI에서 사전학습한 기본 CLIP 가중치 사용
MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"

# 학습 하이퍼파라미터
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-5

# 전체 Recipe1M+ 중 최대 10,000개 이미지-텍스트 쌍만 사용
MAX_SAMPLES = 10000

# Validation Loss가 가장 좋았던 모델을 저장할 파일명
SAVE_PATH = "clip_finetuned.pt"

# GPU 사용 가능하면 cuda, 아니면 cpu 사용
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Train / Validation / Test 분할 비율
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1

# 랜덤 셔플 결과를 고정하기 위한 시드값
# 같은 데이터를 같은 방식으로 나누기 위해 사용
SEED = 42

print(f"사용 디바이스: {DEVICE}")


# ============================================================
# 2. 데이터 로딩 함수
# ============================================================

def load_data(layer1_path, layer2_path, max_samples=MAX_SAMPLES):
    """
    Recipe1M+의 layer1.json과 layer2.json을 연결하여
    이미지 URL과 레시피 제목(text)이 한 쌍으로 묶인 데이터를 만든다.

    반환 형식 예시:
    [
        {
            "id": "레시피 ID",
            "image_url": "이미지 URL",
            "text": "레시피 제목"
        },
        ...
    ]
    """

    print("layer2.json 로딩 중...")

    # layer2.json은 레시피 ID별 이미지 URL 정보를 담고 있음
    with open(layer2_path, "r", encoding="utf-8") as f:
        layer2 = json.load(f)

    # 레시피 ID를 key로, 첫 번째 이미지 URL을 value로 저장
    # 예: {"abc123": "http://...jpg"}
    image_dict = {}

    for item in layer2:
        images = item.get("images", [])

        # 이미지가 존재하는 레시피만 사용
        if images:
            image_dict[item["id"]] = images[0]["url"]

    print("layer1.json 로딩 중...")

    pairs = []

    # layer1.json은 매우 큰 파일이므로 전체를 json.load 하지 않고
    # 한 줄씩 읽어 메모리 사용량을 줄임
    with open(layer1_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # JSON 배열의 시작/끝 기호는 건너뜀
            if line in ["[", "]"]:
                continue

            # 줄 끝에 쉼표가 있으면 JSON 파싱 오류가 나므로 제거
            if line.endswith(","):
                line = line[:-1]

            try:
                recipe = json.loads(line)
            except:
                # JSON 파싱 실패한 줄은 건너뜀
                continue

            recipe_id = recipe.get("id")
            title = recipe.get("title", "")

            # 이미지 URL이 있고 제목도 있는 레시피만 학습에 사용
            if recipe_id in image_dict and title:
                pairs.append({
                    "id": recipe_id,
                    "image_url": image_dict[recipe_id],
                    "text": title
                })

            # 최대 샘플 수에 도달하면 중단
            if len(pairs) >= max_samples:
                break

    print(f"총 {len(pairs)}개 쌍 준비 완료")
    return pairs


# ============================================================
# 3. Train / Validation / Test 분할
# ============================================================

def split_data(pairs):
    """
    전체 데이터를 Train / Validation / Test로 분할한다.

    Train:
        모델 학습에 사용

    Validation:
        학습 중간에 성능을 확인하고,
        가장 좋은 모델을 저장하기 위해 사용

    Test:
        학습이 모두 끝난 뒤 최종 성능(R@1, R@5)을 평가하기 위해 사용

    Data Leakage 방지:
        데이터를 먼저 랜덤 셔플한 뒤 서로 겹치지 않게 분할한다.
    """

    random.seed(SEED)
    random.shuffle(pairs)

    total = len(pairs)

    train_end = int(total * TRAIN_RATIO)
    valid_end = train_end + int(total * VALID_RATIO)

    train_pairs = pairs[:train_end]
    valid_pairs = pairs[train_end:valid_end]
    test_pairs = pairs[valid_end:]

    print(f"Train: {len(train_pairs)}개")
    print(f"Valid: {len(valid_pairs)}개")
    print(f"Test : {len(test_pairs)}개")

    return train_pairs, valid_pairs, test_pairs


# ============================================================
# 4. PyTorch Dataset 정의
# ============================================================

class RecipeDataset(Dataset):
    """
    Recipe1M+ 이미지-텍스트 쌍을 PyTorch Dataset 형태로 변환하는 클래스.

    __getitem__에서 하는 일:
    1. 이미지 URL에서 음식 이미지를 다운로드
    2. CLIP preprocess로 이미지 전처리
    3. 레시피 제목을 CLIP tokenizer로 토큰화
    4. image tensor, text token 반환
    """

    def __init__(self, pairs, preprocess, tokenizer):
        self.pairs = pairs
        self.preprocess = preprocess
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        # 이미지 다운로드 및 전처리
        try:
            response = requests.get(pair["image_url"], timeout=5)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            image = self.preprocess(image)

        except:
            # 이미지 다운로드 실패 또는 손상된 이미지인 경우
            # 학습이 중단되지 않도록 빈 이미지 텐서로 대체
            image = torch.zeros(3, 224, 224)

        # 레시피 제목을 CLIP 텍스트 인코더 입력 형식으로 변환
        text = self.tokenizer([pair["text"]])[0]

        return image, text


# ============================================================
# 5. CLIP Contrastive Loss 계산
# ============================================================

def compute_clip_loss(model, images, texts, loss_fn):
    """
    CLIP의 이미지-텍스트 대조 학습 손실을 계산한다.

    핵심 아이디어:
    - 같은 batch 안에서 i번째 이미지는 i번째 텍스트와 정답 쌍
    - i번째 이미지가 i번째 텍스트와 가장 유사해지도록 학습
    - 동시에 i번째 텍스트도 i번째 이미지와 가장 유사해지도록 학습
    """

    # 이미지와 텍스트를 각각 CLIP 인코더에 통과시켜 feature vector 생성
    image_features = model.encode_image(images)
    text_features = model.encode_text(texts)

    # cosine similarity 계산을 위해 feature vector 정규화
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # CLIP 내부 학습 파라미터인 logit_scale 사용
    # 유사도 값의 스케일을 조정함
    logit_scale = model.logit_scale.exp()

    # 이미지 feature와 텍스트 feature 간 유사도 행렬 계산
    # shape: [batch_size, batch_size]
    logits_per_image = logit_scale * image_features @ text_features.T

    # 텍스트 기준 유사도 행렬은 전치 행렬
    logits_per_text = logits_per_image.T

    # batch 안에서 정답은 같은 인덱스끼리 매칭
    # 예: 0번 이미지 ↔ 0번 텍스트, 1번 이미지 ↔ 1번 텍스트
    labels = torch.arange(len(images), device=DEVICE)

    # 이미지→텍스트 방향 loss와 텍스트→이미지 방향 loss를 평균
    loss = (
        loss_fn(logits_per_image, labels) +
        loss_fn(logits_per_text, labels)
    ) / 2

    return loss


# ============================================================
# 6. Validation Loss 평가
# ============================================================

def evaluate_loss(model, loader, loss_fn):
    """
    Validation 데이터셋에 대한 평균 Loss를 계산한다.

    학습에는 사용하지 않고,
    현재 모델이 처음 보는 검증 데이터에서 얼마나 잘 맞는지 확인하는 용도.
    """

    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, texts in tqdm(loader, desc="Validation"):
            images = images.to(DEVICE)
            texts = texts.to(DEVICE)

            loss = compute_clip_loss(model, images, texts, loss_fn)
            total_loss += loss.item()

    return total_loss / len(loader)


# ============================================================
# 7. Image Retrieval R@1, R@5 평가
# ============================================================

def evaluate_retrieval(model, loader):
    """
    Test Set을 대상으로 Image Retrieval 성능을 평가한다.

    R@1:
        이미지 하나를 넣었을 때,
        가장 유사한 텍스트 1개 안에 정답 레시피 제목이 있는 비율

    R@5:
        이미지 하나를 넣었을 때,
        가장 유사한 텍스트 5개 안에 정답 레시피 제목이 있는 비율

    이 코드는 같은 batch가 아니라 전체 test set에 대해
    이미지 feature와 텍스트 feature를 모두 모은 뒤 similarity matrix를 계산한다.
    """

    model.eval()

    all_image_features = []
    all_text_features = []

    with torch.no_grad():
        for images, texts in tqdm(loader, desc="Test Retrieval"):
            images = images.to(DEVICE)
            texts = texts.to(DEVICE)

            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # GPU 메모리 절약을 위해 CPU로 옮겨 저장
            all_image_features.append(image_features.cpu())
            all_text_features.append(text_features.cpu())

    # 전체 test set feature를 하나의 tensor로 합침
    image_features = torch.cat(all_image_features, dim=0)
    text_features = torch.cat(all_text_features, dim=0)

    # 전체 이미지와 전체 텍스트 간 유사도 행렬
    # shape: [test_size, test_size]
    similarity = image_features @ text_features.T

    total = similarity.size(0)
    r1_correct = 0
    r5_correct = 0

    for i in range(total):
        # i번째 이미지에 대해 가장 유사한 텍스트 5개 인덱스
        top5 = similarity[i].topk(5).indices.tolist()

        # top1이 자기 자신의 텍스트이면 R@1 정답
        if i == top5[0]:
            r1_correct += 1

        # top5 안에 자기 자신의 텍스트가 있으면 R@5 정답
        if i in top5:
            r5_correct += 1

    r1 = r1_correct / total
    r5 = r5_correct / total

    return r1, r5


# ============================================================
# 8. 전체 학습 과정
# ============================================================

def train():
    """
    전체 학습 흐름:

    1. OpenAI pretrained CLIP 모델 로드
    2. Recipe1M+ 이미지-텍스트 쌍 로드
    3. Train / Validation / Test 분할
    4. Train 데이터로 CLIP 파인튜닝
    5. 매 Epoch마다 Validation Loss 계산
    6. 가장 좋은 Validation Loss 모델 저장
    7. Test Set으로 Image Retrieval R@1, R@5 평가
    8. 결과를 metrics_result.json으로 저장
    """

    print("CLIP 모델 로딩 중...")

    # OpenAI 사전학습 CLIP 모델 로드
    # 이 시점에서는 기존 clip_finetuned.pt를 불러오지 않으므로
    # 매번 원본 OpenAI CLIP에서 새로 파인튜닝이 시작됨
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED
    )

    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model = model.to(DEVICE)

    # 데이터 로드 및 분할
    pairs = load_data(LAYER1_PATH, LAYER2_PATH)
    train_pairs, valid_pairs, test_pairs = split_data(pairs)

    # PyTorch Dataset 생성
    train_dataset = RecipeDataset(train_pairs, preprocess, tokenizer)
    valid_dataset = RecipeDataset(valid_pairs, preprocess, tokenizer)
    test_dataset = RecipeDataset(test_pairs, preprocess, tokenizer)

    # DataLoader 생성
    # train_loader만 shuffle=True로 설정하여 학습 순서를 매번 섞음
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # AdamW optimizer 사용
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    # CrossEntropyLoss를 사용하여 이미지-텍스트 매칭 loss 계산
    loss_fn = nn.CrossEntropyLoss()

    best_valid_loss = float("inf")

    print("학습 시작!")

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0

        for images, texts in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images = images.to(DEVICE)
            texts = texts.to(DEVICE)

            # CLIP contrastive loss 계산
            loss = compute_clip_loss(model, images, texts, loss_fn)

            # 역전파 및 파라미터 업데이트
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        # 한 epoch의 평균 train loss
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation set으로 현재 모델 평가
        avg_valid_loss = evaluate_loss(model, valid_loader, loss_fn)

        print(f"Epoch {epoch+1} 완료")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Valid Loss: {avg_valid_loss:.4f}")

        # 각 epoch별 가중치 저장
        torch.save(model.state_dict(), f"clip_epoch{epoch+1}.pt")

        # Validation Loss가 가장 낮은 모델을 최종 모델로 저장
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"Best model 저장: {SAVE_PATH}")

    print("Test Set 평가 시작")

    # 학습이 끝난 모델로 Test Set Retrieval 성능 평가
    r1, r5 = evaluate_retrieval(model, test_loader)

    print(f"Image Retrieval R@1: {r1:.4f}")
    print(f"Image Retrieval R@5: {r5:.4f}")

    # 성능 결과를 json 파일로 저장
    metrics = {
        "model": MODEL_NAME,
        "pretrained": PRETRAINED,
        "max_samples": MAX_SAMPLES,
        "train_size": len(train_pairs),
        "valid_size": len(valid_pairs),
        "test_size": len(test_pairs),
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "best_valid_loss": best_valid_loss,
        "image_retrieval_r1": r1,
        "image_retrieval_r5": r5
    }

    with open("metrics_result.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    print("metrics_result.json 저장 완료")
    print("학습 완료!")


# ============================================================
# 9. 실행 시작점
# ============================================================

if __name__ == "__main__":
    train()