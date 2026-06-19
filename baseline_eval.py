import json
import random
import torch
import open_clip
import requests
from PIL import Image
from io import BytesIO
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

LAYER1_PATH = r"C:\Users\DS\Desktop\recipe1M_layers\layer1.json"
LAYER2_PATH = r"C:\Users\DS\Desktop\recipe1M_layers\layer2.json"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"
BATCH_SIZE = 32
MAX_SAMPLES = 10000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
SEED = 42

print(f"사용 디바이스: {DEVICE}")


def load_data(layer1_path, layer2_path, max_samples=MAX_SAMPLES):
    print("layer2.json 로딩 중...")

    with open(layer2_path, "r", encoding="utf-8") as f:
        layer2 = json.load(f)

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
            title = recipe.get("title", "")

            if recipe_id in image_dict and title:
                pairs.append({
                    "id": recipe_id,
                    "image_url": image_dict[recipe_id],
                    "text": title
                })

            if len(pairs) >= max_samples:
                break

    print(f"총 {len(pairs)}개 쌍 준비 완료")
    return pairs


def split_data(pairs):
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


class RecipeDataset(Dataset):
    def __init__(self, pairs, preprocess, tokenizer):
        self.pairs = pairs
        self.preprocess = preprocess
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        try:
            response = requests.get(pair["image_url"], timeout=5)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            image = self.preprocess(image)
        except:
            image = torch.zeros(3, 224, 224)

        text = self.tokenizer([pair["text"]])[0]

        return image, text


def evaluate_retrieval(model, loader):
    model.eval()

    all_image_features = []
    all_text_features = []

    with torch.no_grad():
        for images, texts in tqdm(loader, desc="Baseline Test Retrieval"):
            images = images.to(DEVICE)
            texts = texts.to(DEVICE)

            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            all_image_features.append(image_features.cpu())
            all_text_features.append(text_features.cpu())

    image_features = torch.cat(all_image_features, dim=0)
    text_features = torch.cat(all_text_features, dim=0)

    similarity = image_features @ text_features.T

    total = similarity.size(0)
    r1_correct = 0
    r5_correct = 0

    for i in range(total):
        top5 = similarity[i].topk(5).indices.tolist()

        if i == top5[0]:
            r1_correct += 1

        if i in top5:
            r5_correct += 1

    r1 = r1_correct / total
    r5 = r5_correct / total

    return r1, r5


def main():
    print("Baseline OpenAI CLIP 모델 로딩 중...")

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED
    )

    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model = model.to(DEVICE)

    pairs = load_data(LAYER1_PATH, LAYER2_PATH)
    _, _, test_pairs = split_data(pairs)

    test_dataset = RecipeDataset(test_pairs, preprocess, tokenizer)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print("Baseline Test Set 평가 시작")

    r1, r5 = evaluate_retrieval(model, test_loader)

    print(f"Baseline Image Retrieval R@1: {r1:.4f}")
    print(f"Baseline Image Retrieval R@5: {r5:.4f}")

    metrics = {
        "model": MODEL_NAME,
        "pretrained": PRETRAINED,
        "max_samples": MAX_SAMPLES,
        "test_size": len(test_pairs),
        "baseline_image_retrieval_r1": r1,
        "baseline_image_retrieval_r5": r5
    }

    with open("baseline_metrics_result.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    print("baseline_metrics_result.json 저장 완료")


if __name__ == "__main__":
    main()