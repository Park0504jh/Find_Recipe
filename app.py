from image_loader import load_image_dict
import streamlit as st
from PIL import Image
import json
import os
import torch
import open_clip
import torch.nn.functional as F
import requests
from deep_translator import GoogleTranslator


# ------------------------
# 이미지 딕셔너리 생성
# ------------------------
image_dict = {}

with open(
    "layer2.json",
    "r",
    encoding="utf-8"
) as f:

    layer2 = json.load(f)

for item in layer2:

    images = item.get(
        "images",
        []
    )

    if images:

        image_dict[
            item["id"]
        ] = images[0]["url"]


# ------------------------
# 번역기
# ------------------------
def translate_text(text, source="auto", target="ko"):

    try:

        return GoogleTranslator(
            source=source,
            target=target
        ).translate(text)

    except:

        return text

# ------------------------
# CLIP 모델 로드
# ------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

clip_model, _, preprocess = (
    open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai"
    )
)

clip_model.load_state_dict(
    torch.load(
        "clip_finetuned.pt",
        map_location=DEVICE
    )
)

clip_model = clip_model.to(DEVICE)
clip_model.eval()

# ------------------------
# 레시피 임베딩 로드
# ------------------------

embedding_data = torch.load(
    "recipe_embeddings.pt",
    map_location="cpu"
)

recipe_db = embedding_data["recipes"]

recipe_embeddings = embedding_data[
    "embeddings"
]

# ------------------------
# 페이지 설정
# ------------------------
st.set_page_config(
    page_title="Find Recipe",
    page_icon="🍳",
    layout="wide"
)

# ------------------------
# 한국어 여부 확인
# ------------------------
def is_korean(text):

    return any(
        '가' <= ch <= '힣'
        for ch in text
    )

# ------------------------
# 존재하지 않는 url skip
# ------------------------
def image_exists(url):

    try:

        response = requests.head(
            url,
            timeout=3,
            allow_redirects=True
        )

        return response.status_code == 200

    except:

        return False
    
# ------------------------
# Recipe1M 검색 함수
# ------------------------
def search_recipe_real(user_input, threshold):

    recipe_path = "recipe1M_layers"

    if not os.path.exists(recipe_path):
        return []

    user_is_korean = is_korean(user_input)

    if user_is_korean:
        try:
            keyword = translate_text(
                user_input,
                source="ko",
                target="en").lower()

        except:
            keyword = user_input.lower()
    else:
        keyword = user_input.lower()

    final_results = []

    with open(recipe_path, "r", encoding="utf-8") as f:
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

            title = recipe.get("title", "").lower()
            ingredients_text = ""

            for ing in recipe.get("ingredients", []):
                if isinstance(ing, dict):
                    ingredients_text += ing.get("text", "").lower() + " "
                elif isinstance(ing, str):
                    ingredients_text += ing.lower() + " "

                if isinstance(
                    ing,
                    dict
                ):

                    ingredients_text += (
                        ing.get(
                            "text",
                            ""
                        ).lower()
                        + " "
                    )

                elif isinstance(
                    ing,
                    str
                ):

                    ingredients_text += (
                        ing.lower()
                        + " "
                    )

            # 제목 + 재료 검색
            confidence = 0.0

            if keyword in title:
                confidence = 1.0

            elif keyword in ingredients_text:
                confidence = 0.5

            if confidence >= threshold:
                recipe["confidence"] = confidence

                recipe_id = recipe.get("id")

                image_url = image_dict.get(recipe_id)

                # 이미지 URL 없음
                if not image_url:
                    continue

                # 이미지 URL 죽어있음
                if not image_exists(image_url):
                    continue

                if confidence >= threshold:
                    recipe["confidence"] = confidence
                    final_results.append(recipe)

            if len(final_results) >= 5:
                break

    return final_results

def find_recipe_from_image(image, threshold):

    image_tensor = (
        preprocess(image)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():

        image_feature = (
            clip_model.encode_image(
                image_tensor
            )
        )

        image_feature = F.normalize(
            image_feature,
            dim=-1
        )

    similarities = (
        image_feature.cpu()
        @ recipe_embeddings.T
    )

    topk_scores, topk_indices = similarities.topk(50)

    results = []

    for score, idx in zip(
        topk_scores[0],
        topk_indices[0]
    ):

        confidence = float(score)

        if confidence < threshold:
            continue

        recipe = recipe_db[idx]

        recipe["confidence"] = confidence

        image_url = image_dict.get(recipe["id"])

        if not image_url:
            continue

        if not image_exists(image_url):
            continue

        results.append(recipe)

        if len(results) >= 5:
            break


    return results


# ------------------------
# CSS
# ------------------------
st.markdown("""
<style>

.main {
    background-color: #f8f9fa;
}

.title {
    text-align: center;
    color: #ff4b4b;
    font-size: 56px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #666;
    font-size: 20px;
    margin-bottom: 30px;
}

div.stButton > button:first-child {
    background-color: #ff4b4b;
    color: white;
    border-radius: 12px;
    width: 100%;
    border: none;
    height: 50px;
    font-size: 18px;
}

.recipe-card {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)

# ------------------------
# 사이드바
# ------------------------
with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/3565/3565418.png",
        width=100
    )

    st.title("Settings")

    st.info(
        "Data Set: Recipe1M+\n\n"
        "Embedding Mode: CLIP\n\n" 
        "Vision Encoder: ViT-B/32\n\n"
        "Text Encoder: Transformer\n\n"
        "Training Framework: PyTorch\n\n"
        "Similarity Metric: Cosine Similarity\n\n"
        "Backend: PyTorch\n\n"
        "Frontend/UI: Streamlit\n\n"
        "Data Processing: Python\n\n"
    )

    st.divider()

    st.subheader("Developer Info")

    st.write("👤 **Name:** 이지원")
    st.caption("Contact: wblee691919@duksung.ac.kr")

    st.write("👤 **Name:** 박정현")
    st.caption("Contact: park0504jh@duksung.ac.kr")

# ------------------------
# 제목
# ------------------------
st.markdown(
    """
    <div class='title'>
    🍳 Find Recipe!
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class='subtitle'>
    재료를 입력하거나 사진을 찍어
    최고의 레시피를 찾아보세요.
    </div>
    """,
    unsafe_allow_html=True
)

# ------------------------
# 탭
# ------------------------
tab1, tab2 = st.tabs([
    "💬 텍스트로 찾기",
    "📸 사진으로 찾기"
])

threshold = st.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.20,
    step=0.01
)

# ------------------------
# 텍스트 검색
# ------------------------
with tab1:

    st.session_state["results"] = []
    if "results" not in st.session_state:
        st.session_state["results"] = []
        st.session_state["query"] = user_input

    st.markdown(
        "## 어떤 요리를 원하시나요?"
    )

    user_input = st.text_input(
        "",
        placeholder="예: 파스타, 치킨, chicken"
    )

    if st.button("🔍 레시피 검색하기", use_container_width=True):

        if user_input.strip():

            with st.spinner("레시피 검색 중..."):
                results = search_recipe_real(user_input, threshold)

                st.session_state["results"] = results

            results = st.session_state["results"]

            if results:

                st.success(
                    f"{len(results)}개의 레시피 발견!"
            )

                display_idx = 1
                for recipe in results:

                    title_text = recipe.get(
                        "title",
                        ""
                    )

                    recipe_id = recipe.get(
                        "id",
                        ""
                    )

                    image_url = image_dict.get(
                        recipe_id
                    )

                    if not image_url:
                        continue

                    if not image_exists(image_url):
                        continue

                    # 한국어 검색이면 번역
                    if is_korean(
                        user_input
                    ):

                        if is_korean(user_input):
                            try:
                                title_text = translate_text(
                                    title_text,
                                    source="en",
                                    target="ko"
                        )
                            except:
                                pass
                    
                    st.markdown(
                        f"""
                        <div class='recipe-card'>
                        <h3>🍳 {display_idx}. {title_text}</h3>
                        <p>⭐ Confidence: {recipe["confidence"]:.2f}</p>
                        <img src="{image_url}"
                            width="350"
                            style="border-radius:10px;
                                    margin-top:10px;">
                        </div>
                        """,
                    unsafe_allow_html=True
                    )

                    display_idx += 1

                    with st.expander("상세 레시피 보기"):

                        st.write("### 🥬 재료")

                        for ing in recipe.get("ingredients", []):
                            ingredient_text = ""
                            if isinstance(ing, dict):
                                ingredient_text = ing.get("text", "")
                            elif isinstance(ing, str):
                                ingredient_text = ing

                            if is_korean(user_input):
                                try:
                                    ingredient_text = translate_text(
                                        ingredient_text,
                                        source="en",
                                        target="ko"
                                    )
                                except:
                                    pass

                            st.write("- " + ingredient_text)

                        st.write("### 👨‍🍳 조리법")

                        instructions = recipe.get("instructions", [])

                        if instructions:

                            MAX_STEPS = 10

                            for step_idx, step in enumerate(
                                instructions[:MAX_STEPS],
                            start=1
                            ):

                                step_text = ""
                                if isinstance(step, dict):
                                    step_text = step.get("text", "")
                                elif isinstance(step, str):
                                    step_text = step

                                if is_korean(user_input):
                                    try:
                                        step_text = translate_text(
                                            step_text,
                                            source="en",
                                            target="ko"
                                        )
                                    except:
                                        pass

                                st.write(f"{step_idx}. " + step_text)
                        else:
                            st.info("조리법 정보가 없습니다.")

            else:
                st.warning("검색 결과가 없습니다.")

        else:
            st.warning("음식을 입력해주세요.")

# ------------------------
# 이미지 검색 탭
# ------------------------
with tab2:

    language_img = st.radio(
        "출력 언어 선택",
        ["한국어", "English"],
        horizontal=True
    )

    st.markdown(
        "## 음식 사진을 올려주세요"
    )

    uploaded_file = st.file_uploader(
        "이미지 업로드",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            caption="업로드된 이미지",
            width=300
        )

        if st.button(
            "🍳 레시피 찾기",
            use_container_width=True
        ):

            with st.spinner(
                "CLIP 분석 중..."
            ):

                results = (
                    find_recipe_from_image(
                        image, threshold
                    )
                )

            st.success(
                f"{len(results)}개의 추천 레시피"
            )

            for idx, recipe in enumerate(
                results,
                start=1
            ):
                
                image_url = image_dict.get(recipe["id"])
                

                title_text = recipe.get("title", "")

                if language_img == "한국어":
                    title_text = translate_text(title_text, "en", "ko")

                st.markdown(
                    f"""
                    <div class='recipe-card'>
                    <h3>
                    🍳 {idx}. {title_text}
                    </h3>
                    ⭐ Confidence: {recipe["confidence"]:.2f}
                    """,
                    unsafe_allow_html=True
                )

                if image_url:
                    st.image(image_url, width=350)

                st.markdown(
                    """
                    </div>
                    """,
                    unsafe_allow_html=True
            )

                with st.expander(
                    "상세 레시피 보기"
                ):

                    st.write("### 🥬 재료")

                    for ing in recipe.get("ingredients", []):

                        if isinstance(ing, dict):
                            ingredient_text = ing.get("text", "")
                        else:
                            ingredient_text = ing

                        if language_img == "한국어":
                            ingredient_text = translate_text(ingredient_text, "en", "ko")

                        st.write("- " + ingredient_text)


                    st.write("### 👨‍🍳 조리법")

                    for step_idx, step in enumerate(
                        recipe.get("instructions", []),
                        start=1
                    ):

                        if isinstance(step, dict):
                            step_text = step.get("text", "")
                        else:
                            step_text = step

                        if language_img == "한국어":
                            step_text = translate_text(step_text, "en", "ko")

                        st.write(f"{step_idx}. {step_text}")

# ------------------------
# Footer
# ------------------------
st.divider()

st.markdown(
    """
    <div style='text-align:center;
    color:gray;'>
    © 2026 Find Recipe Project
    </div>
    """,
    unsafe_allow_html=True
)
