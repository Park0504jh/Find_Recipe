from image_loader import load_image_dict
import streamlit as st
from PIL import Image
import json
import os
from googletrans import Translator

# ------------------------
# 번역기
# ------------------------
translator = Translator()

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
# Recipe1M 검색 함수
# ------------------------
def search_recipe_real(user_input):

    recipe_path = "recipe1M_layers"

    if not os.path.exists(recipe_path):
        return []

    # ------------------------
    # 한국어 입력 처리
    # ------------------------
    user_is_korean = is_korean(
        user_input
    )

    if user_is_korean:

        try:
            keyword = (
                translator.translate(
                    user_input,
                    src="ko",
                    dest="en"
                ).text.lower()
            )

        except:
            keyword = user_input.lower()

    else:
        keyword = user_input.lower()

    final_results = []

    with open(
        recipe_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            # JSON 배열 괄호 제거
            if line in ["[", "]"]:
                continue

            # 마지막 쉼표 제거
            if line.endswith(","):
                line = line[:-1]

            try:
                recipe = json.loads(
                    line
                )

            except:
                continue

            title = recipe.get(
                "title",
                ""
            ).lower()

            ingredients_text = ""

            for ing in recipe.get(
                "ingredients",
                []
            ):

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
            if (
                keyword in title
                or keyword in ingredients_text
            ):

                final_results.append(
                    recipe
                )

            if len(final_results) >= 5:
                break

    return final_results


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
        "데이터셋: Recipe1M+\n\n"
        "모델: YOLOv8"
    )

    st.divider()

    st.subheader(
        "Developer Info"
    )

    st.write(
        "👤 **Name:** 이지원"
    )

    st.caption(
        "Contact: wblee691919@duksung.ac.kr"
    )

    st.write(
        "👤 **Name:** 박정현"
    )

    st.caption(
        "Contact: park0504jh@duksung.ac.kr"
    )

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

# ------------------------
# 텍스트 검색
# ------------------------
with tab1:

    st.markdown(
        "## 어떤 요리를 원하시나요?"
    )

    user_input = st.text_input(
        "",
        placeholder=
        "예: 파스타, 치킨, chicken"
    )

    if st.button(
        "🔍 레시피 검색하기",
        use_container_width=True
    ):

        if user_input.strip():

            with st.spinner(
                "레시피 검색 중..."
            ):

                results = (
                    search_recipe_real(
                        user_input
                    )
                )

            if results:

                image_dict = load_image_dict()

                st.success(
                    f"{len(results)}개의 "
                    "레시피 발견!"
                )

                for idx, recipe in enumerate(
                    results,
                    start=1
                ):

                    title_text = recipe.get(
                        "title",
                        ""
                    )

                    # 한국어 검색이면 번역
                    if is_korean(
                        user_input
                    ):

                        try:
                            title_text = (
                                translator.translate(
                                    title_text,
                                    src="en",
                                    dest="ko"
                                ).text
                            )
                        except:
                            pass

                    st.markdown(
                        f"""
                        <div class='recipe-card'>
                        <h3>
                        🍳 {idx}.
                        {title_text}
                        </h3>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    recipe_id = recipe.get("id")
                    if recipe_id and recipe_id in image_dict:
                        st.image(image_dict[recipe_id], use_container_width=True)
                    with st.expander(
                        "상세 레시피 보기"
                    ):

                        # 재료
                        st.write(
                            "### 🥬 재료"
                        )

                        for ing in recipe.get(
                            "ingredients",
                            []
                        ):

                            ingredient_text = ""

                            if isinstance(
                                ing,
                                dict
                            ):
                                ingredient_text = (
                                    ing.get(
                                        "text",
                                        ""
                                    )
                                )

                            elif isinstance(
                                ing,
                                str
                            ):
                                ingredient_text = ing

                            # 한국어 검색 시 번역
                            if is_korean(
                                user_input
                            ):
                                try:
                                    ingredient_text = (
                                        translator.translate(
                                            ingredient_text,
                                            src="en",
                                            dest="ko"
                                        ).text
                                    )
                                except:
                                    pass

                            st.write(
                                "- "
                                + ingredient_text
                            )

                        # 조리법
                        st.write(
                            "### 👨‍🍳 조리법"
                        )

                        instructions = recipe.get(
                            "instructions",
                            []
                        )

                        if instructions:

                            for step_idx, step in enumerate(
                                instructions,
                                start=1
                            ):

                                step_text = ""

                                if isinstance(
                                    step,
                                    dict
                                ):
                                    step_text = (
                                        step.get(
                                            "text",
                                            ""
                                        )
                                    )

                                elif isinstance(
                                    step,
                                    str
                                ):
                                    step_text = step

                                # 한국어 검색 시 번역
                                if is_korean(
                                    user_input
                                ):
                                    try:
                                        step_text = (
                                            translator.translate(
                                                step_text,
                                                src="en",
                                                dest="ko"
                                            ).text
                                        )
                                    except:
                                        pass

                                st.write(
                                    f"{step_idx}. "
                                    + step_text
                                )

                        else:
                            st.info(
                                "조리법 정보가 없습니다."
                            )

            else:
                st.warning(
                    "검색 결과가 없습니다."
                )

        else:
            st.warning(
                "음식을 입력해주세요."
            )

# ------------------------
# 이미지 검색 탭
# ------------------------
with tab2:

    st.markdown(
        "## 음식 사진을 올려주세요"
    )

    uploaded_file = st.file_uploader(
        "이미지 업로드",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file:

        image = Image.open(
            uploaded_file
        )

        st.image(
            image,
            caption="업로드된 이미지",
            use_container_width=True
        )

        st.success(
            "이미지 업로드 완료!"
        )

# ------------------------
# Footer
# ------------------------
st.divider()

st.markdown(
    """
    <div style='text-align:center;
    color:gray;'>
    © 2026 Find Recipe AI Project
    </div>
    """,
    unsafe_allow_html=True
)