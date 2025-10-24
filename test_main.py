import os
import hashlib
import json
import io
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from ultralytics import YOLO
# LangChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate

# ======================================================
# ⚙️ Cấu hình môi trường
# ======================================================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ Không tìm thấy API key trong file .env")
    st.stop()

DATA_PATH = "text.txt"
FAISS_PATH = "faiss_index"
HASH_PATH = os.path.join(FAISS_PATH, "data_hash.json")

# ======================================================
# 🔹 Hàm tiện ích
# ======================================================
def compute_md5(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def load_text_database(path):
    """Đọc toàn bộ text.txt và tách theo bài thuốc (mỗi bài cách nhau 1 dòng trống)."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return [x.strip() for x in content.split("\n\n") if x.strip()]

# ======================================================
# 🔹 FAISS loader/update
# ======================================================
def load_or_update_faiss(data_path=DATA_PATH, db_path=FAISS_PATH):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    os.makedirs(db_path, exist_ok=True)

    with open(data_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = splitter.split_text(full_text)

    old_hashes = {}
    if os.path.exists(HASH_PATH):
        with open(HASH_PATH, "r", encoding="utf-8") as f:
            old_hashes = json.load(f)

    new_chunks = []
    new_hashes = {}
    for chunk in chunks:
        h = compute_md5(chunk)
        new_hashes[h] = True
        if h not in old_hashes:
            new_chunks.append(chunk)

    if not os.path.exists(os.path.join(db_path, "index.faiss")):
        st.info("🧠 Tạo mới FAISS database...")
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    else:
        vector_store = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        if new_chunks:
            st.info(f"🔄 Phát hiện {len(new_chunks)} đoạn mới → cập nhật FAISS...")
            vector_store.add_texts(new_chunks)
        else:
            st.success("✅ Không có thay đổi mới trong cơ sở dữ liệu.")

    vector_store.save_local(db_path)
    with open(HASH_PATH, "w", encoding="utf-8") as f:
        json.dump(new_hashes, f, ensure_ascii=False, indent=2)

    st.success("✅ FAISS đã được cập nhật.")
    return vector_store

# ======================================================
# 🔹 Hàm tìm bài thuốc chứa nhiều vị trùng nhất
# ======================================================
def find_best_matches(herb_list, database, max_return=5):
    """
    Trả về các bài thuốc có nhiều vị trùng nhất.
    Nếu không có bài nào chứa toàn bộ, giảm dần.
    """
    herbs = [h.strip() for h in herb_list.split(",") if h.strip()]
    matched = []
    for entry in database:
        count = sum(1 for h in herbs if h in entry)
        if count > 0:
            matched.append((entry, count))

    matched.sort(key=lambda x: x[1], reverse=True)

    if not matched:
        return []

    # chọn top bài thuốc có số vị trùng cao nhất hoặc giảm dần
    top_count = matched[0][1]
    filtered = [x for x in matched if x[1] == top_count]

    # nếu không đủ, thêm bài có ít hơn
    if len(filtered) < max_return:
        filtered += [x for x in matched if x[1] < top_count][: max_return - len(filtered)]

    return filtered[:max_return]

# ======================================================
# 🔹 RAG QA Chain
# ======================================================
def create_qa_chain(vector_store, k=5):
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)

    prompt_template = """
    Bạn là **chuyên gia y học cổ truyền Việt Nam**, am hiểu dược liệu và bài thuốc.
    Hãy đọc các bài thuốc trong dữ liệu sau và đưa ra câu trả lời theo yêu cầu.

    -----------------
    Các bài thuốc được chọn từ cơ sở dữ liệu (ưu tiên nhiều vị trùng khớp nhất):
    {top_recipes}
    -----------------
    Câu hỏi người dùng: {question}
    -----------------
    Hãy:
    1. Giải thích ngắn gọn (2–3 câu tổng quát).
    2. Liệt kê chi tiết các bài thuốc liên quan:
       - **Tên bài thuốc:** ...
       - **Thành phần:** ...
       - **Cách dùng:** ...
       - **Chỉ định:** ...
    3. Nếu nhiều bài thuốc, sắp xếp theo độ phù hợp cao nhất.
    4. Nếu không có thông tin, nói: "Không tìm thấy trong cơ sở dữ liệu."
    5. Cuối cùng ghi chú: *(Thông tin chỉ mang tính tham khảo, không thay thế tư vấn y khoa).*
    -----------------
    Trả lời:
    """

    prompt = PromptTemplate(
        input_variables=["top_recipes", "question"],
        template=prompt_template
    )

    def ask_with_database(top_recipes, user_question):
        context = "\n\n".join([r[0] for r in top_recipes])
        final_prompt = prompt.format(top_recipes=context, question=user_question)
        response = llm.invoke(final_prompt)
        return response.content

    return ask_with_database

# ======================================================
# 🔹 Giao diện Streamlit
# ======================================================
def main():
    st.set_page_config(page_title="🌿 Chatbot Thuốc Bắc", page_icon="🌿", layout="wide")
    st.title("🌿 Chatbot Y học Cổ truyền – Nhận diện & Gợi ý bài thuốc")

    # 1️⃣ Khởi tạo database
    if not os.path.exists(DATA_PATH):
        st.error(f"❌ Không tìm thấy file {DATA_PATH}. Hãy đặt nó cùng thư mục ứng dụng.")
        st.stop()

    vector_store = load_or_update_faiss(DATA_PATH, FAISS_PATH)
    text_database = load_text_database(DATA_PATH)

    # 2️⃣ Sidebar
    st.sidebar.header("⚙️ Cài đặt")
    k = st.sidebar.slider("🔢 Số đoạn FAISS (k):", 1, 10, 5)

    st.sidebar.header("📸 Nhận diện vị thuốc")
    uploaded_files = st.sidebar.file_uploader(
        "Tải ảnh vị thuốc (PNG, JPG, JPEG, BMP):",
        accept_multiple_files=True,
        type=["png", "jpg", "jpeg", "bmp"]
    )

    @st.cache_resource
    def load_recognition_model():
        return YOLO('recog.pt')

    recognition_model = load_recognition_model()

    # 3️⃣ Hỏi văn bản
    st.markdown("### 💬 Câu hỏi")
    user_question = st.text_input("Ví dụ: 'Bài thuốc nào trị mất ngủ có vị kỷ tử và hoàng kỳ?'")

    # 4️⃣ Nhận diện thuốc
    if st.sidebar.button("Nhận diện & Gợi ý bài thuốc"):
        if not uploaded_files:
            st.sidebar.error("Vui lòng tải ít nhất một ảnh.")
            return

        detected_herbs = []
        for file in uploaded_files:
            image = Image.open(io.BytesIO(file.read()))
            st.image(image, caption=f"Ảnh: {file.name}", use_container_width=True)

            with st.spinner("Đang nhận diện vị thuốc..."):
                results = recognition_model.predict(image)
                names = recognition_model.names
                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls)
                        name = names[cls]
                        if name not in detected_herbs:
                            detected_herbs.append(name)
                            st.success(f"Phát hiện: **{name}**")

        if not detected_herbs:
            st.warning("⚠️ Không phát hiện được vị thuốc nào.")
            return

        herb_list = ", ".join(detected_herbs)
        st.markdown("### 🧾 Vị thuốc nhận diện:")
        st.write(herb_list)

        # 5️⃣ Map database để chọn bài phù hợp nhất
        matched_recipes = find_best_matches(herb_list, text_database)
        if not matched_recipes:
            st.error("❌ Không tìm thấy bài thuốc nào trong cơ sở dữ liệu.")
            return

        st.markdown("### 📚 Các bài thuốc phù hợp nhất (lọc theo số vị trùng):")
        for i, (text, count) in enumerate(matched_recipes, 1):
            st.markdown(f"{i}. ({count} vị trùng)\n{text}\n---")

        # 6️⃣ Gọi GPT để trả lời tự nhiên hơn
        qa_chain = create_qa_chain(vector_store, k=k)
        st.markdown("### 💬 Gợi ý tổng hợp:")
        answer = qa_chain(matched_recipes, user_question)
        st.markdown(answer)

# 🚀 Run
if __name__ == "__main__":
    main()
