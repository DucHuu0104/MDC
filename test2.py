import os
import hashlib
import json
import io
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

# LangChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
#from your_module import load_or_update_faiss, create_qa_chain, MedicineRecognitionModel
# ======================================================
# 🔍 Giả định bạn có sẵn model nhận dạng thuốc
# recognition_model.py
# class MedicineRecognitionModel:
#     def __init__(self, model_path): ...
#     def predict(self, image): return "Hoàng kỳ"
# ======================================================

# Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ Không tìm thấy API key trong file .env")
    st.stop()

# Cấu hình
DATA_PATH = "text.txt"
FAISS_PATH = "faiss_index"
HASH_PATH = os.path.join(FAISS_PATH, "data_hash.json")

# =============================
# 🔹 Hàm tiện ích
# =============================
def compute_md5(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

# =============================
# 🔹 FAISS loader/update
# =============================
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

# =============================
# 🔹 RAG QA Chain
# =============================
def create_qa_chain(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    prompt_template = """
    Bạn là một **chuyên gia y học cổ truyền Việt Nam** am hiểu dược liệu và bài thuốc bắc.
    Nhiệm vụ của bạn là đọc kỹ ngữ cảnh và trả lời câu hỏi của người dùng dựa trên thông tin có trong đó.

    🔹 Quy tắc trả lời:
    1. Giải thích ngắn gọn (2–3 câu tổng quát).
    2. Liệt kê chi tiết các **bài thuốc** liên quan (nếu có):
       - **Tên bài thuốc:** ...
       - **Thành phần:** ...
       - **Cách dùng:** ...
       - **Chỉ định:** ...
    3. Nếu có nhiều bài thuốc, sắp xếp theo độ phù hợp cao nhất.
    4. Nếu không tìm thấy, trả lời: "Không tìm thấy thông tin trong tài liệu."
    5. Cuối cùng, thêm ghi chú: *(Thông tin chỉ mang tính tham khảo, không thay thế tư vấn y khoa).*

    -----------------
    Ngữ cảnh: {context}
    Câu hỏi: {question}
    -----------------
    Trả lời:
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain

# =============================
# 🔹 Giao diện Streamlit
# =============================
def main():
    st.set_page_config(page_title="🌿 Chatbot Thuốc Bắc", page_icon="🌿", layout="wide")
    st.title("🌿 Chatbot Y học Cổ truyền – Nhận diện & Gợi ý bài thuốc")

    # 1️⃣ Khởi tạo cơ sở dữ liệu
    if not os.path.exists(DATA_PATH):
        st.error(f"❌ Không tìm thấy file {DATA_PATH}. Hãy đặt nó cùng thư mục ứng dụng.")
        st.stop()

    st.info("📂 Đang tải hoặc cập nhật FAISS database từ text.txt...")
    vector_store = load_or_update_faiss(DATA_PATH, FAISS_PATH)

    # 2️⃣ Sidebar: Nhập ảnh để nhận diện thuốc
    st.sidebar.header("📸 Nhận diện vị thuốc")
    uploaded_files = st.sidebar.file_uploader(
        "Tải ảnh vị thuốc (PNG, JPG, JPEG, BMP):",
        accept_multiple_files=True,
        type=["png", "jpg", "jpeg", "bmp"]
    )

    # 3️⃣ Khởi tạo model nhận dạng thuốc (cache để tăng tốc)
    @st.cache_resource
    def load_recognition_model():
        return MedicineRecognitionModel("model/thuoc_recognition.pth")

    recognition_model = load_recognition_model()

    # 4️⃣ Nhập câu hỏi văn bản
    st.markdown("### 💬 Hỏi trực tiếp về bài thuốc / triệu chứng")
    user_question = st.text_input("Ví dụ: 'Bài thuốc nào trị mất ngủ có vị kỷ tử và hoàng kỳ?'")

    if user_question:
        with st.spinner("🧘‍♂️ Đang tra cứu bài thuốc..."):
            qa_chain = create_qa_chain(vector_store)
            result = qa_chain({"query": user_question})

        st.success("✅ Hoàn tất.")
        st.markdown("### 🧠 Kết quả:")
        st.markdown(result["result"])

    # 5️⃣ Nhận dạng thuốc từ ảnh
    st.markdown("---")
    st.markdown("### 🧪 Kết quả nhận diện thuốc")

    if st.sidebar.button("🔍 Nhận diện & Gợi ý bài thuốc"):
        if not uploaded_files:
            st.sidebar.error("❌ Vui lòng tải ít nhất một ảnh.")
        else:
            detected_labels = []

            for file in uploaded_files:
                image = Image.open(io.BytesIO(file.read()))
                st.image(image, caption=f"Ảnh: {file.name}", use_column_width=True)

                with st.spinner("🔎 Đang nhận diện vị thuốc..."):
                    label = recognition_model.predict(image)
                    detected_labels.append(label)
                    st.success(f"✅ Phát hiện: **{label}**")

            if detected_labels:
                st.markdown("### 🧾 Kết quả nhận diện:")
                st.write(", ".join(detected_labels))

                # 🔁 Tự động truy vấn RAG để gợi ý bài thuốc
                st.markdown("### 💬 Gợi ý bài thuốc liên quan:")
                query = f"Các bài thuốc có chứa: {', '.join(detected_labels)}"
                qa_chain = create_qa_chain(vector_store)
                result = qa_chain({"query": query})
                st.markdown(result["result"])

# 🚀 Run app
if __name__ == "__main__":
    main()

