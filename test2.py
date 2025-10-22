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
# 🔹 RAG QA Chain (theo k chunk)
# ======================================================
def create_qa_chain(vector_store, k=5):
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)

    prompt_template = """
    Bạn là **chuyên gia y học cổ truyền Việt Nam**, am hiểu dược liệu và bài thuốc bắc.
    Nhiệm vụ của bạn là đọc kỹ NGỮ CẢNH và trả lời CÂU HỎI dựa trên thông tin có trong đó.

    🔹 Quy tắc trả lời:
    1. Giải thích ngắn gọn (2–3 câu tổng quát).
    2. Liệt kê chi tiết các **bài thuốc** liên quan (nếu có):
       - **Tên bài thuốc:** ...
       - **Thành phần:** ...
       - **Cách dùng:** ...
       - **Chỉ định:** ...
    3. Nếu có nhiều bài thuốc, sắp xếp theo độ phù hợp cao nhất.
    4. Nếu không có thông tin về bài thuốc, đưa ra bài thuốc có số lượng thành phần cao nhất.
    5. Cuối cùng, thêm ghi chú: *(Thông tin chỉ mang tính tham khảo, không thay thế tư vấn y khoa).*
    6. Nếu không tìm thấy, trả lời: "Không tìm thấy thông tin trong tài liệu."

    -----------------
    Ngữ cảnh (từ {k} đoạn gần nhất):
    {context}
    -----------------
    Câu hỏi: {question}
    -----------------
    Trả lời:
    """

    prompt = PromptTemplate(
        input_variables=["context", "question", "k"],
        template=prompt_template
    )

    def ask_with_top_k(query):
        # 🔹 Lấy ra k đoạn gần nhất
        docs = retriever.get_relevant_documents(query)
        top_k_docs = docs[:k]
        context = "\n\n".join([d.page_content for d in top_k_docs])

        # 🔹 Gọi LLM trực tiếp với prompt đã ghép
        final_prompt = prompt.format(context=context, question=query, k=k)
        response = llm.invoke(final_prompt)
        return response.content

    return ask_with_top_k

# ======================================================
# 🔹 Giao diện Streamlit
# ======================================================
def main():
    st.set_page_config(page_title="🌿 Chatbot Thuốc Bắc", page_icon="🌿", layout="wide")
    st.title("🌿 Chatbot Y học Cổ truyền – Nhận diện & Gợi ý bài thuốc")

    # 1️⃣ Khởi tạo cơ sở dữ liệu
    if not os.path.exists(DATA_PATH):
        st.error(f"❌ Không tìm thấy file {DATA_PATH}. Hãy đặt nó cùng thư mục ứng dụng.")
        st.stop()

    st.info("📂 Đang tải hoặc cập nhật FAISS database từ text.txt...")
    vector_store = load_or_update_faiss(DATA_PATH, FAISS_PATH)

    # 2️⃣ Sidebar: Chọn số chunk và tải ảnh
    st.sidebar.header("⚙️ Cài đặt")
    k = st.sidebar.slider("🔢 Số đoạn văn (chunk) sử dụng:", 1, 10, 5)

    st.sidebar.header("📸 Nhận diện vị thuốc")
    uploaded_files = st.sidebar.file_uploader(
        "Tải ảnh vị thuốc (PNG, JPG, JPEG, BMP):",
        accept_multiple_files=True,
        type=["png", "jpg", "jpeg", "bmp"]
    )

    # 3️⃣ Khởi tạo model YOLO nhận dạng thuốc
    @st.cache_resource
    def load_recognition_model():
        return YOLO(r'C:/Users/admin/OneDrive/Desktop/NCKH/recog.pt')

    recognition_model = load_recognition_model()

    # 4️⃣ Nhập câu hỏi văn bản
    st.markdown("### 💬 Hỏi trực tiếp về bài thuốc / triệu chứng")
    user_question = st.text_input("Ví dụ: 'Bài thuốc nào trị mất ngủ có vị kỷ tử và hoàng kỳ?'")

    

    # 5️⃣ Nhận dạng thuốc từ ảnh
    st.markdown("---")
    st.markdown("### 🧪 Kết quả nhận diện thuốc")

    if st.sidebar.button("🔍 Nhận diện & Gợi ý bài thuốc"):
        if not uploaded_files:
            st.sidebar.error("❌ Vui lòng tải ít nhất một ảnh.")
        else:
            detected_herbs = []

            for file in uploaded_files:
                image = Image.open(io.BytesIO(file.read()))
                st.image(image, caption=f"Ảnh: {file.name}", use_container_width=True)

                with st.spinner("🔎 Đang nhận diện vị thuốc..."):
                    result = recognition_model.predict(image)
                    names = recognition_model.names  # dict id -> name
                    for r in result:
                        for box in r.boxes:
                            cls = int(box.cls)
                            name = names[cls]
                            if name not in detected_herbs:
                                detected_herbs.append(name)
                                st.success(f"✅ Phát hiện: **{name}**")

            if detected_herbs:
                
                herb_list = ", ".join(detected_herbs)
                st.markdown("### 🧾 Kết quả nhận diện:")
                st.write(herb_list)

                # 🔁 Tự động truy vấn RAG để gợi ý bài thuốc
                st.markdown("### 💬 Gợi ý bài thuốc liên quan:")
                
                query = f"Các bài thuốc có chứa: {herb_list} dựa trên ngữ cảnh {user_question} "
                qa_chain = create_qa_chain(vector_store, k=k)
                result = qa_chain(query)
                st.markdown(result)

# 🚀 Run app
if __name__ == "__main__":
    main()
