import os
import hashlib
import json
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
#from recognition_model import MedicineRecognitionModel
from PIL import Image

# 🔐 Load API key

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ Không tìm thấy API key trong file .env")
    st.stop()


# Cấu hình

DATA_PATH = "text.txt"           # Database thuốc bắc
FAISS_PATH = "faiss_index"       # Thư mục FAISS
HASH_PATH = os.path.join(FAISS_PATH, "data_hash.json")


# Hàm tiện ích

@st.cache_resource
def load_recognition_model():
    return #MedicineRecognitionModel("model/thuoc_recognition.pth")

recognition_model = load_recognition_model()

def compute_md5(text):
    """Tính hash MD5 của đoạn văn bản."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def compute_file_hash(filepath):
    """Tính hash tổng của file text.txt."""
    with open(filepath, "r", encoding="utf-8") as f:
        return hashlib.md5(f.read().encode("utf-8")).hexdigest()



# Incremental FAISS Loader

def load_or_update_faiss(data_path=DATA_PATH, db_path=FAISS_PATH):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Đảm bảo thư mục tồn tại
    os.makedirs(db_path, exist_ok=True)

    # Đọc nội dung file text
    with open(data_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = splitter.split_text(full_text)

    # Load hash cũ (nếu có)
    old_hashes = {}
    if os.path.exists(HASH_PATH):
        with open(HASH_PATH, "r", encoding="utf-8") as f:
            old_hashes = json.load(f)

    # Tính hash mới của từng chunk
    new_chunks = []
    new_hashes = {}
    for chunk in chunks:
        h = compute_md5(chunk)
        new_hashes[h] = True
        if h not in old_hashes:
            new_chunks.append(chunk)

    # Nếu chưa có FAISS → tạo mới
    if not os.path.exists(os.path.join(db_path, "index.faiss")):
        st.info("🧠 Tạo mới FAISS database...")
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    else:
        # Load FAISS cũ
        vector_store = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        # Nếu có chunk mới thì thêm vào
        if new_chunks:
            st.info(f"🔄 Phát hiện {len(new_chunks)} đoạn mới → cập nhật FAISS...")
            vector_store.add_texts(new_chunks)
        else:
            st.success("✅ Không có thay đổi mới trong database.")

    # Save FAISS + hash
    vector_store.save_local(db_path)
    with open(HASH_PATH, "w", encoding="utf-8") as f:
        json.dump(new_hashes, f, ensure_ascii=False, indent=2)

    st.success("✅ FAISS đã được cập nhật.")
    return vector_store


# RAG QA Chain
def create_qa_chain(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    prompt_template = """
    Bạn là một **chuyên gia y học cổ truyền Việt Nam** am hiểu dược liệu và bài thuốc bắc.
    Nhiệm vụ của bạn là đọc kỹ ngữ cảnh và trả lời câu hỏi của người dùng dựa trên thông tin có trong đó.

    🔹 Quy tắc trả lời:
    1. Giải thích ngắn gọn trước (2–3 câu tổng quát).
    2. Liệt kê chi tiết các **bài thuốc** liên quan (nếu có), gồm:
       - **Tên bài thuốc:** ...
       - **Thành phần:** ...
       - **Cách dùng:** ...
       - **Chỉ định:** ...
       - **Trích dẫn:** ...
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



# UI Streamlit

def main():
    st.set_page_config(page_title="Chatbot Thuốc Bắc", page_icon="🌿", layout="wide")
    st.title("🌿 Chatbot Y học Cổ truyền – Phân tích & Gợi ý bài thuốc")

    # 1️⃣ Tải và kiểm tra dữ liệu
    if not os.path.exists(DATA_PATH):
        st.error(f"❌ Không tìm thấy file {DATA_PATH}. Hãy đặt nó cùng thư mục ứng dụng.")
        st.stop()

    st.info("📂 Đang tải hoặc cập nhật FAISS database từ text.txt...")
    vector_store = load_or_update_faiss(DATA_PATH, FAISS_PATH)
    

    # 2️⃣ Nhập câu hỏi
    st.markdown("### 💬 Hỏi về bài thuốc / triệu chứng")
    user_question = st.text_input("Ví dụ: 'Bài thuốc nào trị mất ngủ có vị kỷ tử và hoàng kỳ?'")

    if user_question:
        with st.spinner("🧘‍♂️ Đang tham khảo bài thuốc trong cơ sở dữ liệu..."):
            qa_chain = create_qa_chain(vector_store)
            result = qa_chain({"query": user_question})

        st.success("✅ Đã hoàn thành quá trình tham khảo bài thuốc.")
        st.markdown("### 🧠 Kết quả phân tích:")
        st.write(result["result"])
        
    
# Run
if __name__ == "__main__":
    main()
