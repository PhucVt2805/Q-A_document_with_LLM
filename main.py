import time
import streamlit as st
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.llms import CTransformers
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# Tạo giao diện với Streamlit
container = st.container()
col1, col2, col3, col4 = container.columns(4)
setting = col1.button('Cài đặt')
col2.button('REFERENCES')
col3.button('Tải lên tệp PDF hoặc doc')
clear = col4.button('Xóa lịch sử')

# Tạo mô hình với CTransformers
llm = CTransformers(model='model/ggml-vistral-7B-chat-q8.gguf', model_type='llama', max_new_tokens=1024, temperature=0.1)

# Tạo embeddings với GPT4All
embeddings = GPT4AllEmbeddings(model_name='all-MiniLM-L6-v2.gguf2.f16.gguf', gpt4all_kwargs={'allow_download': 'True'})

# Định nghĩa template cho ChatPromptTemplate
system_prompt = (
    '''Bạn là trợ lý cho các nhiệm vụ trả lời câu hỏi. Sử dụng những thông tin được truy xuất sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói rằng bạn không biết, đừng cố tạo ra câu trả lời.
    
    {context}
    '''
)

# Khởi tạo ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Tải vectordb
vectordb = FAISS.load_local(folder_path='data/vectordb', embeddings=embeddings, allow_dangerous_deserialization=True)

# Khởi tạo QA chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(vectordb.as_retriever(), question_answer_chain)

# Khởi tạo phản hồi giả lập
def response_generator(text):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


# Khởi tạo session_state để lưu trữ lịch sử tin nhắn
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị tin nhắn lịch sử trên app khi chạy lại
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Nhận đầu vào từ người dùng
if question := st.chat_input("What is up?"):

    # Thêm tin nhắn của người dùng vào lịch sử chat
    st.session_state.messages.append({"role": "user", "content": question})
    # Hiển thị tin nhắn của người dùng trong container chat
    with st.chat_message("user"):
        st.markdown(question)
    response_stream = rag_chain.stream({"input": question})

    # Hiển thị phản hồi của trợ lý trong container chat
    with st.chat_message("ai"):
        for chunk in response_stream:
            if 'answer' in chunk:
                response = chunk["answer"]
                st.write_stream(response_generator(response))
                # Thêm phản hồi của trợ lý vào lịch sử chat
                st.session_state.messages.append({"role": "chatbot", "content": response})

if clear:
    st.session_state.messages = []
