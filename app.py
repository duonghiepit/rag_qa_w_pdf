import chainlit as cl
import torch

from chainlit.types import AskFileResponse

from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface.llms import HuggingFacePipeline

from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain.chains import ConversationalRetrievalChain

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
import asyncio

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# Initialize embedding
embedding = HuggingFaceEmbeddings()

def process_file(file: AskFileResponse):
    """
    Process the uploaded file and split it into documents.
    """
    # Determine the loader based on file type
    if file.type == "text/plain":
        Loader = TextLoader
    elif file.type == "application/pdf":
        Loader = PyPDFLoader
    
    # Load and split the document
    print(file.path)
    loader = Loader(file.path)
    documents = loader.load()
    docs = text_splitter.split_documents(documents)
    
    # Add metadata to documents
    for i, doc in enumerate(docs):
        doc.metadata["source"] = f"source_{i}"
    
    return docs

def get_vector_db(file: AskFileResponse):
    """
    Create a vector database from the processed documents.
    """
    docs = process_file(file)
    print(docs)
    cl.user_session.set("docs", docs)
    vector_db = Chroma.from_documents(documents=docs, embedding=embedding)
    return vector_db

def get_huggingface_llm(model_name: str = "lmsys/vicuna-7b-v1.5", max_new_token: int = 512):
    """
    Initialize and return the HuggingFace language model.
    """

    # Kiểm tra xem có GPU hay không
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        # Sử dụng cấu hình quantization nếu có GPU
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        # Không sử dụng quantization nếu chạy trên CPU
        nf4_config = None
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=nf4_config if device == "cuda" else None,
        low_cpu_mem_usage=True,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_token,
        pad_token_id=tokenizer.eos_token_id,
        device_map="auto",
    )
    
    llm = HuggingFacePipeline(
        pipeline=model_pipeline,
    )
    return llm

# Initialize the language model
LLM = get_huggingface_llm()

# Welcome message for users
welcome_message = """Welcome to the PDF QA! To get started:
1. Upload a PDF or text file
2. Ask a question about the file
"""

@cl.on_chat_start
async def on_chat_start():
    """
    Handle the start of the chat session.
    """
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content=welcome_message,
            accept=["text/plain", "application/pdf"],
            max_size_mb=20,
            timeout=3600
        ).send()

    file = files[0]
    print(file)

    msg = cl.Message(content=f"Processing '{file.name}'...", disable_feedback=True)
    await msg.send()

    vector_db = await cl.make_async(get_vector_db)(file)
    print("vector_db", vector_db)

    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )
    retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={'k': 3})

    chain = ConversationalRetrievalChain.from_llm(
        llm=LLM,
        chain_type="stuff",
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )

    msg.content = f"'{file.name}' processed. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def on_message(message: cl.Message):
    """
    Handle incoming messages and respond with answers.
    """
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()
    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]
    text_elements = []

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

    if source_names:
        answer += f"\nSources: {', '.join(source_names)}"
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()

