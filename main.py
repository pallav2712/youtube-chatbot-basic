from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import TranscriptsDisabled, YouTubeTranscriptApi

load_dotenv()


# 1. Transcript nikalo
video_id = "Gfr50f6ZBvo"  # only the ID, not full URL

try:
    ytt_api = YouTubeTranscriptApi()
    fetched = ytt_api.fetch(video_id, languages=["en"])

    # Flatten it to plain text
    transcript = " ".join(snippet.text for snippet in fetched)
    

except TranscriptsDisabled:
    print("No captions available for this video.")


# 2. Chunks me todo
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.create_documents([transcript])


# 3. Embeddings (local, free) + FAISS vector store
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store = FAISS.from_documents(chunks, embedding_model)
retriever = vector_store.as_retriever(search_kwargs={"k": 4})


# 4. LLM (Groq)
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2)

# 5. Prompt
prompt = PromptTemplate(
    template=(
        "You are a helpful assistant.\n"
        "Answer ONLY from the transcript context below.\n"
        "If the context is not enough, say you don't know.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    ),
    input_variables=["context", "question"],
)


from langchain_core.output_parsers import StrOutputParser


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


question = "What is the video about?"
docs = retriever.invoke(question)
final_prompt = prompt.invoke({"context": format_docs(docs), "question": question})
answer = (llm | StrOutputParser()).invoke(final_prompt)
print(answer)
