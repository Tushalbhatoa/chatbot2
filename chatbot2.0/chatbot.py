from langchain_community.document_loaders import PlaywrightURLLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter

urls = ["https://brainlox.com/courses/category/technical"]
loader = PlaywrightURLLoader(urls = urls)
data = loader.load()

text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)
data = text_splitter.split_documents(data)

embedding = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
vectorestore = Chroma.from_documents(data, embedding, persist_directory= "C:\project-II\chatbot2.0")
