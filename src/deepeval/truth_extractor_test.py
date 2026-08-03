from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from deepeval.metrics import ContextualPrecisionMetric, FaithfulnessMetric
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase
from deepeval import evaluate

judge_model = OllamaModel(model="jeffnyman/ts-evaluator")

metric = FaithfulnessMetric(
    model=judge_model,
    async_mode=False,
    verbose_mode=True,
    truths_extraction_limit=None,
)

def create_rag_system(chunk_size=1000, chunk_overlap=200, k=3):
  """Create a RAG system with configurable parameters."""
  loader = PyPDFLoader("./jeff-nyman-extinction-paper.pdf")
  documents = loader.load()

  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
  )

  chunks = text_splitter.split_documents(documents)

  embeddings = OllamaEmbeddings(model="nomic-embed-text")
  vectorstore = Chroma.from_documents(chunks, embeddings)

  retriever = vectorstore.as_retriever(search_kwargs={"k": k})

  return retriever, len(chunks)

# =========================================================
# Setup baseline RAG system
# =========================================================
print("=" * 60)
print("BASELINE CONFIGURATION: chunk_size=1000, chunk_overlap=200, k=3")
print("=" * 60)

retriever, num_chunks = create_rag_system(
  chunk_size=1000,
  chunk_overlap=200,
  k=3
)

# Get relevant context
question = """How does Jeff Nyman's model explain the relationship
between coevolutionary avalanches and mass extinctions?"""
retrieved_docs = retriever.invoke(question)
retrieval_context = [doc.page_content for doc in retrieved_docs]


print("\nAll chunks:")
print(metric._generate_truths(retrieval_context, multimodal=False))

print("\nEach chunk separately:")
for index, chunk in enumerate(retrieval_context, start=1):
    truths = metric._generate_truths([chunk], multimodal=False)
    print(f"Chunk {index}:")
    print(truths)

print("\nReversed chunk order:")
print(
    metric._generate_truths(
        list(reversed(retrieval_context)),
        multimodal=False,
    )
)