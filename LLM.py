import vertexai
import langchain
from langchain.llms import VertexAI
from langchain.indexes import VectorstoreIndexCreator
import time
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import VertexAIEmbeddings
from typing import List
from pydantic import BaseModel

PROJECT_ID = "nchq-llm-nonprod-experiment"  # @param {type:"string"}
vertexai.init(project=PROJECT_ID, location="us-central1")

llm = VertexAI(
    model_name="text-bison@001",
    max_output_tokens=256,
    temperature=0.1,
    top_p=0.8,
    top_k=40,
    verbose=True,
)

def rate_limit(max_per_minute):
  period = 60 / max_per_minute
  while True:
    before = time.time()
    yield
    after = time.time()
    elapsed = after - before
    sleep_time = max(0, period - elapsed)
    if sleep_time > 0:
      print(f'Sleeping {sleep_time:.1f} seconds')
      time.sleep(sleep_time)

class CustomVertexAIEmbeddings(VertexAIEmbeddings, BaseModel):
      requests_per_minute: int
      num_instances_per_batch: int

      # Overriding embed_documents method
      def embed_documents(self, texts: List[str]):
        limiter = rate_limit(self.requests_per_minute)
        results = []
        docs = list(texts)

        while docs:
          # Working in batches because the API accepts maximum 5
          # documents per request to get embeddings
          head, docs = docs[:self.num_instances_per_batch], docs[self.num_instances_per_batch:]
          chunk = self.client.get_embeddings(head)
          results.extend(chunk)
          next(limiter)

        return [r.values for r in results]

EMBEDDING_QPM = 100
EMBEDDING_NUM_BATCH = 5

embeddings = CustomVertexAIEmbeddings(
    requests_per_minute=EMBEDDING_QPM,
    num_instances_per_batch=EMBEDDING_NUM_BATCH,
)

loader = DirectoryLoader('abstracts', glob="/*.txt")

index = VectorstoreIndexCreator(embedding = embeddings,
                                vectorstore_kwargs = {'persist_directory' : '/Content/ChromaDB',}).from_loaders([loader]);

index.query("Who spoke about HuggingFace?", llm = llm)