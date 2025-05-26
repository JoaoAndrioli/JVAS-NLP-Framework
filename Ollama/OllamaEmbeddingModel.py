import os
import ollama
import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from chromadb import AdminClient

class OllamaEmbeddingModel:
    def __init__(self, embedding_model: str, answer_model: str, persist_directory: str = "chromadb", database_name: str = "default"):
        """
        Initialize the embedding model instance.

        Args:
            embedding_model (str): Identifier for the Ollama embedding model.
            answer_model (str): Identifier for the Ollama generation model.
            persist_directory (str): Directory where all Chroma data is persisted.
            database_name (str): Name of the database to use (enables multiple isolated databases).
        """
        self.embedding_model = embedding_model
        self.answer_model = answer_model

        # Ensure the persist_directory exists.
        os.makedirs(persist_directory, exist_ok=True)

        # Create a Settings object that uses the given persist_directory.
        admin_settings = Settings(persist_directory=persist_directory, is_persistent=True)

        # Use AdminClient to ensure the requested database exists.
        admin_client = AdminClient(admin_settings)
        try:
            admin_client.get_database(database_name)
        except Exception:
            admin_client.create_database(database_name, DEFAULT_TENANT)

        # Instantiate the PersistentClient using the same persist_directory.
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=admin_settings,
            tenant=DEFAULT_TENANT,
            database=database_name,
        )
        self.collection = self.client.get_or_create_collection(name="documents")

    def createEmbedding(self, text: str):
        """
        Creates an embedding for the provided text using Ollama.

        Args:
            text (str): Text to embed.

        Returns:
            List[float]: The embedding vector.
        """
        response = ollama.embeddings(model=self.embedding_model, prompt=text)
        return response.embedding

    def add_texts(self, texts: list):
        """
        Adds a list of texts to the collection after generating their embeddings.

        Args:
            texts (list): List of text strings.
        """
        ids = []
        embeddings = []
        for i, text in enumerate(texts):
            embedding = self.createEmbedding(text)
            # Using a simple incremental id; adjust as needed.
            ids.append(str(i))
            embeddings.append(embedding)
        self.collection.add(ids=ids, documents=texts, embeddings=embeddings)

    def delete_by_texts(self, texts: list):
        """
        Deletes texts from the collection based on an exact match.

        Args:
            texts (list): List of texts to delete.
        """
        for text in texts:
            self.collection.delete(filter={"documents": text})

    def get_all_texts(self):
        """
        Retrieves all texts stored in the collection.

        Returns:
            list: List of stored texts.
        """
        results = self.collection.get()
        return results.get("documents", [])

    def search(self, query: str, n_results: int = 5):
        """
        Searches for texts similar to the query by comparing embeddings.

        Args:
            query (str): Query text.
            n_results (int): Number of results to return.

        Returns:
            list: The first set of matching documents.
        """
        query_embedding = self.createEmbedding(query)
        results = self.collection.query(query_embeddings=[query_embedding], n_results=n_results)
        return results.get("documents", [])[0]

    def generate_answer(self, question: str):
        """
        Generates an answer for a question by searching for relevant texts and then
        using the Ollama generation model to produce a response.

        Args:
            question (str): The question to answer.

        Returns:
            str: The generated answer.
        """
        # Use the search results to build context.
        documents = self.search(question)
        context = "\n".join(documents)
        prompt = f"Question: {question}\nContext:\n{context}\nAnswer:"
        response = ollama.generate(model=self.answer_model, prompt=prompt)
        return response["response"]
