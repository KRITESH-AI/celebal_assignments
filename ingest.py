import traceback

from utils.loader import DocumentLoader
from utils.chunker import TextChunker
from utils.embeddings import EmbeddingModel
from utils.vectordb import VectorDatabase


def main():

    print("=" * 60)
    print("DOCUMENT INGESTION MODULE")
    print("=" * 60)

    try:

        loader = DocumentLoader()
        documents = loader.load_documents()

        if not documents:
            print("\nNo valid documents found.")
            return

        print(f"Successfully loaded {len(documents)} document entries.\n")

 
        chunker = TextChunker(
            chunk_size=500,
            chunk_overlap=100
        )

        chunks = chunker.create_chunks(documents)

        if not chunks:
            print("\n No chunks were created.")
            return

        chunker.print_statistics(chunks)


        embedding_model = EmbeddingModel()
        embeddings = embedding_model.create_embeddings(chunks)

        vector_db = VectorDatabase()

        vector_db.build_index(embeddings)

        vector_db.save(chunks)

        print("=" * 60)
        print("INGESTION COMPLETED SUCCESSFULLY")
        print("=" * 60)

    except Exception:

        print("\nINGESTION FAILED")
        print("-" * 60)

        traceback.print_exc()

        print("-" * 60)


if __name__ == "__main__":
    main()