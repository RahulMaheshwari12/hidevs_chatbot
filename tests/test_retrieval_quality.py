
from retrieval.vector_search import VectorSearch


def main():
    vector_search = VectorSearch()

    test_questions = [
        "What is the Agent Builder Challenge and what is its prize pool?",
        "What is CodeQuest and how does it work?",
        "What is LeetPrompt?",
        "Where is AI House located?",
        "What courses are available in AI Academy?",
    ]

    print("\n" + "=" * 80)
    print("RETRIEVAL QUALITY EVALUATION")
    print("=" * 80)

    for question_number, question in enumerate(test_questions, start=1):

        print("\n" + "=" * 80)
        print(f"QUESTION {question_number}: {question}")
        print("=" * 80)

        results = vector_search.search(question)

        if not results:
            print("No results retrieved.")
            continue

        for rank, document in enumerate(results, start=1):
            print(f"\nRESULT {rank}")
            print(f"Score: {document.metadata.get('score', 'N/A')}")
            print(f"Source: {document.metadata.get('source', 'N/A')}")
            print(f"Point ID: {document.metadata.get('point_id', 'N/A')}")

            content = document.page_content.strip()

            print("\nContent:")
            print(content[:700])

            if len(content) > 700:
                print("... [content truncated for display]")


if __name__ == "__main__":
    main()
