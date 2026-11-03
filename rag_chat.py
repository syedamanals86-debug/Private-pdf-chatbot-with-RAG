from rag_core import ask

while True:
    user_question = input("\nAsk about your PDF (or type quit): ")
    if user_question.lower() == "quit":
        break
    print("\nThinking...\n")
    print(ask(user_question))