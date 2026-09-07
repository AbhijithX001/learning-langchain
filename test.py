from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
import os


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env")


# --------------------------------------------------
# 2. Create the Gemini model
# --------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key,
)


# --------------------------------------------------
# 3. Store conversation history
# --------------------------------------------------

messages = [
    SystemMessage(
        content="You are a helpful AI assistant. "
                "Give clear and simple answers."
    )
]


# --------------------------------------------------
# 4. Chat loop
# --------------------------------------------------

print("Chat started.")
print("Type 'exit' or 'quit' to stop.\n")

while True:

    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    # Add user's message to conversation
    messages.append(
        HumanMessage(content=user_input)
    )

    # Send entire conversation to Gemini
    response = llm.invoke(messages)

    # Add AI response to conversation
    messages.append(response)

    # Print only the actual text
    print("AI:", response.content)
    print()