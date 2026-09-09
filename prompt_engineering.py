"""
PROMPT ENGINEERING LAB
======================

This file is meant to be READ, RUN, and EDITED.

You are learning LangChain + LLM prompting, so do not treat this as a
normal application. Each section is a small experiment.

The main idea:

    Prompt quality -> better instructions -> more predictable output

You will learn these concepts by actually changing prompts and seeing
what the model does.

Topics covered:
    1. What a prompt really is
    2. Zero-shot prompting
    3. Few-shot prompting
    4. Weak prompts vs clear prompts
    5. Giving the model a role
    6. Giving context
    7. Defining the output format
    8. Constraints
    9. Separating instructions from data
    10. Iterative prompt improvement
    11. Interactive experiments

IMPORTANT:
This file intentionally contains "bad" prompts too.
Do not remove them. They are here so you can compare the results.


"""

# ============================================================
# 1. IMPORTS
# ============================================================

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
import os


# ============================================================
# 2. LOAD THE API KEY
# ============================================================


load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env")


# ============================================================
# 3. CREATE THE MODEL
# ============================================================

#
# The important thing for this lesson is not the model itself.
# We are going to keep the model mostly constant and change ONLY
# the prompt.
#
# That makes the experiment useful:
#
#       same model
#       different prompt
#       different result
#
# This helps you understand prompt engineering rather than
# accidentally changing multiple things at once.

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key,
)


# ============================================================
# 4. A SMALL HELPER FUNCTION
# ============================================================

def ask(prompt, system_instruction=None):
    """
    Send a prompt to the model and return only the text response.

    If system_instruction is provided, it becomes a SystemMessage.
    The actual question becomes a HumanMessage.

    This lets you clearly see the difference between:

        SystemMessage -> how the assistant should behave
        HumanMessage  -> what the user wants right now
    """

    messages = []

    if system_instruction:
        messages.append(
            SystemMessage(content=system_instruction)
        )

    messages.append(
        HumanMessage(content=prompt)
    )

    response = llm.invoke(messages)

    return response.content


def show(title, result):
    """Print an experiment in a readable way."""

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(result)
    print()


# ============================================================
# 5. THE MOST BASIC IDEA: A PROMPT IS AN INSTRUCTION
# ============================================================

"""
A prompt is simply the input you give to the language model.

For example:

    "Explain HTTP."

That is a valid prompt.

But it leaves many things undefined:

    - Who is the explanation for?
    - How much detail?
    - Should examples be included?
    - What terminology should be used?
    - What should the output look like?

The model has to guess.

A better prompt reduces the amount of guessing.

Think of prompting as giving a task to another person.

Bad:

    "Explain networking."

Better:

    "Explain computer networking to a beginner.
     Use simple language and give one real-world example."

The second prompt contains more useful information.
"""


# ============================================================
# 6. ZERO-SHOT PROMPTING
# ============================================================

"""
ZERO-SHOT = give the model a task without giving examples.

You tell the model what you want, but you do not show it examples
of the desired answer.

Example:

    "Classify this review as positive or negative:
     'The laptop is excellent and very fast.'"

There is no example before the task.

That is zero-shot prompting.

Run this section and observe the output.
"""

def zero_shot_demo():
    prompt = """
Classify the following review as either POSITIVE or NEGATIVE.

Review:
"The battery lasts all day and the screen looks great."

Return only the classification.
"""

    result = ask(prompt)

    show("ZERO-SHOT PROMPTING", result)


# ============================================================
# 7. FEW-SHOT PROMPTING
# ============================================================

"""
FEW-SHOT = give the model examples before asking it to perform
the actual task.

You are showing the model a pattern.

Example:

    Review: "The phone is amazing."
    Classification: POSITIVE

    Review: "The phone is terrible."
    Classification: NEGATIVE

    Review: "The phone is fast and reliable."
    Classification: ?

The first two are examples.
The third is the real task.

This is called few-shot prompting.

The examples teach the model:

    input -> expected output

Few-shot prompting is especially useful when your desired output
format or classification rule is not obvious.
"""

def few_shot_demo():
    prompt = """
Classify reviews as POSITIVE or NEGATIVE.

Example 1:
Review: "The camera is fantastic."
Classification: POSITIVE

Example 2:
Review: "The battery is terrible."
Classification: NEGATIVE

Now classify this review:

Review: "The laptop is fast and reliable."
Classification:
"""

    result = ask(prompt)

    show("FEW-SHOT PROMPTING", result)


# ============================================================
# 8. ZERO-SHOT VS FEW-SHOT
# ============================================================

"""
The key difference:

ZERO-SHOT:
    Instruction only.

FEW-SHOT:
    Instruction + examples.

Use zero-shot when the task is already clear.

Use few-shot when you want the model to follow a particular
pattern that is difficult to describe with words alone.
"""

def compare_zero_and_few_shot():
    print("\nZERO-SHOT RESULT")
    print("-" * 70)

    zero_prompt = """
Convert the following sentence into a formal sentence:

"Hey, send me the report ASAP."
"""

    print(ask(zero_prompt))

    print("\nFEW-SHOT RESULT")
    print("-" * 70)

    few_prompt = """
Convert informal messages into formal messages.

Example:
Informal: "Hey, send me the report ASAP."
Formal: "Please send me the report at your earliest convenience."

Example:
Informal: "Can you fix this quickly?"
Formal: "Could you please resolve this issue as soon as possible?"

Now convert:
Informal: "Give me the details tomorrow."
Formal:
"""

    print(ask(few_prompt))


# ============================================================
# 9. WEAK PROMPT VS STRONGER PROMPT
# ============================================================

"""
This is one of the most useful exercises in prompt engineering.

A weak prompt:

    "Tell me about TCP."

There is nothing technically wrong with it.

But it is underspecified.

A stronger prompt defines:

    ROLE
    AUDIENCE
    TASK
    SCOPE
    FORMAT
    CONSTRAINTS

For example:

    "You are a networking tutor.
     Explain TCP to a beginner.
     Explain why TCP uses a handshake.
     Use one real-world analogy.
     Keep the answer under 200 words."

Now the model has much less room to guess.
"""

def weak_vs_strong():
    weak_prompt = "Tell me about TCP."

    strong_prompt = """
You are a computer networking tutor.

Explain TCP to a beginner who understands basic IP addressing
but has never studied TCP deeply.

Cover only:
1. What TCP is
2. Why TCP uses a connection
3. What the three-way handshake does

Use simple language.
Give one real-world analogy.
Keep the answer under 200 words.
"""

    show("WEAK PROMPT", ask(weak_prompt))
    show("STRONGER PROMPT", ask(strong_prompt))


# ============================================================
# 10. ROLE / PERSONA
# ============================================================

"""
You can tell the model what role it should take.

For example:

    "You are a Python tutor."

This does NOT magically make the model an expert.
It mainly gives the model useful context about the style and
purpose of the answer.

Compare these two prompts.
"""

def role_demo():
    system_instruction = """
You are a Python tutor teaching a beginner.
Use simple language.
Do not assume advanced programming knowledge.
"""

    prompt = """
Explain what a Python virtual environment is.
Use one simple analogy.
"""

    result = ask(prompt, system_instruction)

    show("ROLE + TASK", result)


# ============================================================
# 11. CONTEXT
# ============================================================

"""
Context tells the model information it should use when completing
the task.

Without context:

    "Write a summary."

Summary of WHAT?

With context:

    "Here is a paragraph about TCP...
     Summarize it in 3 bullet points."

The model now knows what it should summarize.

Context is often one of the biggest differences between a useful
LLM application and a vague chatbot.
"""

def context_demo():
    context = """
TCP is a transport-layer protocol.
It provides reliable, ordered delivery of data.
It uses acknowledgements and retransmission.
It establishes a connection using a three-way handshake.
"""

    prompt = f"""
Using ONLY the context below, explain TCP in 3 bullet points.

CONTEXT:
{context}
"""

    result = ask(prompt)

    show("CONTEXT-BASED PROMPT", result)


# ============================================================
# 12. OUTPUT FORMAT
# ============================================================

"""
If you do not specify a format, the model chooses one.

For example:

    "Give me information about Python."

You might get paragraphs.

If your application needs predictable output, specify the format.

For example:

    "Return exactly this structure:

     Name:
     Definition:
     Example:
     Use case:"

You are not just asking WHAT to answer.
You are also asking HOW to answer.
"""

def output_format_demo():
    prompt = """
Explain Python virtual environments.

Return the answer using exactly these four headings:

Name:
Definition:
Example:
Why it is useful:

Keep each section short.
Do not add any other headings.
"""

    result = ask(prompt)

    show("OUTPUT FORMAT", result)


# ============================================================
# 13. CONSTRAINTS
# ============================================================

"""
Constraints limit the model.

Examples:

    - "Use exactly 3 bullet points."
    - "Use less than 100 words."
    - "Do not use technical jargon."
    - "Return only JSON."
    - "Give exactly one example."

Constraints are useful because LLMs naturally generate flexible
answers.

If your application needs consistency, constraints become useful.
"""

def constraints_demo():
    prompt = """
Explain DNS.

Rules:
- Exactly 3 bullet points.
- Each bullet must be one sentence.
- Use beginner-friendly language.
- Include exactly one real-world analogy.
"""

    result = ask(prompt)

    show("CONSTRAINTS", result)


# ============================================================
# 14. SEPARATE INSTRUCTIONS FROM DATA
# ============================================================

"""
A useful prompt structure is:

    INSTRUCTIONS
    ------------
    What should the model do?

    DATA
    ----
    What should the model work on?

For example:

    INSTRUCTIONS:
    Summarize the text in 3 bullet points.

    DATA:
    <user's text>

This becomes especially useful when the data comes from users,
files, websites, databases, or APIs.

Keeping the two conceptually separate makes prompts easier to
read, debug, and maintain.
"""

def instruction_and_data_demo():
    user_text = """
LangChain is a framework for building applications powered by
language models. It provides abstractions for models, prompts,
message handling, tools, retrieval, and agents.
"""

    prompt = f"""
INSTRUCTIONS:
Summarize the DATA below in exactly 3 bullet points.
Use simple language.

DATA:
{user_text}
"""

    result = ask(prompt)

    show("INSTRUCTIONS + DATA", result)


# ============================================================
# 15. PROMPT ITERATION
# ============================================================

"""
Good prompt engineering is usually iterative.

You write:

    Version 1
        ↓
    Run it
        ↓
    Observe the problem
        ↓
    Change the prompt
        ↓
    Run it again
        ↓
    Compare

Do not randomly add huge instructions.

Instead, identify the actual problem.

Example:

Version 1:
    "Explain Python."

Problem:
    The answer may be too broad.

Version 2:
    "Explain Python to a beginner."

Problem:
    Still potentially too long.

Version 3:
    "Explain Python to a beginner in 5 bullet points."

Problem:
    Maybe you need examples.

Version 4:
    "Explain Python to a beginner in 5 bullet points.
     Include one short code example."

Prompt engineering is often this process.
"""


def prompt_iteration_demo():
    prompts = [
        """
Explain Python.
""",
        """
Explain Python to a beginner in simple language.
""",
        """
Explain Python to a beginner in exactly 5 bullet points.
""",
        """
Explain Python to a beginner in exactly 5 bullet points.
Include one short code example.
Do not discuss advanced topics.
"""
    ]

    for number, prompt in enumerate(prompts, start=1):
        print("\n" + "#" * 70)
        print(f"PROMPT VERSION {number}")
        print("#" * 70)
        print(prompt.strip())
        print("\nMODEL:")
        print(ask(prompt))


# ============================================================
# 16. AN IMPORTANT LESSON:
#     MORE PROMPT DOES NOT ALWAYS MEAN BETTER PROMPT
# ============================================================

"""
Do NOT assume that the longest prompt is automatically the best.

Compare:

    "Explain TCP."

with:

    "You are the world's greatest networking expert, teacher,
     researcher, engineer, scientist, professor..."

That extra text may not help.

A good prompt gives the model the information it actually needs.

Think:

    Clear > Long
    Specific > Vague
    Relevant context > Random instructions
    Useful examples > Unnecessary examples

Your goal is not to write the biggest prompt.

Your goal is to remove ambiguity.
"""


# ============================================================
# 17. YOUR FIRST REAL EXERCISE
# ============================================================

"""
EXERCISE 1
----------

Start with this:

    "Explain HTTP."

Then improve it yourself.

Try to add:

    1. Audience
    2. Scope
    3. Format
    4. Example
    5. Constraint

A possible final prompt could be:

    "You are a networking tutor.
     Explain HTTP to a beginner.
     Cover request, response, methods, and status codes.
     Use 5 bullet points.
     Give one simple browser example.
     Keep it under 200 words."

But do NOT simply copy that.

Write your own version first.

The point is to practice designing the prompt.
"""

def exercise_1():
    print("\nWrite your own prompt for this task:")
    print("TASK: Explain HTTP to a beginner.")
    print("\nYou can use these ideas:")
    print("  - Role")
    print("  - Audience")
    print("  - Context")
    print("  - Output format")
    print("  - Constraints")
    print()

    prompt = input("Your prompt: ")

    if not prompt.strip():
        print("No prompt entered.")
        return

    print("\nMODEL RESPONSE:")
    print(ask(prompt))


# ============================================================
# 18. YOUR SECOND EXERCISE: ZERO-SHOT -> FEW-SHOT
# ============================================================

"""
EXERCISE 2
----------

Task:

    Convert technical explanations into beginner-friendly
    explanations.

First, write a ZERO-SHOT prompt.

Then change it into a FEW-SHOT prompt.

The important thing is to notice what the examples change.

For example, your few-shot prompt could contain:

Example:
Technical: "DNS resolves domain names to IP addresses."
Beginner: "DNS works like the internet's phonebook."

Then give the model a new technical sentence.
"""

def exercise_2():
    print("\nEXERCISE 2")
    print("-" * 70)

    prompt = input(
        "\nWrite a prompt that converts this into beginner language:\n"
        "\"TCP uses acknowledgements to provide reliable delivery.\"\n\n"
        "Your prompt: "
    )

    if not prompt.strip():
        print("No prompt entered.")
        return

    print("\nMODEL RESPONSE:")
    print(ask(prompt))

    print("\nNow ask yourself:")
    print("Would examples make the expected style clearer?")


# ============================================================
# 19. YOUR THIRD EXERCISE: OUTPUT CONTROL
# ============================================================

"""
EXERCISE 3
----------

Create a prompt that asks the model to explain:

    "What is an API?"

But your answer must contain:

    Definition:
    Real-world analogy:
    Technical example:
    One common mistake:

Try to make the output predictable.
"""

def exercise_3():
    prompt = """
Explain what an API is.

Return exactly these sections:

Definition:
Real-world analogy:
Technical example:
One common mistake:

Use simple language.
"""

    print("\nEXERCISE 3")
    print("-" * 70)
    print("\nStarter prompt:")
    print(prompt)

    print("\nMODEL RESPONSE:")
    print(ask(prompt))


# ============================================================
# 20. INTERACTIVE PROMPT LAB
# ============================================================

"""
This is the part you should spend time with.

You can enter ANY prompt.

Try changing only one thing at a time.

For example:

Run 1:
    Explain REST API.

Run 2:
    Explain REST API to a beginner.

Run 3:
    Explain REST API to a beginner in 5 bullet points.

Run 4:
    Explain REST API to a beginner in 5 bullet points.
    Include one example.

Run 5:
    Explain REST API to a beginner in 5 bullet points.
    Include one example.
    Keep it under 150 words.

Then compare the outputs.

That is real prompt engineering practice.
"""

def interactive_lab():
    print("\n" + "=" * 70)
    print("INTERACTIVE PROMPT LAB")
    print("=" * 70)
    print("Type a prompt and see what the model does.")
    print("Type 'back' to return to the menu.\n")

    while True:
        prompt = input("Prompt > ")

        if prompt.lower().strip() == "back":
            break

        if not prompt.strip():
            continue

        print("\nAI:")
        print(ask(prompt))
        print()


# ============================================================
# 21. MAIN MENU
# ============================================================

"""
You do not have to run every experiment every time.

Use the menu to choose one.

Recommended learning order:

    1 -> Understand zero-shot
    2 -> Understand few-shot
    3 -> Compare them
    4 -> Weak vs strong prompts
    5 -> Role
    6 -> Context
    7 -> Output format
    8 -> Constraints
    9 -> Instructions + data
    10 -> Prompt iteration
    11-13 -> Do the exercises
    14 -> Free experimentation

After you understand the basics, spend most of your time in
the interactive lab.
"""

def main():
    while True:
        print("\n")
        print("=" * 70)
        print("PROMPT ENGINEERING LAB")
        print("=" * 70)
        print("1. Zero-shot prompting")
        print("2. Few-shot prompting")
        print("3. Compare zero-shot vs few-shot")
        print("4. Weak prompt vs stronger prompt")
        print("5. Role / system instruction")
        print("6. Context")
        print("7. Output format")
        print("8. Constraints")
        print("9. Instructions + data")
        print("10. Prompt iteration")
        print("11. Exercise: improve a prompt")
        print("12. Exercise: zero-shot -> few-shot")
        print("13. Exercise: control output format")
        print("14. Interactive prompt lab")
        print("0. Exit")
        print("=" * 70)

        choice = input("Choose an experiment: ").strip()

        if choice == "1":
            zero_shot_demo()
        elif choice == "2":
            few_shot_demo()
        elif choice == "3":
            compare_zero_and_few_shot()
        elif choice == "4":
            weak_vs_strong()
        elif choice == "5":
            role_demo()
        elif choice == "6":
            context_demo()
        elif choice == "7":
            output_format_demo()
        elif choice == "8":
            constraints_demo()
        elif choice == "9":
            instruction_and_data_demo()
        elif choice == "10":
            prompt_iteration_demo()
        elif choice == "11":
            exercise_1()
        elif choice == "12":
            exercise_2()
        elif choice == "13":
            exercise_3()
        elif choice == "14":
            interactive_lab()
        elif choice == "0":
            print("Goodbye!")
            break
        else:
            print("Choose a number from 0 to 14.")


# ============================================================
# 22. PROGRAM ENTRY POINT
# ============================================================

# Python executes main() only when this file is run directly.
#
# Run it with:
#
#     python prompt_engineering_lab.py
#
# You can also import functions from this file later without
# automatically starting the menu.

if __name__ == "__main__":
    main()
