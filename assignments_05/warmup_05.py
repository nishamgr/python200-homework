                    # --- Part 1: Warmup Exercises ---

# API Question 1
from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()
client = OpenAI()

#To generate chat with responses
response = client.chat.completions.create(
    model="gpt-4o-mini",    #prefered model string name
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]    #alist of messages
)


print("Response text :",response.choices[0].message.content)    #a list of chat responses
print("Model :", response.model)    #the name of the model that generated the response
print("Total Tokens :", response.usage.total_tokens)    #object that tells you about token usage

# API Question 2

prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]
#loop once per temperature value
for temp in temperatures:
    response = client.chat.completions.create(  #seperate API call for this temperature
        model = "gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temp,
    )
    #one completion (the default)
    print(f"Temperature {temp}: {response.choices[0].message.content}\n")
    
    #My observation from running the same prompt 3 times:
    # Temperature 0: gives constently the same consultancy name
    # Temperature 1.5: give different answers the whole 3 times
    # Temperature 0.7 gave same answer twice, and only one different.
    #I would use Temperature 0 for a consistent, reproducible output.

# API Question 3
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
    n=3,
    temperature=1.0
)
# several completions (n=3)
for choice in response.choices:
    print(f"Temperature 1.0: {choice.message.content}\n")

# API Question 4
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain how neural networks work."}],
    max_tokens=15,
)
print({response.choices[0].message.content})

#What happened?-> The response stopped midsentence because max_tokens=15
#limited the reply to 15 tokens, far fewer than a full explanation 
#Why use max_tokens in a real application? -> it saves money, since API usage
#is billed per token. It also keeps responses short and fast, and prevents
#overly long replies.

                    # --- System Messages and Personas ---

# System Question 1
#personality 1: patient python tutor
messages = [
    {"role": "system", "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]
response = client.chat.completions.create(model='gpt-4o-mini', messages = messages)
print(response.choices[0].message.content)

#personality 2: wise prist
messages = [
    {"role": "system", "content": "You are a wise prist who answers in a calm, gently tone. You always encourage things and end with a scripture as a word of encouragement."},
    {"role": "user", "content": "I don't understand God sometimes."}
]
#API call
response = client.chat.completions.create(model='gpt-4o-mini', messages = messages)
print(response.choices[0].message.content)

#What changed? -> Only the system message changed. 
#The user question stayed the same, and the tutor's and prist's answers were  differed in the tone and style, each matching its system prompt.
#This shows that the system message controls the model's personality and style, not just the content of the question.

# System Question 2

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]
#API call
response = client.chat.completions.create(model='gpt-4o-mini', messages = messages)
print(response.choices[0].message.content)

#Why does model know Jordan's name, eventho its stateless? -> The API call doesn't remember anything between calls, But
#the name is given in the list inside messages conversation. The model reads that history as context for this single call, 
#so it can answer from it.

                    # --- Prompt Engineering ---

# Prompt Question 1 — Zero-Shot

#func, wrapppig the API call in a fuc 
#so that we it can be reuse if for every prompt qyestions wihout rewriting the call each time.
def get_completion(prompt: str, model="gpt-4o-mini", temperature=0):    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}], 
        temperature=temperature,
    )
    return response.choices[0].message.content

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]
#loop over reviews
print("Q1: Zero-Shot")
for i, review in enumerate(reviews, start=1):
#zero-shot: instruction only, no examples oof expected outputs
    prompt = (
        "Classify the sentiment of each review below as positive, negative, or mixed.\n\n" 
        "Respond with only one word.\n\n"
        f"Review: {review}"
        )

    result = get_completion(prompt)
    print(f"Review {i}: {result}")
    
# Prompt Question 2 — One-Shot

example = '''Example:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed
'''

print("\nQ2: One-Shot")
for i, review in enumerate(reviews, start=1):
    prompt = (
        f"{example}\n"
        "Classify the sentiment of the review as positive, negative, mixed.\n"
        "Reply in the same format as the example, with only one word.\n"
        f'Review: "{review}"\n'
        "Sentiment:"
    )

    result = get_completion(prompt)
    print(f"Review {i}: {result}")
    
# Adding one example changed the format of the output compared to Q1,
# but the classifications did not change. Both Q1 and Q2 labeled the reviews
# positive, negative, and mixed.
# Q1 returned (Positive, Negative, Mixed). Even though the model saw the labels
# in lowercase in the instruction, it had no example of an answer, so it fell
# back on its default of capitalizing a one word reply.
# Q2 returned (positive, negative, mixed), so the model matched the lowercase
# style of the example by copying that exact format.
# This shows more about format consistency than about accuracy. It suggests that
# giving an example is a stronger format signal than describing the labels in
# the instruction.

# Prompt Question 3 — Few-Shot
#with 3 examples
example = '''Example 1:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed

Example 2:
Review: "Stopped working after few use."
Sentiment: negative

Example 3:
Review: "Works perfectly, loving it."
Sentiment: positive
'''

print("\nQ3: Few-Shot")
for i, review in enumerate(reviews, start=1):
    prompt = (
        f"{example}\n"
        "Classify the sentiment of the review as positive, negative, mixed.\n"
        "Reply in the same format as the example, with only one word.\n"
        f'Review: "{review}"\n'
        "Sentiment:"
    )

    result = get_completion(prompt)
    print(f"Review {i}: {result}")
    
# comparing all three approaches(zero-shot, one-shot, few-shot), 
#All three gave the same classifications(positive, negative, mixed)
#Zero-shot: I wiould choe this for simple tasks, where the model already knows 
#what to do and the exact format doesn't matter.
#One-shot: I would choose this when I mainly need to control the output format wihout having to have a long prompt
#one example was enough to  make the model return lowercase labels.
#few-shot: I would choose this for more subjective tasks, giving several examples shows that the model
#waht each label looks like, so that it can make better choice. The downside is a large prompt.

    
#Prompt Question 4 — Chain of Thought
print("\nQ4 — Chain of Thought")
prompt = """
Show your step-by-step reasoning, then give the final answer on its own line labelled: Final answer: <value>

Problem: A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later
takes a new job that pays $7,500 more per year than her post-raise salary.
What is her final annual salary?
"""

response = get_completion(prompt, temperature=0)
print(response)

#Asking the model to reason step by step tends to improve accuracy on problems like this because it breaks the calculation into smaller steps instead of
#jumping straight to an answer. It also lets me check each step, so I can spot
#mistakes and trust the result more.

# Prompt Question 5 — Structured Output
review = "I've been using this tool for three months. It handles large datasets well, \
but the UI is clunky and the export options are limited."

print("\nQ5 — Structured Output")
prompt = f"""
Analyze the sentiment of this customer review and respond only with valid JSON.
Use eaxctly these keys:
sentiment: positive, negative, or mixed
confidence: a float from 0 to 1
reason: one sentence explaining the sentiment

Return only JSON, with no markdown and no extra text.

Review: "{review}"
"""
#API call
response = get_completion(prompt, temperature=0)
print("Raw Response:", response)

#parse JSON with try/except
try:
    result = json.loads(response)   #converts JSON str into Python dict 
    #prints key anme and labels
    print("Parsed sentiment:", result["sentiment"])
    print("Confidence:", result["confidence"])
    print("Reason:", result["reason"])
except json.JSONDecodeError:
    print("Error: response was not valid JSON")
    print("Raw response for debugging:", response)


# Prompt Question 6 — Delimiters
#prompt 1: text with instructions
user_text = "First boil a pot of water. Once boiling, add a handful of salt and the \
pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."

print("\nQ6 — Delimiters")

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

User:
```{user_text}```
Format your response as:
"No steps provided."
"""
#API call
response = get_completion(prompt, temperature=0)
print("prompt 1 (instructions):")
print(response)

#PROMPT 2: Regular paragraphs, no instructions
user_text = "The weather today was cold with very little sun. A lot of people were wearing big jackets \
    and not many were out."
    
prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""
#API call
response = get_completion(prompt, temperature=0)
print("\nprompt 2 (No instructions):")
print(response)

#when a prompt has multiple sections, instructions, examples, text to analyze, code, the model can lose track
#where one part ends and the next begins.
#The delimiters help prevent the model from confusing the user's text with my instructions.


# ---  Local Models with Ollama --- 
"""
Ollama Q1: output of 
ollama run qwen3:0.6b "Explain what a large language model is in two sentences."

Terminal output: 
A large language model is an AI system designed to understand and generate human-like text, trained on vast amounts of text to 
learn patterns and improve its ability to comprehend and create language. It excels in understanding context and generating 
diverse text, making it invaluable for tasks like writing, translation, or answering questions.
"""


#Ollama Question 1
print("\nQ7 - OpenAI response")

prompt = "Explain what a large language modle is in two sentences."
response = get_completion(prompt, temperature=0)
print(response)


#Comparision: The OpenAI model(gpt-4o-mini) followed the instruction excatly and returned two clear sentences.
#The local model (qwen3:0.6b) gave a similar output in two sentences, but it first showed its thinking and it 
#also added extra third sentence. Which I think, it didn't really follow the two sentences limit instruction
#The advantage of running model locally is that it's private, keeps my data private. Where as open AI needs an API key and internet to run.
#which mean it costs money, on the other hand it fives better output and more consistent answer.