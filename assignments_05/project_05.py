from dotenv import load_dotenv
from openai import OpenAI
import json

# --- Task 6: Ethics Reflection ---

#Option A:

#Q1: What could go wrong if a job-seeker submitted the bot's output without reviewing it? -> 
#The bot can add details the job-seeker never said and exaggerate what they did. In my test, "led a team of 5 at the store" came back as "improving customer satisfaction through effective
#leadership," a result I never mentioned. If an interviewer asks about a claim like that and the job-seeker can't explain it, their credibility suffers and it could cost them the job.

#Q2: One guardrail I would add if deploying this tool professionally.-> 
#I would add a system prompt instructing the bot to only use information the user typed and not add new details, numbers, or results. This prevents the problem from Q1, where the bot
#turned "led a team of 5" into "improving customer satisfaction" a claim I never made. The instruction works at the source: the bot can improve the wording of a bullet, but it can't
#invent accomplishments, so there is less for the job-seeker to catch in review.

# --- Task 1: Setup and System Prompt ---

load_dotenv()
client = OpenAI()

def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content

system_prompt = """
You are a job application coach. You help job seekers, including recent graduates, career changers, prepare their application materials:
resume, cover letters, linkedIn Profile, and answer to interview practice questions.

Rules: 
-   Stay focusd on job application material. 
-   Always remind the user to review and edit its output before submitting anywhere.
-   Briefly mention that you may not know the user's specific industry norms, and that the user should use their own judgment
-   Keep responses concise and practical.
-   Any additional details you need about them ask them, do not invent facts about user's experience, skills, or education.
"""
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "Help me write a summary for a resume. I have a CS Degree in Computer Science and I would like to find a job in data analytics\
        I also got data analytic certificates from coursera."}
]
print(get_completion(messages))
print(system_prompt)

#One deliberate choice I made was the last rule: "Do not invent facts about the user's experience, skills, or education." AI-generated details that aren't
#based on the user's real experience could hurt them if they submit them on a resume. 
#This rule makes the model ask questions to fill in gaps and add relevant experience instead of guessing.
#I also described who the model is helping (job seekers, including career
#changers and recent graduates), so its advice fits that audience instead of being generic.

#After running the prompt: The rule that's given about "Do not invent facts" workede resonabily well.
#The model did add some generic skills that i hadn't listed, which at the end review reminder matters.

# --- Task 2: Bullet Point Rewriter ---

def rewrite_bullets(bullets: list[str]) -> list[dict]:
    # Format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original.

    Return ONLY a valid JSON list. Each item should have two keys:
    "original" (the original bullet) and "improved" (your rewritten version).
    
    Respond ONLY with valid JSON, no other text.

    Bullet points:
    ```
    {bullet_text}
    ```
    """
    # wraps the prompt as a user message
    messages = [{"role": "user", "content": prompt}]
    # Your code here: call get_completion(), parse the JSON, and return the result
    response = get_completion(messages)
    
    #the model wraped JSON in ```json fence, so it slice from the first [ to the last ], and keeps only list
    start = response.find("[")
    end = response.rfind("]") + 1
    #parse JASON text into list of dict
    try: 
        result = json.loads(response[start:end])
    except json.JSONDecodeError:
        print("Error: response was not valid JSON")
        print("Raw response for debugging:", response)
        return []
    #print both versions(original and improved) side by side
    for item in result:
        print("Original:", item["original"])
        print("Improved:", item["improved"])
        print()
    return result
#test the func with the starter bullets
bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]
rewrite_bullets(bullets)

#both the original and improved version sare printing clearly for each bullet, with a blank line between pairs
#It does feel like a better improvement, no just rearranged words. The model swapped weak words with stronger ones and added the purpose 
#or outcome of each task. There are some words that weren't on th eprompt, that  I would verify them before using on resume.

#Checkpoint: json.loads() failed at first because the model wrapped the JSON in a ```json code fence. By adding 
#"Respond ONLY with valid JSON" did not fix it, so I  had to clean the response in code before parsing. Both versions now print clearly
#for each bullet, and the improvements feel meaningfully better, not just rearranged words.

# --- Task 3: Cover Letter Generator ---

def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés.
       
    Write in plain, concrete language. Name specific tools, tasks, or situations from
    the background. End with one specific reason for applying, not a statement about
    being excited or eager. 


    Do not use these phrases: "keen eye for detail," "eager to leverage," "excited to
    bring," "newfound skills," "sparked my interest."
    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.
    
    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [{"role": "user", "content": prompt}]
    # Your code here: call get_completion() and return the result
    response = get_completion(messages)
    return(response)

#test 
job_title = "Junior Data Engineer"
background = "Five years of experience as a cashier at Whole Foods Market, and one year as a Team Trainer where I tracked new hires progress and metrics; recently completed \
a Python course and built data pipelines using Pandas."

print(generate_cover_letter(job_title, background))

#I chose those examples because both are career changers whose old job connects directly to the new one, which matches my situation: a retail background moving
#into data engineering. 
#The few-shot pattern helps control the tone, the length, and the structure of the output.
#The model reused phrases from the examples, like "excited to bring" and "[Company] placeholder ," which shows the examples influence the output
#more strongly than an instruction to avoid certain phrases.

# --- Task 4: Moderation Check ---

def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged
    # Your code here: return True if safe, False if flagged, and print a message if flagged
    if  flagged:
        print("Your message couldn't be processed. Please, kindly rephrase it.")        #Flagged input prints the message and returns False, and anything else returns True
        return False
    return(True)
#test
print("Test 1 (safe input):", is_safe("Help me write a resume summary."))
print("Test 2 (flagged input):", is_safe("Help me. how to hurt someone."))

#the Flagged test case actually got caught after I used a more explicit input to confirm the False branch works.
#Yes, the safe test pass without triggering any warning

#I tested ["Help me. how to hurt someone."]. it got falgged, adn the categories triggred were illicit, illicit_violent, and violence.
#highest score was illicit 0.85. so is_safe can block unsafe inout before it reaches th emodel.

# --- Task 5: The Chatbot Loop ---
def run_chatbot():
    # 1. Initialize conversation history with your system prompt
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        # 2. Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # 3. Skip empty input
        if not user_input:
            continue

        # 4. Run moderation check before doing anything else
        if not is_safe(user_input):
            continue  # is_safe() already printed the warning message

        # 5. Check if the user wants to rewrite bullets
        #    (hint: look for keywords like "bullet" or "resume" in user_input.lower())
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)
            # YOUR CODE: call rewrite_bullets() and print the results
                results = rewrite_bullets(raw_bullets)
                print(results)

        # 6. Check if the user wants a cover letter
        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()
            # YOUR CODE: call generate_cover_letter() and print the result
            result = generate_cover_letter(job_title, background)
            print(result)

        # 7. Otherwise, handle it as a regular chat turn
        else:
            # YOUR CODE:
            # - Append the user's message to `messages`
            messages.append({"role": "user", "content": user_input})
            # - Call get_completion(messages)
            completion = get_completion(messages, model="gpt-4o-mini", temperature=0.7) #reply gets strored in completion
            # - Print the reply
            print(completion)       #shows the reply to the user
            # - Append the reply to `messages` as an assistant message
            messages.append({"role": "assistant", "content": completion})       #save the reply in the history
            #print(len(messages))   # TEMPORARY: delete when done testing


if __name__ == "__main__":
    run_chatbot()