from dotenv import load_dotenv
import os

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex


# --- RAG Concepts ---

#Concepts Question 1

#Scenario A: RAG
#I chose RAG because the assistant needs to answer questions from the team's own PDFs, 
#and since those PDFs are updated every quarter, RAG can simply pull from
#the newest versions instead of needing to be retrained.

#Scenario B: Fine-tunning
#The startup wants a rare, dry, minimalist brand voice, which is a matter of style rather than facts. 
#They already have 3,000 in-house examples, which are ideal training data to teach the model to write in that voice.

#Scenario C: Prompt-Engineering
#The report is only two pages and is needed just once, so she can paste it into the prompt along with her question and get the answer directly. 
# Since she doesn't need this to work for any other document, the model doesn't need to be trained(fine-tuning) or have documents stored and searched(RAG).

#Concepts Question 2
#A confidently wrong answer is more harmful than "I am not sure" because people use the tone of a response to decide how much to trust it. 
#If the model admits uncertainty, the person knows to double-check; if it sounds confident, they have no reason to question it and may act on wrong information.
#Example: Someone asks a model what dose of a medicine works best for a fever. The model is not a doctor and does not know that patient's age, weight, or
#health conditions, so a confident but wrong dose could put them in danger.
#Tone matters because a hallucination is written in the same fluent, authoritative style as a correct answer, so nothing in the wording warns the reader.

#Concepts Question 3

"""
steps = [
    
    "Extract text from source documents",
    "Split text into chunks",
    "Convert text chunks into embeddings",
    "Receive the user's query",      
    "Embed the user's query",
    "Retrieve the most relevant chunks",
    "Inject retrieved chunks into the prompt",
    "Generate a response from the LLM",  
]
"""

# --- Keyword RAG ---

import string

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]
    

# Keyword Question 1:

query = "What are your hours on weekends?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}
#the search step: cleans query into keywords, compare them to each document's words, and return the best match
results = simple_keyword_retrieval(query, documents, verbose=True)
print("Selected document:", results[0][0])

#The lotalty.txt document got selected because, which is the wrong one. 
#he query tokens ("what", "your", "hours", "weekends"). hours.txt matched only "weekends", hiring.txt and loyalty.txt only matched "your".
#so all three documents tied with a score of 1. The tie was broken by sorting the names in reverse alphabetical order, therefore
#loyalty.txt came first. It counted the shared words without understanding meaning, and "your" is a common word that should have been treated as a stopword.

# Keyword Question 2

query2 = "Do you have anything without caffeine?"
print("\nQuery 2")
documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}
#the search step: cleans query into keywords, compare them to each document's words, and return the best match
results = simple_keyword_retrieval(query2, documents, verbose=True)
print("Q2 Selected document:", results[0][0])

#No document was selected. It printed "None Found", because non of the query words ['anything', 'caffeine', 'do', 'have', 'without'] appear in any document.
#menu.txt is the document that should have been retrieved, since it lits the coffee drins and food the shop serves, but it shares no exact keyword with the query, so the 
#keyword matching couldn't match them. Therefore, Keyword RAG got this wrong.
#Semantic retrival query using embedding would do better, because it compares related meaning rather than exact words and would be able to find documents about caffeine to the coffee drinks
#and menu items in the menu.txt. 

# Keyword Question 3

query3 = "How do I sign up for rewards?"

print("\nQuery 3")
documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}
#the search step: cleans query into keywords, compare them to each document's words, and return the best match
results = simple_keyword_retrieval(query3, documents, verbose=True)
print("Q3 Selected document:", results[0][0])

#Prediction(before running): loyalty.txt: because it talks about points and a loyalty program, which is close in meaning to "rewards".
#Result: My prediction was not correct! No document was selected because every document had an overlap of 0. This is similar to Query 2:
#loyalty.txt has the answer, but it says "join" and "loyalty program" while the query says "sign up" and "rewards". The meaning is the same, but keyword
#matching only sees identical words, so it found no overlap in any document. Semantic retrieval would connect these phrases by meaning.

# --- Semantic RAG Concepts ---

#Semantic Question 1

#1. A vector embedding is a list of numbers representing the meaning of text.
#2. The chunk with 0.85 is more relevant because it is close to 1. The number tells us that the relationship between the texts is very similar in meaning to the query.
#3. Semantic search compares meaning instead of exact words because it uses vector embeddings, which are a mathematical way of finding meanings that are close.

# Semantic Question 2

"""
| Feature                    | Keyword RAG                       | Semantic RAG                         |
|----------------------------|-----------------------------------|--------------------------------------|                        
| What is compared?          | Exact word overlap                | Meaning (query vs. chunk embeddings) |
| What is retrieved?         | Full document                     | Top matching chunk                   |
| Can it handle synonyms?    | No                                | Yes                                  |
| Storage format             | Plain text dictionary             | Vectors(list of numbers)             |
| Relevance score            | Number of overlapping keywords    | Cosine similarity                    |
"""

# --- LlamaIndex ---

#LlamaIndex Question 1

#the embedding model
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

documents = SimpleDirectoryReader("./brightleaf_pdfs").load_data()
print(len(documents))

#LlamaIndex Question 1

#loads every pdfs in the folder into LlamaIndexc Doc objects
docs = SimpleDirectoryReader("brightleaf_pdfs").load_data()

#vector index automatically handles chunking + embeddings + stores it
index = VectorStoreIndex.from_documents(docs)

#creates a query engine that retrieves the 3 most similar chunks
#for each question and passes them to LLM as a context for its answer
query_engine = index.as_query_engine(similarity_top_k=3)

questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]
#runs each question through the pipeline
for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)        #enbeds the question, finds the top 3 chunks, & has the LLM answer using the chunks
    print(f"\nA:", response)
    
    #response.source_nodes holds the 3 retrived chunks with their scores
    for node_with_score in response.source_nodes:
        text = node_with_score.node.get_content()[:150].replace("\n", " ")      #keeps on the first 150 characters
        #cosine similarity between the question and the chunk higher the closer
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text snippet: { text}...")
        print("_" * 30)

#Yes, mostly the retrieved chunks look relevent to the question.
#For each question the top ranked chunk was relevant (for Q1, benefit into, 0.7751) and for Q2: "Network and Data Security, 0.7031"
#There is †wo other chunks in each query that were weaker matches around 0.57-0.62 and not really related to the queston, such as the 
#BrightLeaf's collaboration with EcoVolt Energy and the company overview.

#The model's response sound confident and specific. They gace very detailed info on Employee's benefits such as 
#401k matche up to 5%, 90-day credential rotation, and did not use phrases like "based on context" or "I am not sure." 

#No, nothing unexpected got retrieved. The off-topic chunks ranked lowest, probably because it did not seem to 
#affect the answer. They are possibly the next closet chunk, since top_k i sakways 3 even when only one chunk is a strong match.

### LlamaIndex Question 2

# Re-run the queries from Q1: once with similarity_top_k=1 and once with similarity_top_k=5
# similarity_top_k=1 : The model only saw the benefits section(score 0.7752), so the answer was to the point with details.

#similarity_top_k=5 : The answers were about the same , with a few extra health details. Since relevent chunk is retrieved
#extra low scoring chunks add no useful info, it rather costs more tokens and risks distraction the model with unrelated text.

### LlamaIndex Question 3

#Query :     "What is BrightLeaf's policy on pets in the office?",
#Retrieval was weak, the top score was 0.57, and other chunks around 0.43.
#The chunks are nothing about pets. The pipeline still returned 3 chunks because k=3, yet nothing related to the pets.
#The model said "he provided information does not include any details regarding BrightLeaf's policy on pets in the office." instead of hallucinating
#Then it means tha the pipeline handled an unanswerable question well. I had expected if the pipeline can't answer, that it would hallunciate.

### LlamaIndex Question 4

#create judge llm
llm = OpenAI(model="gpt-4o-mini", temperature=0.2)

#define evaluator
#FaithfulnessEvaluator and RelevancyEvaluator to compute the faithfulness 
#and relevancy metrics respectively for the given RAG framework
faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
relevancy_evaluator = RelevancyEvaluator(llm=llm)

#get a response to a query
queries = [
        "What employee benefits does BrightLeaf offer?",
        "What is BrightLeaf's policy on pets in the office?",
]

for q in queries:
    response = query_engine.query(q)

    # Evaluate faithfulness and relevancy
    faith_result = faithfulness_evaluator.evaluate_response(query=q, response=response)
    rel_result = relevancy_evaluator.evaluate_response(query=q, response=response)
    
    print(f"\nQ: {q}")
    print("Faithfulness Evaluation: " + str(faith_result.score))
    print("Relevancy Result: " +str(rel_result.score))
    
#Faithfulness: A score of 1.0 means the response is fully supported by the retrieved context: every claim can be traced back to the chunks, so nothing
#was invented. A score of 0.0 means the response is not supported by the retrieved context, so the model may have invented details or gone beyond
#what the chunks say.

#Relevancy measures whether the retrieved context and the response actually answer the user's question. Faithfulness checks the response against the
#retrieved context, while relevancy checks the query against the response and context.

#Yes, the scores changed. The benefits query scored 1.0 on both because the documents contain a benefits section and the answer was grounded in it. The
#pets query scored 0.0 on both because nothing in the documents covers pets. Relevancy failed because the retrieved chunks did not address the question,
#and faithfulness likely failed because the "no information available" answer is not something the context supports.

#LLM-as-a-judge means using a language model to grade another model's output, by giving it the question, the retrieved context, and the answer and
#asking whether the answer is grounded and relevant. It is used instead of a simple accuracy metric because RAG answers have no single exact correct
#string: many different wordings can be right, so exact-match or accuracy would mark good answers as wrong and would not detect hallucination. A judge
#LLM can understand meaning, check claims against the source text, and scale without human written reference answers. 
