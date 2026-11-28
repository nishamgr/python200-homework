from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding

from dotenv import load_dotenv
#load API key
if load_dotenv():
    print("sucess")
else:
    print("oops")

# --- Step 1: Setup ---
from pathlib import Path
#direting the folder
docs_dir = Path(__file__).parent/ "resources" / "groundwork_docs"
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"  #returns True if the folder exists, or gives error message if the condition is false

# --- Step 2: Load the Documents ---

# Load documents directly from PDFs in the folder
docs = SimpleDirectoryReader("resources/groundwork_docs").load_data()

#prints no. of documents
print(f"Loaded {len(docs)} documents.")

#goes thru each docs and doc.metadata pulls out file name
for doc in docs:
    print(doc.metadata["file_name"])
      
# --- Step 3: Build the Index and Query Engine ---

# Specify the new model
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
from llama_index.llms.openai import OpenAI

Settings.llm = OpenAI(model="gpt-4o-mini")
#build index and query engine
# Build a vector index automatically (handles chunking + embeddings)
index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine(similarity_top_k=3)

#prints confirmation message once index is ready
print("Index build successfully. Ready to answer questions.")

# --- Step 4: Query the Assistant ---

questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]
#runs each question through the pipeline
for q in questions:
    print(f"\nQ: {q}")
    response  = query_engine.query(q)   #enbeds the question, finds the top 3 chunks, & has the LLM answer using the chunks
    print(f"\nA: ", response)
    
    #source ndoes is sorted best first, so[0] is the top retrieved chunk
    top = response.source_nodes[0]
    text = top.node.get_content()[:200]. replace("\n", " ")
    
    print( f"Source document: {top.node.metadata['file_name']}")
    #cosine similarity between the question and the chunk higher the closer
    print(f"Similarity score: {top.score:.4f}")
    print(f"Text snipet: {text}...")
    print("_" * 60)
    
#The assistant sounded confident and fluent on every answer. The strongest retrieval was "How did Groundwork Coffee get started?"
#(0.76), and the weakest was "How does the loyalty program work?" (0.28). For the weekend hours and dairy-free milk questions, the top-ranked chunk
#did not come from the file that holds the answer (our_story.txt and
#seasonal_specials.txt), yet the model still gave answers that matched the documents, presumably using the other retrieved chunks.
#What surprised me is that the top-ranked chunk and the source of the answer don't always match, so checking only the top node can be misleading.

# --- Step 5: Find a Failure ---
question2 = "What are the cup sizes that you offer?"

print(f"\n {question2}")
response = query_engine.query(question2)
print(f"\n {response}")

#prints all 3 retrieved chunks, best first
for i, node_with_score in enumerate(response.source_nodes, start=1):
    text = node_with_score.node.get_content()[:200].replace("\n", " ")
    print(f"\nSource {i}: {node_with_score.node.metadata['file_name']}")
    print(f"Similarity score: {node_with_score.score:.4f}")
    print(f"Text snippet: {text}...")
    
#Q: "What are the sizes of the cups that you offer?"
#I expected this to be hard because none of the documents mention cup sizes, so the answer isn't in the data.

#What happened: retrieval still returned 3 chunks (scores 0.37, 0.30, 0.29), since it always returns the closest matches even when none are relevant.
#The scores were lower than in most earlier questions, which signals a weak match. The model did not make up sizes. It said the documents don't specify
#cup sizes and suggested contacting the shop directly.

#Tone: here the model sounded less certain and hedged, which was the right behavior. But earlier answers, like weekend hours where the top chunk was
#the wrong file, sounded just as confident as correct ones. So a model's tone is not a reliable sign of accuracy, and AI answers should be checked
#against the source documents.

#Improvements: add cup size info to the menu document, set a minimum similarity score so weak matches are filtered out, and adjust the prompt so
#the assistant clearly says when the answer isn't in the documents.

# --- Step 6: Reflection ---

#1: My LlamaIndex code in project_06.py was very short. The main part took about 5 lines.
#In the Keyword RAG lesson, I wrote a function called simple_keyword_retrieval that was about 35 lines long. And all it did was find the best document by
#matching words. It did not split the text into chunks, it did not create embeddings, and it did not write an answer.
#LlamaIndex did all of that for me in just a few lines. When I ran VectorStoreIndex.from_documents(), it automatically split my documents into
#chunks, turned them into embeddings, and stored them in an index. If I had to write all of that myself, it would take a lot more code and I would
# probably make more mistakes.
#This tells me that a framework is valuable because it saves time and does the repetitive work for me. It also worked better: keyword search missed
#questions like "How do I sign up for rewards?" because the document said "loyalty program," but semantic search understands similar meanings.
#The downside is that I can't always see what is happening inside. For example, I did not choose the chunk size myself. Also, the default model
#(gpt-3.5-turbo) was retired, which caused an error, and I had to learn what the framework was doing in the background to fix it.

#2: Use case: an HR policy assistant for a company.
#Employees often keep asking HR the same questions about PTO, benefits, and the answers are buried in long handbook PDFs. My pipeline could
#load those documents and let employees ask in plain English. Semantic search would find the right policy even if they use different words, like (time
#off) instead of (paid leave.) HR saves time, and when a policy changes, the documents can be updated without retraining a model. Because the model can
#sound confident even when retrieval is weak, I would show the source with each answer and have it say "Contact HR" when it isn't sure.

#3: Even when retrieval works correctly, the model can still make mistakes with the context it receives. It might blend details from different chunks,
#misread a number, or add a detail that isn't in the documents, and it will still sound confident. RAG only controls what text the model sees, not
#whether it uses that text correctly. This is why answers should be checked against the sources, and why tools like the faithfulness evaluator exist. 