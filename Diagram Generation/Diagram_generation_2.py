from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import time
import re
import traceback
import os
import glob

# Load environment variables from .env file
load_dotenv("API.env")

def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

# Initialize the LLM model
llm = ChatGroq(model="llama3-70b-8192", temperature=0)

# Initialize embeddings for RAG
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Function to load and process books
def load_books(book_path="books"):
    documents = []
    print(f"📁 Checking for books at: {book_path}")
    
    # Check if book_path is a file or directory
    if os.path.isfile(book_path):
        files = [book_path]
        print(f"📄 Found single file: {book_path}")
    elif os.path.isdir(book_path):
        files = glob.glob(os.path.join(book_path, "*.txt")) + glob.glob(os.path.join(book_path, "*.pdf"))
        print(f"📂 Found {len(files)} files in directory: {files}")
    else:
        print(f"❌ Path '{book_path}' does not exist.")
        return documents

    for file in files:
        try:
            if file.endswith(".txt"):
                print(f"📜 Loading text file: {file}")
                loader = TextLoader(file)
                documents.extend(loader.load())
            elif file.endswith(".pdf"):
                print(f"📕 Loading PDF file: {file}")
                loader = PyPDFLoader(file)
                documents.extend(loader.load())
        except Exception as e:
            print(f"⚠️ Error loading {file}: {str(e)}")
    
    if not documents:
        print("⚠️ No documents loaded successfully.")
        return documents
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"📑 Processed {len(chunks)} document chunks.")
    return chunks

# Create or load vector store
def get_vector_store(book_path="books"):
    if os.path.exists("faiss_index"):
        print("🔍 Loading existing FAISS index.")
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    else:
        print("🛠️ Creating new FAISS index.")
        chunks = load_books(book_path)
        if not chunks:
            print("⚠️ No books found. Proceeding without RAG context.")
            return None
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local("faiss_index")
        print("💾 FAISS index saved.")
    return vector_store

# Retrieve relevant content for the topic
def retrieve_context(topic, vector_store, k=4):
    if vector_store is None:
        return ""
    docs = vector_store.similarity_search(topic, k=k)
    context = "\n\n".join([doc.page_content for doc in docs])
    print(f"🔎 Retrieved {len(docs)} relevant document chunks for topic: {topic}")
    return context

# Input from user for the book path and topic
#book_path = input("Enter the path to your book (PDF or directory, default 'books'): ") or "books"
topic = input("Enter what you want to learn: ")

# Load vector store for RAG
vector_store = get_vector_store("C:\\Users\\Windows 11\\Desktop\\UN\\Project 3\\code\\project-3/Diagram Generation\Hands-On Large Language Models Language Understanding and Generation (Jay Alammar) (Z-Library).pdf")

# Retrieve relevant book content
book_context = retrieve_context(topic, vector_store)

# First prompt: Explanation generation with RAG context
prompt_content = """
You are an expert explainer working in a multi-agent system.

Your job is to generate a clear, structured, and technically accurate description of a given topic so it can be turned into a diagram. Use the provided book content as the primary source of information to ensure the explanation is grounded in the reference material.

**Book Content**:
{book_context}

Requirements:
- Break down the topic into logical steps, layers, or components based on the book content.
- Describe each part in order of how it flows or interacts with others.
- Focus on function, purpose, input/output, and relationships.
- Keep the explanation concise but informative — not too short, not too long.
- The description should be suitable for transforming into a technical diagram using Graphviz.
- If the book content is insufficient, supplement with general knowledge but prioritize the book.

Your output must be plain text, without code or special formatting.
"""

# Format prompt with book context
prompt_content = prompt_content.format(book_context=book_context if book_context else "No book content available.")

# Get the structured description from the LLM
content = llm.invoke([SystemMessage(content=prompt_content), HumanMessage(content=topic)])

# Second prompt: Code generation based on the description
prompt_sys = """
You are an expert technical diagram assistant. Your job is to generate annotated, educational diagrams using Python's Graphviz library.

Your output should be a Python script that:
- Use `Digraph(format='png')` and set `dpi='300'` for high-quality image generation.
- Uses `graphviz.Digraph()` to create diagrams.
- Is vertical (top to bottom) using `rankdir='TB'`.
- Explains the topic clearly, with well-labeled nodes and structured flow.
- Includes **side notes** as `note`-shaped nodes connected with dashed edges for learning aid.
- Uses colors and groupings to visually separate concepts (e.g., attention, embedding, logic, output).
- Adds emoji or icons when relevant to make the diagram engaging.
- Groups components logically (e.g., encoder, decoder, steps).
- Adds short, simple labels for each node.
- Adds 1–2 sentence notes for difficult concepts (e.g., “Masked Self-Attention” or “Softmax Layer”).
Your goal is to teach students or junior developers about a complex technical subject in a clean, visual, and readable format.

🧠 The diagram should:
- Represent concepts with *friendly labels* and visual structure.
- Help someone understand the subject quickly.
- Be suitable for use in a blog post, classroom, or tech presentation.

🛠️ Your response should ONLY include the full Python code using Graphviz to render the diagram. Don't add explanations outside of the code. Make sure the final diagram is complete and renders without error.

💡 Example Topics:
- How Self-Attention Works in Transformers
- Neural Network Architecture (CNN, RNN, etc.)
- How Gradient Descent Works
- LLM Pipeline: Tokenization to Generation
- Blockchain Transaction Lifecycle

Only output working Python code using the `graphviz` library. Do not include markdown formatting or extra explanation.
---

Here is an example of the kind of Python code you should generate:

EX 1:
```python
from graphviz import Digraph

g = Digraph('Transformer', format='png')
g.attr(rankdir='TB', dpi='300', nodesep='0.7', ranksep='1.1')
g.attr('node', shape='box', style='filled,rounded', fontsize='10', fontname='Helvetica', color='#444444')

# === Color Palette ===
colors = {
    'attention': '#E6CCFF',
    'masked_attention': '#FFD6CC',
    'ff': '#D9F0FF',
    'norm': '#E2F7E1',
    'embed': '#FFE9EC',
    'pos': '#FFFFFF',
    'final': '#FFFACD',
    'label': '#F0F0F0',
    'note': '#FFFFDD'
}

# === Main Nodes ===
def styled_node(name, label, fill):
    g.node(name, label, fillcolor=fill)

styled_node('Input', '🡒 Inputs', 'white')
styled_node('InputEmbed', 'Input Embedding', colors['embed'])
styled_node('PosEnc1', 'Positional Encoding', colors['pos'])
styled_node('EncMHA', 'Multi-Head Attention', colors['attention'])
styled_node('EncNorm1', 'Add & Norm', colors['norm'])
styled_node('EncFF', 'Feed Forward', colors['ff'])
styled_node('EncNorm2', 'Add & Norm', colors['norm'])
styled_node('EncoderNx', 'Nx Encoder Blocks', colors['label'])

styled_node('Output', '🡒 Outputs (shifted right)', 'white')
styled_node('OutputEmbed', 'Output Embedding', colors['embed'])
styled_node('PosEnc2', 'Positional Encoding', colors['pos'])
styled_node('DecMaskedMHA', 'Masked\nMulti-Head Attention', colors['masked_attention'])
styled_node('DecNorm1', 'Add & Norm', colors['norm'])
styled_node('DecMHA', 'Multi-Head Attention\n(from Encoder)', colors['attention'])
styled_node('DecNorm2', 'Add & Norm', colors['norm'])
styled_node('DecFF', 'Feed Forward', colors['ff'])
styled_node('DecNorm3', 'Add & Norm', colors['norm'])
styled_node('DecoderNx', 'Nx Decoder Blocks', colors['label'])

styled_node('Linear', 'Linear', colors['final'])
styled_node('Softmax', 'Softmax', colors['final'])
styled_node('OutputProbs', '🎯 Output Probabilities', 'white')

# === Flow Edges ===
g.edge('Input', 'InputEmbed')
g.edge('InputEmbed', 'PosEnc1')
g.edge('PosEnc1', 'EncMHA')
g.edge('EncMHA', 'EncNorm1')
g.edge('EncNorm1', 'EncFF')
g.edge('EncFF', 'EncNorm2')
g.edge('EncNorm2', 'EncoderNx')

g.edge('Output', 'OutputEmbed')
g.edge('OutputEmbed', 'PosEnc2')
g.edge('PosEnc2', 'DecMaskedMHA')
g.edge('DecMaskedMHA', 'DecNorm1')
g.edge('DecNorm1', 'DecMHA')
g.edge('EncoderNx', 'DecMHA')  # Cross attention
g.edge('DecMHA', 'DecNorm2')
g.edge('DecNorm2', 'DecFF')
g.edge('DecFF', 'DecNorm3')
g.edge('DecNorm3', 'DecoderNx')

g.edge('DecoderNx', 'Linear')
g.edge('Linear', 'Softmax')
g.edge('Softmax', 'OutputProbs')

# === Subgraph Labels ===
with g.subgraph() as encoder:
    encoder.attr(rank='same')
    encoder.node('EncoderNx')
    encoder.attr(label='<<B>ENCODER</B>>', labelloc='t', style='dashed')

with g.subgraph() as decoder:
    decoder.attr(rank='same')
    decoder.node('DecoderNx')
    decoder.attr(label='<<B>DECODER</B>>', labelloc='t', style='dashed')

# === Educational Notes ===
g.attr('node', shape='note', style='filled', fillcolor=colors['note'], fontcolor='black')

g.node('NoteMHA', '🧠 Self-Attention allows each word to "look at" other words in a sentence\nto decide which ones are important for its own representation.')
g.edge('EncMHA', 'NoteMHA', style='dashed', arrowhead='none')

g.node('NoteMaskedMHA', '🚫 Masked Self-Attention prevents the decoder from "cheating" by looking at future words during training.')
g.edge('DecMaskedMHA', 'NoteMaskedMHA', style='dashed', arrowhead='none')

g.node('NoteCrossAttention', '🔄 Decoder MHA (cross-attention) focuses on the encoder\'s output.\nIt helps the decoder decide what parts of the input are relevant.')
g.edge('DecMHA', 'NoteCrossAttention', style='dashed', arrowhead='none')

g.node('NotePositionalEncoding', '📍 Since self-attention lacks order, positional encoding adds\ninformation about word positions.')
g.edge('PosEnc1', 'NotePositionalEncoding', style='dashed', arrowhead='none')

g.node('NoteNx', '♻️ The blocks are repeated Nx times (often 6). Each repetition refines\nthe representation further.')
g.edge('EncoderNx', 'NoteNx', style='dashed', arrowhead='none')

g.node('NoteSoftmax', '🎯 Final layer produces probabilities for each word in the vocabulary.')
g.edge('Softmax', 'NoteSoftmax', style='dashed', arrowhead='none')

# === Render ===
g.render('transformer_with_notes', view=True)
```

EX 2:

```python
from graphviz import Digraph

g = Digraph('Transformer', format='png')
g.attr(rankdir='TB', dpi='300', nodesep='0.7', ranksep='1.1')
g.attr('node', shape='box', style='filled,rounded', fontsize='10', fontname='Helvetica', color='#444444')

# === Color Palette ===
colors = {
    'attention': '#E6CCFF',     # lavender
    'masked_attention': '#FFD6CC',  # light peach
    'ff': '#D9F0FF',            # light blue
    'norm': '#E2F7E1',          # mint green
    'embed': '#FFE9EC',         # soft pink
    'pos': '#FFFFFF',           # white
    'final': '#FFFACD',         # lemon chiffon
    'label': '#F0F0F0'
}

# === Helper to make styled nodes ===
def styled_node(name, label, fill):
    g.node(name, label, fillcolor=fill)

# === ENCODER BLOCK ===
styled_node('Input', '🡒 Inputs', 'white')
styled_node('InputEmbed', 'Input Embedding', colors['embed'])
styled_node('PosEnc1', 'Positional Encoding', colors['pos'])
styled_node('EncMHA', 'Multi-Head Attention', colors['attention'])
styled_node('EncNorm1', 'Add & Norm', colors['norm'])
styled_node('EncFF', 'Feed Forward', colors['ff'])
styled_node('EncNorm2', 'Add & Norm', colors['norm'])
styled_node('EncoderNx', 'Nx Encoder Blocks', colors['label'])

# === DECODER BLOCK ===
styled_node('Output', '🡒 Outputs (shifted right)', 'white')
styled_node('OutputEmbed', 'Output Embedding', colors['embed'])
styled_node('PosEnc2', 'Positional Encoding', colors['pos'])
styled_node('DecMaskedMHA', 'Masked MHA', colors['masked_attention'])
styled_node('DecNorm1', 'Add & Norm', colors['norm'])
styled_node('DecMHA', 'Multi-Head Attention\n(from Encoder)', colors['attention'])
styled_node('DecNorm2', 'Add & Norm', colors['norm'])
styled_node('DecFF', 'Feed Forward', colors['ff'])
styled_node('DecNorm3', 'Add & Norm', colors['norm'])
styled_node('DecoderNx', 'Nx Decoder Blocks', colors['label'])

# === FINAL LAYERS ===
styled_node('Linear', 'Linear', colors['final'])
styled_node('Softmax', 'Softmax', colors['final'])
styled_node('OutputProbs', '🎯 Output Probabilities', 'white')

# === FLOW: Encoder ===
g.edge('Input', 'InputEmbed')
g.edge('InputEmbed', 'PosEnc1')
g.edge('PosEnc1', 'EncMHA')
g.edge('EncMHA', 'EncNorm1')
g.edge('EncNorm1', 'EncFF')
g.edge('EncFF', 'EncNorm2')
g.edge('EncNorm2', 'EncoderNx')

# === FLOW: Decoder ===
g.edge('Output', 'OutputEmbed')
g.edge('OutputEmbed', 'PosEnc2')
g.edge('PosEnc2', 'DecMaskedMHA')
g.edge('DecMaskedMHA', 'DecNorm1')
g.edge('DecNorm1', 'DecMHA')
g.edge('EncoderNx', 'DecMHA')  # Cross attention from encoder
g.edge('DecMHA', 'DecNorm2')
g.edge('DecNorm2', 'DecFF')
g.edge('DecFF', 'DecNorm3')
g.edge('DecNorm3', 'DecoderNx')

# === FINAL FLOW ===
g.edge('DecoderNx', 'Linear')
g.edge('Linear', 'Softmax')
g.edge('Softmax', 'OutputProbs')

# === GROUP LABELS ===
with g.subgraph() as encoder:
    encoder.attr(rank='same')
    encoder.node('EncoderNx')
    encoder.attr(label='<<B>ENCODER</B>>', labelloc='t', style='dashed')

with g.subgraph() as decoder:
    decoder.attr(rank='same')
    decoder.node('DecoderNx')
    decoder.attr(label='<<B>DECODER</B>>', labelloc='t', style='dashed')

# === Render ===
g.render('modern_transformer_diagram', view=True)
```
"""

# Prompt for fixing erroneous code
prompt_fix_code = """
You are an expert Python and Graphviz debugging assistant. Your task is to fix errors in a Python script that uses the Graphviz library to generate a diagram.

You will receive:
- The original Python code that failed.
- The error message and traceback from the execution attempt.
- The topic and description the code is trying to visualize.

Your job is to:
- Analyze the error message and traceback to identify the issue.
- Fix the code to resolve the error while ensuring it still generates a valid Graphviz diagram for the given topic and description.
- Ensure the fixed code adheres to the original requirements (e.g., uses `Digraph(format='png')`, `rankdir='TB'`, `dpi='300'`, includes notes, colors, etc.).
- Only output the corrected Python code, without any explanations or markdown formatting.
- If the error is unrelated to Graphviz (e.g., syntax error), fix it while preserving the diagram's structure.
- If the error is due to missing imports, invalid node/edge definitions, or incorrect Graphviz syntax, correct those specifically.

**Original Topic**: {topic}

**Original Description**: {description}

**Erroneous Code**:
```python
{erroneous_code}
```

**Error Message and Traceback**:
```
{error_message}
```

**Output**: Only the corrected Python code using the `graphviz` library.
"""

# Get the Graphviz Python code from LLM
response = llm.invoke([SystemMessage(content=prompt_sys), HumanMessage(content=content.content)])

# Extract and clean the generated code
response_text = response.content
graphviz_code = response_text.split('```')[1].strip('python\n') if '```' in response_text else response_text.strip()

# Handle the code execution safely with isolated namespace
filename = f"diagram_{topic.replace(' ', '_')}_{int(time.time())}"
filename = sanitize_filename(filename)

# First attempt to execute the code
try:
    exec(graphviz_code)
    print(f"✅ Diagram generated successfully: {filename}.png")
except Exception as e:
    error_message = f"{str(e)}\n\n{traceback.format_exc()}"
    print(f"❌ Initial attempt failed with error: {str(e)}")

    # Enter loop for error correction (up to 2 additional attempts)
    for i in range(2):
        # Prepare prompt to fix the code
        fix_prompt = prompt_fix_code.format(
            topic=topic,
            description=content.content,
            erroneous_code=graphviz_code,
            error_message=error_message
        )

        # Get corrected code from LLM
        fix_response = llm.invoke([SystemMessage(content=fix_prompt)])
        fixed_code = fix_response.content

        # Clean the fixed code
        if '```' in fixed_code:
            fixed_code = fixed_code.split('```')[1].strip('python\n')
        else:
            fixed_code = fixed_code.strip()

        print(f"🛠️ Attempting to fix code for retry {i+1}/2")
        graphviz_code = fixed_code  # Update code for next attempt

        try:
            exec(graphviz_code)
            print(f"✅ Diagram generated successfully: {filename}.png")
            break
        except Exception as e:
            error_message = f"{str(e)}\n\n{traceback.format_exc()}"
            print(f"❌ Attempt {i+1}/2 failed with error: {str(e)}")
            if i == 1:
                print(f"❌ All attempts failed. Could not generate diagram.")

# Optionally, save the topic, description, and code to a text file for review
with open(f"{filename}.txt", "w", encoding="utf-8") as f: 
    f.write(f"## Topic: {topic}\n\n")
    f.write(f"### Description:\n{content.content}\n\n")
    f.write(f"### Book Context:\n{book_context}\n\n")
    f.write(f"### Code:\n{graphviz_code}")