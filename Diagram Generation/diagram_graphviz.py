from graphviz import Digraph
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import time
import re

# Load environment variables from .env file
load_dotenv("API.env")

def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

# Initialize the LLM model
llm = ChatGroq(model="llama3-70b-8192", temperature=0)

# Input from user for the topic to explain
topic = input("Enter what you want to learn: ")

# First prompt: Explanation generation
prompt_content = """
You are an expert explainer working in a multi-agent system.

Your job is to generate a clear, structured, and technically accurate description of a given topic so it can be turned into a diagram.

Requirements:
- Break down the topic into logical steps, layers, or components.
- Describe each part in order of how it flows or interacts with others.
- Focus on function, purpose, input/output, and relationships.
- Keep the explanation concise but informative — not too short, not too long.
- The description should be suitable for transforming into a technical diagram using Graphviz.

Your output must be plain text, without code or special formatting.
"""

# Get the structured description from the LLM
content = llm.invoke([SystemMessage(content=prompt_content), HumanMessage(content=topic)])

# Second prompt: Code generation based on the description
prompt_sys = """
You are an expert diagram generator for technical systems, deep learning, and workflows.

Your job is to convert a structured topic description into professional Python code using the `graphviz` library to create a high-resolution, annotated diagram.

Guidelines:
- Use `Digraph(format='png')` and set `dpi='300'` for high-quality image generation.
- Use `rankdir='LR'`, `splines='ortho'`, and set clean spacing with `nodesep` and `ranksep`.
- Structure the diagram as a flow from left to right.
- Each node should have:
  - A clear, readable label (emojis are optional for clarity)
  - Shape: `box`
  - Fill color based on function (input, processing, output, optional/extra)
  - Internal description of operation, input/output shape, etc.
- Add **annotation notes** using `shape='note'` to explain major parts of the process.
- Use invisible dashed edges to connect notes to their relevant nodes.
- Organize the code with clear comments (e.g., MAIN NODES, NOTES, EDGES).
- End with: `graph.render("diagram_filename", view=True)`

Only output working Python code using the `graphviz` library. Do not include markdown formatting or extra explanation.

---

Here is an example of the kind of Python code you should generate:

```python
from graphviz import Digraph

graph = Digraph(format='png')
graph.attr(dpi='300')
graph.attr(rankdir='LR', splines='ortho', nodesep='0.8', ranksep='0.9')
graph.attr('node', fontname='Helvetica', fontsize='12', shape='box', style='filled', color='lightgray')

# MAIN FLOW
graph.node('Input', '📷 Input Image\n(224x224x3)', color='lightblue')
graph.node('Conv', '🧠 Conv Layer\n32 Filters\n3x3', color='lightyellow')
graph.node('ReLU', '⚡ ReLU Activation', color='lightyellow')
graph.node('Pool', '📉 Max Pooling\n2x2', color='lightgoldenrod1')
graph.node('Flatten', '📦 Flatten', color='lightgray')
graph.node('Dense', '🔢 Dense Layer\n128 Units\nReLU', color='lightsteelblue1')
graph.node('Output', '🔚 Output Layer\nSoftmax', color='lightgreen')

# EXPLANATORY NOTES
graph.attr('node', shape='note', style='filled', color='cornsilk', fontcolor='black', fontsize='10')
graph.node('NoteConv', '📝 Convolution extracts features\nlike edges and textures')
graph.node('NoteDense', '📝 Dense layer combines features\ninto class probabilities')

# EDGES
graph.edge('Input', 'Conv')
graph.edge('Conv', 'ReLU')
graph.edge('ReLU', 'Pool')
graph.edge('Pool', 'Flatten')
graph.edge('Flatten', 'Dense')
graph.edge('Dense', 'Output')

# NOTE CONNECTIONS
graph.edge('Conv', 'NoteConv', style='dashed', arrowhead='none')
graph.edge('Dense', 'NoteDense', style='dashed', arrowhead='none')

# RENDER
graph.render('simple_dl_pipeline', view=True)
"""

#Get the Graphviz Python code from LLM
response = llm.invoke([SystemMessage(content=prompt_sys), HumanMessage(content=content.content)])

#Extract and clean the generated code
response_text = response.content
print("Generated Content:", content.content)

#Handle the code execution safely with isolated namespace
try:
    graphviz_code = response_text.split('```')[1].strip('python\n')
    filename = f"diagram_{topic.replace(' ', '_')}_{int(time.time())}"
    filename = sanitize_filename(filename)

    # Safely execute the generated code in an isolated namespace
    namespace = {}
    exec(graphviz_code, {}, namespace)
except Exception as e:
    print(f"❌ Error executing generated code: {e}")

#Optionally, save the topic, description, and code to a text file for review
with open(f"{filename}.txt", "w", encoding="utf-8") as f: 
    f.write(f"## Topic: {topic}\n\n")
    f.write(f"### Description:\n{content.content}\n\n")
    f.write(f"### Code:\n{graphviz_code}")