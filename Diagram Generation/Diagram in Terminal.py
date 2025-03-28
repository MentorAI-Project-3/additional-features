from langchain_groq import ChatGroq
from langchain.schema import HumanMessage ,AIMessage,SystemMessage
from dotenv import load_dotenv
load_dotenv("project-3\\API.env")
import subprocess
import re

llm = ChatGroq(model="llama3-70b-8192", temperature=0.3)
topic = input("Enter what you want learn: ")
prompt_content = f"""
You are an agent working in a multi-agent system.
Your job is to provide a description that explains a {topic}
This description will be passed to create a diagram using the Mermaid language.
"""

content = llm.invoke([SystemMessage(content=prompt_content)])


prompt_sys = f"""
Based on the following content, create a code Mermaid.
Your respones just code Mermaid.

content:
{content.content}

Make sure you create the code without errors and following the correct syntax.

These some Examples for code Mermaid.
EX 1:
```
graph TD;
    A["Input Data"] --> B["Neural Network"];
    B --> C["Hidden Layers"];
    C -->|"Weights & Biases"| D["Activation Functions"];
    D --> E["Backpropagation & Optimization"];
    E --> F["Output Predictions"];

    C --> G["Layer 1: Convolution/ Fully Connected"];
    G --> H["Layer 2: Non-Linearity (ReLU)"];
    H --> I["Layer 3: Pooling (Max/Avg)"];
    I --> J["Layer 4: Fully Connected Layer"];
    J --> K["Softmax / Sigmoid for Classification"];

    E --> L["Loss Function (MSE, Cross-Entropy)"];
    L --> M["Gradient Descent (Adam, SGD)"];
    M -->|"Adjust Weights"| C;
```
EX 2:
```
graph LR
    classDef node stroke-width:2px,stroke:#333,fill:#fff;
    classDef edge stroke-width:2px,stroke:#333;

    subgraph Input Embeddings
        Token1[Token 1]
        Token2[Token 2]
        TokenN[Token N]
    end

    subgraph QueryKeyValueMatrices
        Q[Query Matrix]
        K[Key Matrix]
        V[Value Matrix]
    end

    subgraph SelfAttentionMechanism
        AttentionWeights[Attention Weights]
        Softmax[Softmax]
        ContextVector[Context Vector]
    end

    Token1 -->| Embedding | Q
    Token2 -->| Embedding | Q
    TokenN -->| Embedding | Q

    Token1 -->| Embedding | K
    Token2 -->| Embedding | K
    TokenN -->| Embedding | K

    Token1 -->| Embedding | V
    Token2 -->| Embedding | V
    TokenN -->| Embedding | V

    Q -->| Matrix Multiplication | AttentionWeights
    K -->| Matrix Multiplication | AttentionWeights
    V -->| Matrix Multiplication | AttentionWeights

    AttentionWeights -->| Softmax | Softmax
    Softmax -->| Weighted Sum | ContextVector
```

"""
print(content.content)
response = llm.invoke([HumanMessage(content=topic),SystemMessage(content=prompt_sys)])
response_text = response.content
response_text = re.sub(r'\bmermaid\b', '', response.content, flags=re.IGNORECASE)

print(response_text)
#mermaid_code = response_text.split('```')

# Save to a file
with open("diagram.mmd", "w") as f:
    f.write(response_text)

# Use full path to mmdc
mmdc_path = "C:\\Users\\Windows 11\\AppData\\Roaming\\npm\\mmdc.cmd"  # Update this path!
subprocess.run([mmdc_path, "-i", "diagram.mmd", "-o", "diagram.png", "-s", "4"], shell=True)

print("Mermaid diagram generated: diagram.png")

