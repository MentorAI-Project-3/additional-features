import subprocess

# Fixed Mermaid diagram definition (wrapped text in quotes)
mermaid_code = """
graph LR
    classDef input fill:#cce,stroke:#333,stroke-width:2px;
    classDef conv fill:#f9f,stroke:#333,stroke-width:2px;
    classDef pool fill:#ccf,stroke:#333,stroke-width:2px;
    classDef flatten fill:#ffc,stroke:#333,stroke-width:2px;
    classDef dense fill:#ff9,stroke:#333,stroke-width:2px;
    classDef output fill:#9f9,stroke:#333,stroke-width:2px;

    input["Input Data"] --> conv1["Convolutional Layer"]
    conv1 --> pool1["Max Pooling Layer"]
    pool1 --> conv2["Convolutional Layer"]
    conv2 --> pool2["Max Pooling Layer"]
    pool2 --> conv3["Convolutional Layer"]
    conv3 --> pool3["Max Pooling Layer"]
    pool3 --> flatten["Flatten Layer"]
    flatten --> dense1["Dense Layer"]
    dense1 --> dense2["Dense Layer"]
    dense2 --> output["Output Layer"]

    class input input;
    class conv1,conv2,conv3 conv;
    class pool1,pool2,pool3 pool;
    class flatten flatten;
    class dense1,dense2 dense;
    class output output;

    """

# Save to a file
with open("diagram.mmd", "w") as f:
    f.write(mermaid_code)

# Use full path to mmdc
mmdc_path = "C:\\Users\\Windows 11\\AppData\\Roaming\\npm\\mmdc.cmd"  # Update this path!
subprocess.run([mmdc_path, "-i", "diagram.mmd", "-o", "diagram_mer.png", "-s", "4"], shell=True)

print("Mermaid diagram generated: diagram.png")

