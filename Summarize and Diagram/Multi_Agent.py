
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from All_Tools_used import summarize_video, diagram_generation
load_dotenv("API.env")

# Initialize LLM (Using Groq's Llama3-70B model)
llm = ChatGroq(model="llama3-70b-8192", temperature=0)


# Define LangChain Tools
summarize_tool = Tool(
    name="Summarize Text",
    func=summarize_video,
    description=(
        "Downloads a video from YouTube, extracts the audio, transcribes it using Whisper, "
        "and generates a brief summary of the content. "
        "Takes a YouTube link as input and returns a summary. "
        "IMPORTANT: The output is already summarized. Do NOT summarize it again or pass it to any other summarization step."
    )
)
diagram_tool = Tool(
    name="Diagram Generation",
    func=diagram_generation,
    description="Generates a diagram for a given topic using RAG."
)

# Initialize Agent
agent = initialize_agent(
    tools=[diagram_tool, summarize_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={
        "prefix": """You are an agent designed to assist with tasks using the following tools. 
        When you use the Summarize Text tool, your final answer should be exactly the output of that tool, 
        without any further processing or summarization."""
    }
 )

# Test the agent
test_topic = input("Enter what you want: ")
result = agent.invoke({"input": test_topic})
print(result)
