import yt_dlp
import textwrap
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_groq import ChatGroq
from groq import Groq
from dotenv import load_dotenv

load_dotenv("API.env")
# Initialize LLM (Using Groq's Llama3-70B model)
llm = ChatGroq(model="llama3-70b-8192", temperature=0)

def download_audio(url, filename="audio"):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': filename,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(url, download=True)
    return "audio"

# Function to transcribe video
def transcribe_audio(filename: str) -> str:
    client = Groq()
    filename = "audio.mp3" # Use the converted audio file
    with open(filename, "rb") as file:
        transcription = client.audio.transcriptions.create(
          file=(filename, file.read()),
          model="whisper-large-v3",
          response_format="verbose_json",
        )
    return transcription.text


# Function to summarize text
def summarize_text(text: str) -> str:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0, separators=[" ", ",", "\n"]
    )
    texts = text_splitter.split_text(text)
    docs = [Document(page_content=t) for t in texts]
    chain = load_summarize_chain(llm, chain_type="stuff")
    summary = chain.run(docs)
    return textwrap.fill(summary, width=1000)

# Define LangChain Tools
download_tool = Tool(
    name="Download YouTube Video",
    func=download_audio,
    description="Downloads a YouTube video as MP3 given a URL."
)

transcribe_tool = Tool(
    name="Transcribe audio",
    func=transcribe_audio,
    description="Transcribes an MP3 audio file into text."
)

summarize_tool = Tool(
    name="Summarize Text",
    func=summarize_text,
    description="Summarizes a long text into a concise format."
)

# Initialize Agent
# agent = initialize_agent(
#    tools=[download_tool, transcribe_tool, summarize_tool],
#    llm=llm,
#    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#    verbose=True
# )

# Run the agent in one prompt
url = input("Enter YouTube URL: ")
download_audio(url)
transcribed_text = transcribe_audio("audio.mp3")
summary = summarize_text(transcribed_text)
print(summary)