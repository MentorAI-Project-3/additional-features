import sys
import asyncio
import pymupdf as fitz
import json
import speech_recognition as sr  # For voice input
import threading  # For question timer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from gtts import gTTS
from IPython.display import Audio
import os
from mutagen.mp3 import MP3
# to install thing in env use this:
# .venv\Scripts\python.exe -m pip install "name of library"

# Initialize text-to-speech engine using gTTS (Async)
async def speak(text, filename="output.mp3"):
    tts = gTTS(text=text, lang='en')
    tts.save(filename)
    audio = MP3(filename)
    duration = audio.info.length
    os.system(f"start {filename}")  # Play the audio using the system's default player
    await asyncio.sleep(duration)  # Non-blocking delay to allow the audio to play
    return Audio(filename, autoplay=True)

# Load PDF and Extract All Text
def load_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

# Split Text into Manageable Chunks for LLM
def split_text(text, chunk_size=1000):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
    return text_splitter.split_text(text)

# Generate Quiz for Each Text Chunk
async def generate_quiz_from_chunk(llm, text_chunk, num_questions=1):
    prompt = f"""
    Based on the following content, create a {num_questions}-question multiple-choice quiz.
    Ensure that the correct answers are evenly distributed across all answer choices (A, B, C, and D),
    rather than favoring one letter. Make sure the distribution appears natural.

    Content:
    {text_chunk}

    Format the output as a valid JSON structure like this:
    {{
        "questions": [
            {{
                "question": "What is NLP?",
                "options": {{
                    "A": "Natural Language Processing",
                    "B": "Neural Language Processing",
                    "C": "New Language Processing",
                    "D": "None of the above"
                }},
                "answer": "A"
            }}
        ]
    }}

    Only return valid JSON, no extra text.
    """

    response = await asyncio.to_thread(llm.invoke, [HumanMessage(content=prompt)])
    response_text = response.content

    try:
        quiz_data = json.loads(response_text)  # Ensure valid JSON
        return quiz_data.get("questions", [])  # Return only questions
    except json.JSONDecodeError:
        print("❌ Error: LLM did not return valid JSON.")
        return []

# Function to get voice input (Async)
async def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Listening for answer... (Say A, B, C, or D)")
        recognizer.adjust_for_ambient_noise(source)

        try:
            audio = recognizer.listen(source, timeout=5)  # 5-second timeout
            text = recognizer.recognize_google(audio).strip().upper()
            print(text)
            text = text.split(' ')
            if text[-1] in ["A", "B", "C", "D"]:
                return text[-1]
            else:
                print("❌ Unrecognized answer. Please try again.")
                return await get_voice_input()
        except sr.UnknownValueError:
            print("❌ Could not understand audio. Please try again.")
            return await get_voice_input()
        except sr.RequestError:
            print("❌ Speech Recognition service unavailable.")
            return None
        except Exception as e:
            print(f"❌ Error: {e}")
            return None

# Function to get user input (keyboard or voice)
async def get_user_input():
    print("Press 'V' to use voice input or type your answer (A, B, C, or D): ")
    if voice_enable == True: await speak("Press V to use voice input or type your answer.")
    choice = input().strip().upper()
    if choice == "V":
        return await get_voice_input()
    return choice

# Conduct an Interactive Quiz with a Timer
async def run_quiz(quiz_questions):
    if not quiz_questions:
        print("❌ No quiz questions available.")
        return

    score = 0

    i =0
    for q in quiz_questions:
        i +=1 
        print("\n" + f'{i}. {q["question"]}')  # Display question (No voice output for question)
        if voice_enable == True:
            await speak(q["question"])

        for key, value in q["options"].items():
            print(f"{key}: {value}")
            if voice_enable == True:# Speak only the options
                await speak(f"{key}: {value}") 

        user_answer = None

        user_answer = await get_user_input()

        while user_answer not in ["A", "B", "C", "D"]:
            print("❌ Invalid input. Please enter A, B, C, or D.")
            if voice_enable== True: await speak("Invalid input. Please enter A, B, C, or D.")
            user_answer = await get_user_input()

        if user_answer == q["answer"]:
            print("✅ Correct!")
            if voice_enable== True: await speak("Correct!")
            score += 1
        else:
            print(f"❌ Incorrect. The correct answer is {q['answer']}: {q['options'][q['answer']]}")  
            if voice_enable== True: await speak(f"Incorrect. The correct answer is {q['answer']}.")

    print(f"\n🏆 Your final score: {score}/{len(quiz_questions)}")
    if voice_enable== True: await speak(f"Your final score is {score} out of {len(quiz_questions)}")

# Main Execution Function
async def main():
    pdf_path = "C:\\Users\\Windows 11\\Desktop\\UN\\Project 3\\code\\project-3\\Quiz Maker\\Information_Lecture_1[1].pdf"
    num_questions_per_chunk = 1  # Adjust as needed

    # Load and process the PDF
    text = load_pdf(pdf_path)
    text_chunks = split_text(text, chunk_size=1000)  # Adjust chunk size as needed

    # Initialize LLM
    llm = ChatGroq(model="llama3-70b-8192", temperature=0.4, api_key='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')

    # Generate quiz questions for each chunk asynchronously
    all_questions = []
    tasks = [generate_quiz_from_chunk(llm, chunk, num_questions=num_questions_per_chunk) for chunk in text_chunks]
    results = await asyncio.gather(*tasks)

    for quiz_questions in results:
        all_questions.extend(quiz_questions)  # Combine all questions

    # Run the full quiz
    global voice_enable
    print("Are you want a listening the Text or just Read it?\nEnter Yes, Y, No or N.")
    voice_enable = input().strip().upper()
    while voice_enable not in ["Y", "YES", "N", "NO"]:
            print("❌ Invalid input. Please enter Yes, Y, No or N.")
            voice_enable = input().strip().upper()
    if voice_enable in ["YES","Y"]:
        voice_enable = True
    else:
        voice_enable = False
    await run_quiz(all_questions)

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())
