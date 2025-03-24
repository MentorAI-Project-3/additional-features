import asyncio
import pymupdf as fitz
import json
import speech_recognition as sr 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from gtts import gTTS
import os
from mutagen.mp3 import MP3
from playsound import playsound  # Add this import at the top

# Initialize text-to-speech engine using gTTS (Async)
async def speak(text, filename="output.mp3"):
    try:
        # Create speech file
        tts = gTTS(text=text, lang='en')
        tts.save(filename)
        
        await asyncio.to_thread(playsound, filename)
        
        try:
            os.remove(filename)
        except:
            pass

    except Exception as e:
        print(f"Speech synthesis error: {e}")
        return None

    return None


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
    
    try:
        with sr.Microphone(device_index=None) as source:  # None uses default microphone
            print("\n🎤 Adjusting for ambient noise... Please wait...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("🎤 Listening for answer... (Say A, B, C, or D)")
            
            # Adjust recognition settings for better accuracy
            recognizer.energy_threshold = 1000  # Increase if in noisy environment
            recognizer.dynamic_energy_threshold = True
            recognizer.pause_threshold = 0.5  # Shorter pause detection
            
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)
                print("Processing your speech...")
                text = recognizer.recognize_google(audio).strip().upper()
                print(f"Recognized: {text}")
                
                # Check for letter anywhere in the response
                for word in text.split():
                    if word in ["A", "B", "C", "D"]:
                        return word
                
                print("❌ Unrecognized answer. Please say A, B, C, or D clearly.")
                return await get_voice_input()
                
            except sr.WaitTimeoutError:
                print("❌ No speech detected. Please try again.")
                return await get_voice_input()
            except sr.UnknownValueError:
                print("❌ Could not understand audio. Please speak more clearly.")
                return await get_voice_input()
            except sr.RequestError as e:
                print(f"❌ Speech Recognition service error: {e}")
                return None
                
    except Exception as e:
        print(f"❌ Microphone error: {e}")
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
    pdf_path = "file_path"
    num_questions_per_chunk = 1  # Adjust as needed

    # Load and process the PDF
    text = load_pdf(pdf_path)
    text_chunks = split_text(text, chunk_size=1000)  # Adjust chunk size as needed

    # Initialize LLM
    llm = ChatGroq(model="llama3-70b-8192", temperature=0.4, api_key='API Groq')

    # Generate quiz questions for each chunk asynchronously
    all_questions = []
    tasks = [generate_quiz_from_chunk(llm, chunk, num_questions=num_questions_per_chunk) for chunk in text_chunks]
    results = await asyncio.gather(*tasks)

    for quiz_questions in results:
        all_questions.extend(quiz_questions)  # Combine all questions

    # Run the full quiz
    global voice_enable
    print("Type V to read the questions outlout or R to read them\n")
    voice_enable = input().strip().upper()
    while voice_enable not in ["V","v", "R","r"]:
            print("❌ Invalid input. Please enter V or R.")
            voice_enable = input().strip().upper()
    if voice_enable in ["V", "v"]:
        voice_enable = True
    else:
        voice_enable = False
    await run_quiz(all_questions)

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())
