import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os

load_dotenv()  # Load variables from .env
API_KEY = os.getenv("HUGGING_FACE_API_KEY")


def scrape_website(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        paragraphs = soup.find_all("p")
        content = " ".join(p.get_text() for p in paragraphs)
        return content.strip()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching website content: {e}")
        return None

def process_content(content):
    max_length = 2000  
    return content[:max_length] + "..." if len(content) > max_length else content


def chat_with_huggingface(prompt, context):
    API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
    API_KEY = os.getenv("HUGGING_FACE_API_KEY")  
    
    if not API_KEY:
        return "API key is missing. Please check your environment setup."

    headers = {"Authorization": f"Bearer {API_KEY}"}
    payload = {"inputs": {"question": prompt, "context": context}, "options": {"wait_for_model": True}}

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  
        data = response.json()
        return data.get("answer", "I'm sorry, I couldn't generate a response.")
    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred while processing your request."


def main():
    url = input("Enter the website URL: ")
    print("Fetching website content...")
    context = scrape_website(url)
    
    
    if context:
        print("Processing content...")
        context = process_content(context)
        
        print("Chatbot is ready! Type 'exit' to quit.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                print("Goodbye!")
                break
            
            response = chat_with_huggingface(user_input, context)
            print(f"Chatbot: {response}")
    else:
        print("Failed to fetch or process website content. Exiting.")

if __name__ == "__main__":
    main()
