import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import string

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath("intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# normalization function to normalize user input to improve matching
def normalize_input(input_text):
    input_text = input_text.lower()  # Convert to lowercase
    input_text = input_text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return input_text

# Python function to chat with the chatbot
def chatbot(input_text):
    input_text = normalize_input(input_text)  # to normalize user input
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
    return "I'm sorry, I didn't understand that. Can you please rephrase?" # implemented fallback response
        
counter = 0

def main():
    global counter
    st.set_page_config(page_title="AI Chatbot", layout="wide")
    st.title("🤖 Chat with Our Intelligent Chatbot")
    st.subheader("Ask me anything and I'll do my best to respond!")

    # Create a sidebar menu with options
    menu = ["Chat", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Home Menu - Chatting with Chatbot
    if choice == "Chat":
        st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation.")

        # Check if the chat_log.csv file exists, and if not, create it with column names
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User  Input', 'Chatbot Response', 'Timestamp'])

        counter += 1
        user_input = st.text_input("Type your message here...", key=f"user_input_{counter}", 
                                    placeholder="Type your message here...", 
                                    label_visibility="collapsed")

        if user_input:
            # Convert the user input to a string
            user_input_str = str(user_input)

            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}", 
                          label_visibility="collapsed")

            # Get the current timestamp
            timestamp = datetime.datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")

            # Save the user input and chatbot response to the chat_log.csv file
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()

    # Conversation History Menu
    elif choice == "Conversation History":
        st.header("📜 Conversation History")
        with st.expander("Click to see Conversation History", expanded=False):
            st.markdown("<div style='background-color: black; color: white; padding: 10px; border-radius: 10px;'>", unsafe_allow_html=True)
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip the header row
                for row in csv_reader:
                    st.markdown(f"<div style='background-color: #333; padding: 10px; border-radius: 10px;'>"
                                f"<strong>User:</strong> {row[0]}<br>"
                                f"<strong>Chatbot:</strong> {row[1]}<br>"
                                f"<small>{row[2]}</small></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # About Menu
    elif choice == "About":
        st.header("About This Project")
        st.write("This project aims to develop an intelligent chatbot capable of understanding and responding to user queries using Natural Language Processing (NLP) and Machine Learning techniques. The chatbot is designed to provide accurate, context-aware responses based on predefined intents and patterns.")
        
        st.subheader("Project Overview:")
        st.write("""
        The project is divided into three parts:
                1 NLP & Machine Learning: Implements TfidfVectorizer for feature extraction and Logistic Regression for intent classification.
                2 Custom Intent Handling: The chatbot is trained on an intents.json file, allowing it to recognize and respond to user inputs dynamically.
                3 Interactive Web Interface: Built using Streamlit, the chatbot provides an intuitive and user-friendly experience
        """)

        st.subheader("Dataset:")
        st.write("""
        The chatbot is trained on a structured dataset of intents, which includes:
            Intents: Categorized user queries such as greetings, FAQs, and general inquiries.
            Patterns: Sample user inputs associated with each intent.
            Responses: Predefined chatbot replies based on detected intent.
        The model is trained using a supervised learning approach with Logistic Regression, ensuring effective intent classification.
        """)

        st.subheader("Streamlit Chatbot Interface:")
        st.write('''
        The chatbot’s interface is designed for a seamless user experience:
            User-friendly chatbox: Allows users to input queries and receive instant responses.
            Conversation history tracking: Saves user interactions for reference.
            Modern UI/UX: Built with Streamlit for a visually appealing and responsive interface.
        ''')

        st.subheader("Conclusion:")
        st.write("In this project, a chatbot is built that can understand and respond to user input based on intents. The chatbot was trained using NLP and Logistic Regression, and the interface was built using Streamlit. This project can be extended by adding more data, using more sophisticated NLP techniques, deep learning algorithms.")

if __name__ == '__main__':
    main()