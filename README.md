# Implementation of ChatBot using NLP

**About This Project**
This project aims to develop an intelligent chatbot capable of understanding and responding to user queries using Natural Language Processing (NLP) and Machine Learning techniques. The chatbot is designed to provide accurate, context-aware responses based on predefined intents and patterns.

**Project Overview**
This chatbot is built using:
- NLP & Machine Learning: Implements TfidfVectorizer for feature extraction and Logistic Regression for intent classification.
- Custom Intent Handling: The chatbot is trained on an intents.json file, allowing it to recognize and respond to user inputs dynamically.
- Interactive Web Interface: Built using Streamlit, the chatbot provides an intuitive and user-friendly experience.

**Dataset & Model**
The chatbot is trained on a structured dataset of intents, which includes:
- Intents: Categorized user queries such as greetings, FAQs, and general inquiries.
- Patterns: Sample user inputs associated with each intent.
- Responses: Predefined chatbot replies based on detected intent.
The model is trained using a supervised learning approach with Logistic Regression, ensuring effective intent classification.

**Chatbot Interface**
The chatbot’s interface is designed for a seamless user experience:
- User-friendly chatbox: Allows users to input queries and receive instant responses.
- Conversation history tracking: Saves user interactions for reference.
- Modern UI/UX: Built with Streamlit for a visually appealing and responsive interface.

**Future Enhancements**
This chatbot can be improved further by:
- Expanding the dataset for broader topic coverage.
- Integrating deep learning models like Transformers for improved accuracy.
- Enhancing the UI with more interactive elements.
- Deploying as a cloud-based API for wider accessibility.
**Conclusion**
This project demonstrates how AI and NLP can be leveraged to build an interactive chatbot. With further enhancements, it can evolve into a more advanced AI assistant for real-world applications.




week 1 :
- downloaded and imported required packages
- created basic intents
- added more intents

week 2 :
- Created vectorizer and classifier
- Preprocessed the data
- Trainined the model
- Defined function to normalize user input 
- Defined function to chat with ChatBot
- Tested ChatBot

week 3 :
- Building Web Interface for the chatbot using Streamlit
- Interface consists 3 sections - Home, Conversation History & About
