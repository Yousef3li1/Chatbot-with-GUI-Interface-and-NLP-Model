# 🤖 Chatbot with GUI Interface and NLP Model 💬

## 🌟 Overview

This project implements a **Chatbot** using Natural Language Processing (NLP) to understand and respond to user queries. The chatbot is built using a pre-trained machine learning model and offers a user-friendly GUI interface for interaction. The system uses the **Keras** library for training the model, **NLTK** for text preprocessing, and **tkinter** for the chat interface. The chatbot can learn from intents and provide relevant responses based on user input.

### Key Features:

- **Natural Language Understanding**: The chatbot uses a deep learning model to predict the appropriate response based on user input.
- **User-Friendly GUI**: Built with `tkinter`, providing a simple chat interface.
- **Training System**: Allows the chatbot to be trained on new intents and patterns using predefined datasets.
- **Customizable Intents**: You can add new intents to expand the chatbot's functionality.

---

## ✨ Features

- **Chat Interface**: GUI chat interface with the ability to interact with the chatbot in real time.
- **Machine Learning Model**: Trained on intents and patterns to predict and generate responses.
- **Intent-based Responses**: Uses intents from a JSON file to guide the chatbot's response generation.
- **Text Preprocessing**: Tokenization, lemmatization, and bag of words (BOW) techniques used for sentence processing.
- **Customizable**: You can train the chatbot with new intents and patterns.

---

## 🛠 Components

- **Python Libraries**:
  - `nltk`: For text processing and tokenization.
  - `keras`: For building and training the machine learning model.
  - `tkinter`: For creating the GUI interface.
  - `pickle`: For saving and loading pre-trained models and data.
- **Training Data**: The chatbot is trained using patterns and intents stored in a JSON file (`intents.json`).
- **Model**: The chatbot uses a neural network model (`chatbot_model.h5`) for prediction.

---

## 🧠 Training the Model

-To train the chatbot, the script train_chatbot.py processes the intents from the intents.json file and creates a model using Keras. The model architecture consists of the following layers:

Input Layer: Accepts a bag of words input (representing the user's sentence).
Hidden Layers: Two dense layers with ReLU activation.
Output Layer: A softmax layer to predict the most likely intent.
Training Process:
Tokenize the input patterns.
Lemmatize each word and create a bag of words (BOW).
Train the model using the Keras Sequential API with the following architecture:
Dense layer with 128 neurons
Dropout layer (0.5)
Dense layer with 64 neurons
Dropout layer (0.5)
Output layer with softmax activation (number of classes)
After training, the model is saved as chatbot_model.h5.
---
## 💬 Chatbot Interaction
The chatgui.py file is responsible for providing a GUI interface where users can chat with the chatbot. The GUI has the following components:

Chat Log: Displays the conversation between the user and the bot.
Entry Box: Allows users to type messages.
Send Button: Sends the user's input and displays the bot's response.
How to Chat:
Type a message in the Entry Box.
Click the Send button to submit the message.
The chatbot will process the input and display a response in the Chat Log.
---
##⚠️ Notes
Model Accuracy: The chatbot's responses are based on the training data in intents.json. You can improve the accuracy of the bot by adding more patterns and intents.
Customization: To add new functionality, you can update the intents.json file with new intents and patterns.
Dependencies: Make sure all necessary libraries are installed before running the scripts.
---
##📢 Contributions
We welcome contributions! Whether you want to:

Improve the chatbot's responses.
Add more intents.
Enhance the GUI interface. Feel free to fork the project and submit a pull request.

---
