# SymptomBuddy

## Inspiration

Since doctors, ERs, ICUs, and hospitals are overloaded with coronavirus patients, we decided to build a program that helps diagnose people with diseases. This means that the people won't need to go to the doctors which would further overload ERs, hospitals, and increase their risk of spreading COVID-19.

## What it does
Symptom Buddy is a chatbot that helps diagnose your symptoms and provides suggestions for the best course of action based on the severity of your diagnosis.

## How we built it
- We created a deep neural network for predicting diagnoses based on symptoms
- We created a chatbot application that interacts with the user and determines the disease using the ML model built in part 1
- We created a tkinter GUI frontend for the chatbot

## Challenges we ran into
- Mapping and preprocessing our data for our DNN
- Minimizing the loss function for our machine learning model
- Integrating our trained models to the front-end

## Accomplishments that we're proud of
- Creating a usable program within 2 days
- Using multiple ML models
- Having an interactive GUI for users

## What we learned
- How to format data for a model to train on
- How to create and train a ML model
- How to build a chatbot
- Integrating the frontend with a backend trained models

## What's next for Symptom Buddy
- Better NLP so the chatbot can continue a conversation
- Have Symptom Buddy provide resources and further reading/articles to the user about the diagnosed disease
- Pull location services to suggest the appropriate clinic, support groups, hospital, or urgent care centre in your local neighbourhood
