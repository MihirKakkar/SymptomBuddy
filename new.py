#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#Creating GUI with tkinter
import json
import tkinter
from tkinter import *
from tensorflow.keras.models import load_model, model_from_json
import io
import os
import pandas as pd
import numpy as np

symptoms_url = "https://raw.githubusercontent.com/KellieChong/MacHacks2021/main/symptoms_diseases_data/sym_t.csv"
diagnosis_url = "https://raw.githubusercontent.com/KellieChong/MacHacks2021/main/symptoms_diseases_data/dia_t.csv"
symptomsdf = pd.read_csv(symptoms_url)
diagnosisdf = pd.read_csv(diagnosis_url)

# symptomsdf = symptomsdf.tonumpy()
# diagnosisdf = diagnosisdf.tonumpy()
#not really sure if we need to load these
f = open("/Users/kelliechong/documents/MacHacks/model.json",)
json_model = json.loads(f.read())

model = load_model('/Users/kelliechong/documents/MacHacks/diagnosis_model_updated.h5')
model.load_weights('/Users/kelliechong/documents/MacHacks/diagnosis_model_updated.h5')

symptomsdf = pd.DataFrame(symptomsdf).to_numpy()
diagnosisdf = pd.DataFrame(diagnosisdf).to_numpy()
rows = len(symptomsdf)

symptoms_words = []#np.empty(rows,)


i=0

for i in range(0,int(rows)):
          description = str(symptomsdf[i]).lower().split() #row i, column 1
          description = [s.replace('(', '') for s in description]
          description = [s.replace(')', '') for s in description]
          symptoms_words.append(description)
          i += 1
symptoms_words = np.asarray(symptoms_words)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def response2symptoms(result):
    threshold = 0.6
    result = lower(result).split()
    compare = np.zeros(rows,)
    j = 0
    for j in range(len(compare)):
        compare[j] = {x for x in result for y in symptoms_words[j] if similar(x, y) > threshold}
        j += 1
    symptom = np.where(RSS == np.amax(compare))
    return symptom
    
def diagnosis(symptom):
    dia = model.predict(symptom,random.randint(0,4))
    diagnosis_text = diagnosisdf[dia][2]
    return diagnosis_text
    
def chatbot_response(msg):
    #ints = predict_class(msg, model)
    res = diagnosis
    #fitting and saving the model
    #model_result = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
    #model.save('diagnosis_model.h5', model_result)

    return res

def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))

        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


base = Tk()
base.title("Symptoms Buddy")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

#Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)

ChatLog.config(state=DISABLED)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

#Create Button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#000000",fg='#ffffff',
                    command= send)

#Create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")
#EntryBox.bind("<Return>", send)


#Place all components on the screen
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

base.mainloop()