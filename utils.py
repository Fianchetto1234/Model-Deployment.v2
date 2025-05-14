import pickle

tokenizer = pickle.load(open('models/cv.pkl','rb'))
model = pickle.load(open('models/clf.pkl','rb'))

def make_prediction(email):
    tokenised_email = tokenizer.transform([email_text])
    prediction = model.predict(tokenised_email)[0]
    return prediction
