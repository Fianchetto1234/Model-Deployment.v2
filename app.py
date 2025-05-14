from flask import Flask, render_template, request, jsonify
import pickle

tokenizer = pickle.load(open('models/cv.pkl','rb'))
model = pickle.load(open('models/clf.pkl','rb')) 

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST']) 
def submit(): 
    email_text = request.form.get('email_body')
    tokenised_email = tokenizer.transform([email_text]) 
    prediction = model.predict(tokenised_email)
    return render_template('index.html', response=prediction[0], website_text=email_text)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(force=True)
    email_text = data['email_body']
    tokenised_email = tokenizer.transform([email_text])
    prediction = model.predict(tokenised_email)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)