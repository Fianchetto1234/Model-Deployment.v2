from flask import Flask, render_template, request, jsonify
import pickle
from utils import make_prediction

tokenizer = pickle.load(open('models/cv.pkl','rb'))
model = pickle.load(open('models/clf.pkl','rb')) 

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST']) 
def submit(): 
    email_text = request.form.get('email_body')
    return render_template('index.html', response=make_prediction(email_text), website_text=email_text)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(force=True)
    email_text = data['email_body']
    return jsonify({'prediction': make_prediction(email_text)})

if __name__ == '__main__':
    app.run(debug=True)