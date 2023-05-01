import nltk
import pickle
import numpy as np
import json
import random
import subprocess
from pysondb import db
from operations import DatabaseOperator
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
from flask import Flask, render_template, request, redirect, url_for, session

# Run once !
# nltk.download('popular')

lemmatizer = WordNetLemmatizer()


def load_chatbot_data():
    model = load_model('chatbot_model.h5')
    intents = json.loads(open('navigation.json').read())
    words = pickle.load(open('texts.pkl', 'rb'))
    classes = pickle.load(open('labels.pkl', 'rb'))
    return model, intents, words, classes


model, intents, words, classes = load_chatbot_data()


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower())
                      for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag


def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    # Control only
                    print("found in bag: %s" % w)
    return (np.array(bag))


def predict_class(sentence, model):
    # Dilter out predictions below with error_threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"data": classes[r[0]], "probability": str(r[1])})
    return return_list


def get_response(ints, intents_json):
    tag = ints[0]['data']
    list_of_intents = intents_json['data']
    for i in list_of_intents:
        if (i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result


def get_response_advanced(ints, intents_json):
    tag = ints[0]['data']
    list_of_intents = intents_json['data']
    for i in list_of_intents:
        if (i['tag'] == tag):
            result = check_context(i['context'])
            print(result)
            break
    return result


def reset_password():
    return '<a href="/pass_reset">Reset Password</a>'


def check_order_status():
    result = db_operator.get()
    return f"""Your order number is {result['OrderNumber']} and is
     {result['OrderStatus']}.
      For more information type parcels do accquire additional help."""


def tracking_parcels():
    result = db_operator.get()
    return f"""Your parcel is {result['Delivery']}, and carrier is
    {result['Carrier']}, and tracking number is {result['TrackingNumber']}."""


def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = get_response(ints, intents)
    res2 = get_response_advanced(ints, intents)
    if res2:
        return res + '\n' + res2
    else:
        return res


def check_context(query_type):
    query_type = query_type[0]  # extracting just data
    if query_type == "reset_password":
        return reset_password()
    elif query_type == "check_order_status":
        return check_order_status()
    elif query_type == "tracking_parcels":
        return tracking_parcels()
    return False


# Main Body
users = {
    "admin": {
        "password": "pass123",
        "role": "Administrator"
    },
    "customer": {
        "password": "pass123",
        "role": "Customer"
    }
}

db_operator = DatabaseOperator(user="user")
database = db.getDb('navigation.json')
app = Flask(__name__)
app.secret_key = "bachelorproject020"
app.static_folder = 'static'


# Home page
@app.route("/")
def home():
    if "username" in session:
        return render_template("index.html")
    else:
        return redirect(url_for("login"))


# Reset password page
@app.route("/pass_reset")
def pass_reset():
    session.pop("username", None)
    session.pop("role", None)
    return render_template("login.html",
                           error="Your password has been reset to 'pass123' ")


# Login page
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if username in users and password == users[username]["password"]:
            session["username"] = username
            session["role"] = users[username]["role"]
            return redirect(url_for("home"))
        else:
            return render_template("login.html",
                                   error="Invalid username or password.")
    else:
        return render_template("login.html")


# Logout
@app.route("/logout")
def logout():
    session.pop("username", None)
    session.pop("role", None)
    return render_template("login.html",
                           error="You have successfully logged out.")


# Get Bot Response from user
@app.route("/get")
def get_bot_response():
    msg = request.args.get('msg')
    return chatbot_response(msg)


# Add intent page
@app.route('/add', methods=['GET', 'POST'])
def add():
    if request.method == 'POST':
        tag = request.form['tag']
        patterns = request.form['patterns'].split(',')
        responses = request.form['responses'].split(',')
        context = request.form['context'].split(',')
        intent = {
            'tag': tag,
            'patterns': patterns,
            'responses': responses,
            'context': context
        }
        database.add(intent)
        return redirect(url_for('admin'))


# Update intent page
@app.route('/update', methods=['GET', 'POST'])
def update():
    if request.method == 'POST':
        db_id = request.form['id']
        patterns = request.form['patterns'].split(',')
        responses = request.form['responses'].split(',')
        context = request.form['context'].split(',')
        intent = {
            'patterns': patterns,
            'responses': responses,
            'context': context
        }

        database.updateById(db_id, intent)
        return redirect(url_for('admin'))


# Delete intent page
@app.route('/delete/', methods=['GET', 'POST'])
def delete():
    if request.method == 'POST':
        db_id = request.form['id']
        database.deleteById(db_id)
        return redirect(url_for('admin'))


# List View
@app.route('/admin/')
def admin():
    return render_template("admin.html", intents=database.getAll())


# Train page
@app.route('/train', methods=['GET', 'POST'])
def train():
    return render_template("train.html")


# Executing the python script for training the bot
@app.route('/execute')
def execute():
    result = subprocess.run(['python', 'chatbot.py'],
                            capture_output=True, text=True)
    # Get the output of the script
    output = result.stdout
    model, intents, words, classes = load_chatbot_data()
    # Render the template with the output
    return render_template('output.html', output=output)


if __name__ == "__main__":
    app.run(host='0,0,0,0', port=80)
