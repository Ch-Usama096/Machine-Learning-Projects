from flask import Flask , render_template , request , redirect , url_for
import pickle
import spacy
import re


# Load the Spacy Pre-Trained Model
nlp = spacy.load("en_core_web_sm")

# Create the Function to transform the Text 
def preprocessing_text(text):
    text = re.sub("[^a-zA-Z]" , " " , text) # Only extract the Text Data
    text = text.lower() # Convert the Text into lower Case
    text = nlp(text)  # Read the Text Data
    corpus = []
    # Clean the Text Data
    for token in text:
        if token.is_punct or token.is_stop or len(token) == 1 or len(token) == 2:
            continue
        corpus.append(token.lemma_)
    return " ".join(corpus)

# Create the Lambda function for the Class Label
class_label = lambda x: "NOT SPAM TEXT" if x == 0 else "SPAM TEXT"

# Load the Model/Tf-Idf Vectorizer/Transform_text
mlModel = pickle.load(open("pickle_files\model.pkl" , "rb"))
tfIdf   = pickle.load(open("pickle_files\Vectroizer.pkl" , "rb"))


# Create the Object of Flask
app = Flask(__name__)


# Create the First Decorator for (Home Page)
@app.route("/")
def home_page():
    return render_template("homePage.html")

# Create the Second Decorator for (Generate the Request)
@app.route("/generate_request")
def generate_request():
    return render_template("generateRequest.html")

# Create the Third Decorator for (Process the Request)
@app.route('/process_request' , methods = ["POST" , "GET"])
def process_request():
    if request.method == "POST":
        text = request.form["text"]
        
        # Transform the Text
        text = preprocessing_text(text)

        # Convert the text into Vector
        textVector = tfIdf.transform([text]).toarray()

        # Predict the Result
        prediction = mlModel.predict(textVector)
        
        # get the Class Label
        label = class_label(prediction[0])

        return render_template("generateRequest.html" , prediction = label)
    
# Run the App
if "__main__" == __name__:
    app.run(debug=True) 