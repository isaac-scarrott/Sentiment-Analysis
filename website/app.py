from flask import Flask , render_template, request
import sentimentAnalysis
app = Flask(__name__)

Sentiment = "50%"
SentimentObject = sentimentAnalysis.sentimentAnalysisObject()

Trend1, Trend2, Trend3, Trend4, Trend5 = SentimentObject.getCurrentTrends()

@app.route("/")
def index():
	Trend1, Trend2, Trend3, Trend4, Trend5 = SentimentObject.getCurrentTrends()
	return render_template("index.html", sentiment = Sentiment, trend1 = Trend1, trend2 = Trend2, trend3 = Trend3, trend4 = Trend4, trend5 = Trend5)

@app.route("/sentiment-search/", methods=['POST'])
def sentimentsearch():
    text = request.form['search_keyword']
    Sentiment = SentimentObject.sentimentSearchInput(text)
    Trend1, Trend2, Trend3, Trend4, Trend5 = SentimentObject.getNewTrends()
    return render_template('index.html', sentiment = Sentiment, trend1 = Trend1, trend2 = Trend2, trend3 = Trend3, trend4 = Trend4, trend5 = Trend5);

@app.route('/trend1link', methods=['GET', 'POST'])
def trend1Sentiment():
	Sentiment = SentimentObject.sentimentSearchInput(Trend1)
	return render_template('index.html', sentiment = Sentiment, trend1 = Trend1, trend2 = Trend2, trend3 = Trend3, trend4 = Trend4, trend5 = Trend5);

@app.route('/trend2link', methods=['GET', 'POST'])
def trend2Sentiment():
	Sentiment = SentimentObject.sentimentSearchInput(Trend2)
	return render_template('index.html', sentiment = Sentiment, trend1 = Trend1, trend2 = Trend2, trend3 = Trend3, trend4 = Trend4, trend5 = Trend5);

@app.route('/trend3link', methods=['GET', 'POST'])
def trend3Sentiment():
	Sentiment = SentimentObject.sentimentSearchInput(Trend3)
	return render_template('index.html', sentiment = Sentiment, trend1 = Trend1, trend2 = Trend2, trend3 = Trend3, trend4 = Trend4, trend5 = Trend5);

@app.route('/trend4link', methods=['GET', 'POST'])
def trend4Sentiment():
	Sentiment = SentimentObject.sentimentSearchInput(Trend4)
	return render_template('index.html', sentiment = Sentiment, trend1 = Trend1, trend2 = Trend2, trend3 = Trend3, trend4 = Trend4, trend5 = Trend5);

@app.route('/trend5link', methods=['GET', 'POST'])
def trend5Sentiment():
	Sentiment = SentimentObject.sentimentSearchInput(Trend5)
	return render_template('index.html', sentiment = Sentiment, trend1 = Trend1, trend2 = Trend2, trend3 = Trend3, trend4 = Trend4, trend5 = Trend5);

if __name__ ==	"__main__":
	app.run(debug=True, threaded = False)
