Text-Summarizer
A multi-technique text summarization system built with Streamlit, combining BERT-based extractive summarization, WordNet-based abstractive compression, and query-driven interactive summarization, with ROUGE/METEOR evaluation and word cloud visualization.
Overview
This interactive NLP application summarizes text using three complementary approaches:
Extractive Summarization — a BERT-based model (bert-base-uncased) scores sentences and selects the top-ranked ones based on a user-defined percentage of the original text.
Abstractive Compression — condenses sentences by removing stopwords and substituting remaining words with WordNet synonyms.
Interactive Summarization — extracts sentences relevant to a user-specified query/topic.
The app also evaluates summary quality against a reference summary using ROUGE and METEOR scores, and generates a word cloud visualization of the resulting summary.
Technologies Used
Language: Python
Web Framework: Streamlit
NLP / ML: 
oHugging Face Transformers (BERT — bert-base-uncased)
oPyTorch
oNLTK (tokenization, stopwords, WordNet, METEOR score)
oRouge (ROUGE score evaluation)
Visualization: Matplotlib, WordCloud
Other: NumPy


Prerequisites
Python 3.8+
pip
Run Locally
Clone the project
git clone https://github.com/<your-username>/TextSummarizer.git
Go to the project directory
cd TextSummarizer
(Recommended) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
Install dependencies
pip install -r requirements.txt
If you don't have a requirements.txt yet, create one with:
streamlit
torch
transformers
nltk
rouge
matplotlib
numpy
wordcloud
Download required NLTK data
This is handled automatically on first run (nltk.download(...) calls in the script), but you can also run it manually:
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
Start the app
streamlit run app.py
The app will open in your browser at http://localhost:8501.
How to Use
1.Paste or type your input text in the Input Text field.
2.Provide a Reference Summary to enable ROUGE/METEOR evaluation.
3.Adjust the Percentage slider to control how much of the text is retained in the extractive summary.
4.Click Summarize to generate: 
oThe extractive summary
oThe compressed (abstractive) summary
oThe interactive/query-relevant summary
oROUGE and METEOR evaluation scores
oA word cloud visualization
Project Structure
TextSummarizer/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md
