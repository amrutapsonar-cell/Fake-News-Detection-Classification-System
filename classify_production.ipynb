{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "8ad5fc0d-de80-4c81-92fd-bed0116b7961",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Production Samples (Top 5):\n",
      "                                                    text  prediction\n",
      "70754  Hillary Clinton Waiting In Wings Of Stage Sinc...           0\n",
      "70755  UNITED NATIONS  —   Defying extraordinary pres...           1\n",
      "70756  This is an old video that exposes so much abou...           0\n",
      "70757  Ukrainian soldiers caught deserting are being ...           0\n",
      "70758  Support for New Zealand s opposition Labour Pa...           1\n"
     ]
    }
   ],
   "source": [
    "import sqlite3\n",
    "import pandas as pd\n",
    "import joblib\n",
    "import re\n",
    "\n",
    "def preprocess_text(text):\n",
    "    if not isinstance(text, str): return \"\"\n",
    "    text = text.lower()\n",
    "    text = re.sub(r'[^a-z\\s]', '', text)\n",
    "    return text\n",
    "\n",
    "def run_production():\n",
    "   # 1. Load Live Records - add .copy()\n",
    "    conn = sqlite3.connect('Classification.db')\n",
    "    df = pd.read_sql_query(\"SELECT * FROM fake_news_classification\", conn)\n",
    "    live_df = df.iloc[55031+15723:].copy()\n",
    "    conn.close()\n",
    "\n",
    "    # 2. Load Assets\n",
    "    model = joblib.load('fake_news_model.pkl')\n",
    "    tfidf = joblib.load('tfidf_vectorizer.pkl')\n",
    "\n",
    "    # 3. Preprocess & Predict\n",
    "    live_df['clean_text'] = live_df['text'].apply(preprocess_text)\n",
    "    X_live = tfidf.transform(live_df['clean_text'])\n",
    "    \n",
    "    live_df['prediction'] = model.predict(X_live)\n",
    "\n",
    "    # 4. Show Sample Results\n",
    "    print(\"Production Samples (Top 5):\")\n",
    "    print(live_df[['text', 'prediction']].head())\n",
    "    \n",
    "    # Save results to CSV for review\n",
    "    live_df.to_csv('production_results.csv', index=False)\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    run_production()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f36e3290-19b0-4fcb-8ab2-af200e36f8d9",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
