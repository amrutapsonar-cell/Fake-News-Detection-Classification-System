{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "b1c384ce-8951-41b3-bbc6-924054789565",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Evaluation Results:\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.94      0.96      0.95      8730\n",
      "           1       0.95      0.93      0.94      6993\n",
      "\n",
      "    accuracy                           0.95     15723\n",
      "   macro avg       0.95      0.94      0.94     15723\n",
      "weighted avg       0.95      0.95      0.95     15723\n",
      "\n",
      "Model and Vectorizer saved.\n"
     ]
    }
   ],
   "source": [
    "import sqlite3\n",
    "import pandas as pd\n",
    "import joblib\n",
    "import re\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import classification_report, accuracy_score\n",
    "\n",
    "def preprocess_text(text):\n",
    "    \"\"\"Consistent text cleaning for training and inference.\"\"\"\n",
    "    if not isinstance(text, str): return \"\"\n",
    "    text = text.lower()\n",
    "    text = re.sub(r'[^a-z\\s]', '', text)\n",
    "    return text\n",
    "\n",
    "def train():\n",
    "    # 1. Load Data\n",
    "    conn = sqlite3.connect('Classification.db')\n",
    "    query = \"SELECT * FROM fake_news_classification\"\n",
    "    df = pd.read_sql_query(query, conn)\n",
    "    conn.close()\n",
    "\n",
    "       # 2. Split Data per requirements - add .copy() to the end\n",
    "    train_df = df.iloc[:55031].copy()\n",
    "    eval_df = df.iloc[55031:55031+15723].copy()\n",
    "    \n",
    "    # 3. Preprocess (The error will now disappear)\n",
    "    train_df['clean_text'] = train_df['text'].apply(preprocess_text)\n",
    "    eval_df['clean_text'] = eval_df['text'].apply(preprocess_text)\n",
    "\n",
    "    # 4. Feature Extraction (TF-IDF)\n",
    "    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')\n",
    "    X_train = tfidf.fit_transform(train_df['clean_text'])\n",
    "    y_train = train_df['label'] # Assuming column name is 'label'\n",
    "    \n",
    "    X_eval = tfidf.transform(eval_df['clean_text'])\n",
    "    y_eval = eval_df['label']\n",
    "\n",
    "    # 5. Model Selection: Random Forest\n",
    "    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)\n",
    "    model.fit(X_train, y_train)\n",
    "\n",
    "    # 6. Evaluation\n",
    "    preds = model.predict(X_eval)\n",
    "    print(\"Evaluation Results:\")\n",
    "    print(classification_report(y_eval, preds))\n",
    "\n",
    "    # 7. Save Model and Vectorizer\n",
    "    joblib.dump(model, 'fake_news_model.pkl')\n",
    "    joblib.dump(tfidf, 'tfidf_vectorizer.pkl')\n",
    "    print(\"Model and Vectorizer saved.\")\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    train()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a507ddb4-96c2-4996-8db0-6f1b453fb4ae",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "744a379d-0246-4b66-97a6-42934dc0cad6",
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
