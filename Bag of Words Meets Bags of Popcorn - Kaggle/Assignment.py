import pandas as pd 
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression

import re, string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def main_ml():
    print("Loading train and test data.....")
    print("")
    train = pd.read_csv('........path of the training file (tsv file).........', delimiter="\t")
    test = pd.read_csv('........path of the test file (tsv file).........', delimiter="\t")
    

    print("Pre-processing training data....HOLD ON, this might take some time...")
    print("")
    corpus = []
    count = 0
    for i in range(0,train.shape[0]):
        review = re.sub('[^a-zA-Z]', ' ', train['review'].values[i])  #keeping only alphabets
        review = review.lower()  #lowering
        review = review.split()
        lemma = WordNetLemmatizer()
        review = [lemma.lemmatize(word) for word in review if not word in set(stopwords.words('english'))] #removing stopwords
        review = ' '.join(review)
        corpus.append(review)

        count += 1
        if count%1000 == 0:
            print("Completed "+ str(count) + " of " + str(train.shape[0]) + " rows ")

    

    #joining the cleaned reviews to the main train file and saving it as a csv file
    train['cleaned_review'] = corpus
    
    print("Pre-processing done....now splitting the data into train-test-split: 70:30 ratio..")
    print("")
    X = train['cleaned_review']
    y = train['sentiment']


    train_X,test_X, train_y, test_y = train_test_split(X,y, test_size=0.3, random_state=0) #splitting into 70:30


    print("Applying the best model on this dataset...after hyperparameter tuning")

    logistic = LogisticRegression(C= 2, tol= 0.001, solver= 'sag')
    print(logistic)
    print("")
    

    ngram_unibi = CountVectorizer(analyzer = "word",
                                           tokenizer = None,
                                           preprocessor = None,
                                           ngram_range = (1,2),
                                           strip_accents = 'unicode',
                                           max_features = 10000)




    ngram_unibi_train = ngram_unibi.fit_transform(train_X)
    ngram_unibi_test = ngram_unibi.transform(test_X)

    tfidf_transformer = TfidfTransformer()
    ngram_unibi_tfidf = tfidf_transformer.fit_transform(ngram_unibi_train)
    ngram_unibi_tfidf_test = tfidf_transformer.transform(ngram_unibi_test)

    logistic.fit(ngram_unibi_tfidf, train_y)

    print("Training completed...results on the validation set are:")
    ngram_unibi_tfidf_pred = logistic.predict(ngram_unibi_tfidf_test)
    
    print("Confusion matrix..")
    print(confusion_matrix(test_y,ngram_unibi_tfidf_pred))
    print("")
    print("Classification Report..")
    print(classification_report(test_y,ngram_unibi_tfidf_pred))
    print("")


    # preprocessing test data

    print("Pre-processing test data....HOLD ON, this might take some time...")
    print("")
    corpus_test = []
    for i in range(0,test.shape[0]):
        review = re.sub('[^a-zA-Z]', ' ', test['review'].values[i])
        review = review.lower()
        review = review.split()
        lemma = WordNetLemmatizer()
        review = [lemma.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus_test.append(review)

        count += 1
        if count%1000 == 0:
            print("Completed "+ str(count) + " of " + str(test.shape[0]) + " rows ")
    

    test['cleaned_review'] = corpus_test
    
    print("Transforming the test data accordingly...")
    print("")
    test_text = test['cleaned_review']
    test_text_bow_10000feat = ngram_unibi.transform(test_text)
    test_text_bow_10000feat = tfidf_transformer.transform(test_text_bow_10000feat)
    pred = logistic.predict(test_text_bow_10000feat)
    predicted = pd.DataFrame()
    predicted['id'] = test['id']
    predicted['sentiment'] = pred
    predicted.to_csv("test_predictions_DhrubajitDas.csv", index=False)
    print("saved prediction file as 'test_predictions_DhrubajitDas.csv'....")


if __name__ == '__main__':
    main_ml()
