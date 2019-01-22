import sys
import nltk
import numpy as np
import pandas as pd
import pickle

from sqlalchemy import create_engine
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    '''
    Load data from sqllite database

    Input: database_filepath - path of sqllite database to load

    Output: None
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('Messages', engine)
    X = df['message']
    Y = df.drop(axis=1, labels=['id','message','original','genre'])
    category_names = Y.columns.values
    return X, Y, category_names


def tokenize(text):
    '''
    Transform text into tokens

    Input: text - line of text to tokenize

    Output: list of tokens
    '''
    # tokenize text
    tokens = word_tokenize(text)

    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []

    for tok in tokens:
        
        # lemmatize, normalize case, and remove white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Build classification pipeline model

    Input: None

    Output: classification nodel
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ])
    
    # List of parameters used for permutation in grid search tuning
    parameters = {
        'tfidf__use_idf': [False, True],
        'clf__n_estimators': [50, 100],
        'clf__min_samples_split': [2, 4, 8]
    }

    # Using grid search to find optimal set of parameters
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)

    return pipeline


def get_performance(Y_test, Y_pred):
    '''
    Get performance metrics for the output of our classification

    Input:
        Y_test - the actual labels for our test set
        Y_pred - the predicted labels for our test set

    Output: dataframe with performance metrics for each output label
    '''
    col_perf = []

    for col in Y_test.columns.values:
        print(classification_report(Y_test[col], Y_pred[col]))

        col_accuracy = (Y_test[col] == Y_pred[col]).mean()
        col_precision = precision_score(Y_test[col], Y_pred[col])
        col_recall = recall_score(Y_test[col], Y_pred[col])
        col_f1_score = f1_score(Y_test[col], Y_pred[col])
        
        col_perf.append([col_accuracy, col_precision, col_recall, col_f1_score])
        
    df_perf = pd.DataFrame(col_perf, index=Y_test.columns.values,\
                      columns=['accuracy', 'precision', 'recall', 'f1-score'])

    print(df_perf.mean())
    
    return df_perf
    

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate performance of our classification model

    Input:
        model - classification model
        X_test - text input from our test set
        Y_test - actual labels from our test set
        category_names - list of output labels (categories)

    Output: None
    '''

    Y_pred = pd.DataFrame(model.predict(X_test), index=Y_test.index.values,\
         columns=category_names)
    
    df_perf = get_performance(Y_test, Y_pred)
    return


def save_model(model, model_filepath):
    '''
    Save model to disk so it can be consumed by our web app

    Input:
        model - classification model
        model_filepath - path where model is saved

    Output: None
    '''

    pickle.dump(model, open(model_filepath, 'wb'))
    return


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
