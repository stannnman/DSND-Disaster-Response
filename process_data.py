import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):

    # Load messages and categories from their respective CSV files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Merge the messages and categories datasets using the common id
    return messages.merge(categories, how='outer', on='id')


def clean_data(df):

    # create a dataframe of the 36 individual category columns
    categories_split = df['categories'].str.split(pat=';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories_split.iloc[0]
    
    # use this row to extract a list of new column names for categories
    category_colnames = row.apply(lambda x: x[:-2])
    
    # rename the columns of categories
    df = pd.concat([df, pd.DataFrame(columns=category_colnames)], axis=1)
    
    # convert category values to just numbers 0 or 1
    for column in category_colnames:
        df[column] = df['categories'].apply(\
            lambda x: 0 if x.find(column+'-1')==-1 else 1)
    
    # drop the original categories column  
    df.drop(axis=1, labels='categories', inplace=True)
    
    # drop any duplicate rows
    df = df[~df.duplicated(keep='first')]
    
    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('Messages', engine, index=False)
    return  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
