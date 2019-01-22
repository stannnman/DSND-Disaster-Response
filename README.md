# Disaster Response Message Classification #

Udacity Data Science Nanodegree Data Engineering Project

### Key Deliverables ###
- **process_data.py** - Load data from CSV, clean and transform it, and save to a sqllite database.
- **train_classifier.py** - Create a text classification pipeline, tune it and save it to disk.
- **run.py** - Run a web app, displaying charts and providing user interface for interactive text classification.

### Libraries Used ###
- numpy and pandas for data manipulation
- flask and plotly for web interface and data viz.
- sklearn for data manipulation and machine learning.
- pickle for dumping model to disk.

### Execution ###
- To run ETL pipeline that cleans data and stores it in a database<br/>
`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        
- To run ML pipeline that trains classifier and saves it to disk<br/>
`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

- To run the Flask web app.<br/>
`python run.py`

- Go to http://0.0.0.0:3001/ to load web app in a browser.
