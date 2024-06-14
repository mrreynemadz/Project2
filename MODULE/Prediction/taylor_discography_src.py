
import streamlit as st

st.header('Taylor Swift Discography Analysis and Model')
st.subheader('This python code is implemented for Streamlit')

st.subheader('Data Preparation')
st.code('''
        import pandas_profiling as p_prof
        import pandas as pd_basic

        # Display the count of unique albums
        # Filter the dataset for '1989'
        datasetCSV = pd_basic.read_csv('./DATASET/taylor_discography.csv')
        datasetCSV.head
        single_data = datasetCSV[datasetCSV['album'] == '1989']
        # # print(datasetCSV['album'].unique())
        # # Display the optimal values for '1989'
        # print(single_data.describe())
        #report = p_prof.ProfileReport(datasetCSV, explorative=True)
        # report
        report = p_prof.ProfileReport(single_data, title="Taylor Swift's Discography", explorative=True)
        report
    ''')

st.subheader('Machine Learning Model Training')
st.code('''
        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score
        import pandas as pd

        # Load the dataset
        datasetCSV = pd.read_csv('./DATASET/taylor_discography.csv')

        # Preprocess the data
        X = datasetCSV[['loudness', 'speechiness', 'acousticness']]
        y = datasetCSV['track_name']

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Define the parameter grid
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': [1, 0.1, 0.01, 0.001],
            'kernel': ['rbf', 'poly', 'sigmoid']
        }

        # Create a GridSearchCV object
        # grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)

        # Choose a machine learning model
        grid = RandomForestClassifier(n_estimators=100)

        # Train the model
        grid.fit(X_train, y_train)

        # Print the best parameters
        # print(grid.best_params_)

    ''')

st.subheader('Performance Test (Accuracy)')
st.code('''
        # Evaluate the model
        y_pred = grid.predict(X_test)
        print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

        # Use the model to make predictions
        def track_name(loudness, speechiness , acousticness):
            return grid.predict([[loudness, speechiness , acousticness]])

        # print(track_name(30, 70))
        ''')
