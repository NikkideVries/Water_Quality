# imports: 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def evaluate_random_forest(X_train, y_train, X_val, y_val, max_depth=4, random_state=123):
    '''
    This function will create a random forest model on given specifications, and print out a table. 
    
    '''
    # Make the model
    forest = RandomForestClassifier(max_depth=max_depth, random_state=random_state)

    # Fit the model (on train and only train)
    forest.fit(X_train, y_train)

    # Use the model
    in_sample_accuracy = forest.score(X_train, y_train)
    out_of_sample_accuracy = forest.score(X_val, y_val)

    # Create a DataFrame to display results
    output = {
        "max_depth": [max_depth],
        "train_accuracy": [in_sample_accuracy],
        "validate_accuracy": [out_of_sample_accuracy]
    }

    table = pd.DataFrame(output)
    table["difference"] = table["train_accuracy"] - table["validate_accuracy"]

    return table




def test_random_forest(X_train, y_train, X_val, y_val, X_test, y_test, max_depth=4, random_state=123):
    '''
    This function will create a random forest model on given specifications, and print out a table. 
    
    '''
    # Make the model
    forest = RandomForestClassifier(max_depth=max_depth, random_state=random_state)

    # Fit the model (on train and only train)
    forest.fit(X_train, y_train)

    # Use the model
    in_sample_accuracy = forest.score(X_train, y_train)
    out_of_sample_accuracy = forest.score(X_val, y_val)
    test_accuracy = forest.score(X_test, y_test)

    # Create a DataFrame to display results
    output = {
        "max_depth": [max_depth],
        "train_accuracy": [in_sample_accuracy],
        "validate_accuracy": [out_of_sample_accuracy],
        "test_accuracy": [test_accuracy]
    }

    table = pd.DataFrame(output)

    return table

