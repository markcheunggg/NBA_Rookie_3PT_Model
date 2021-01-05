import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

"""
NBA Rookie-to-Sophomore 3P% Predictor by Mark Cheung

The following code predicts an inputted NBA player's increase in three-point percentage from their rookie season to
their second season using both a Random Forests model and an XGBoost model. The features used in these models are draft
age, draft position, and college & rookie season shooting stats. 

Additionally, by using K-Means Clustering, this code also lists previous NBA players who had similar early
career shooting statistics to the inputted player.
"""

"""
Import and read the data.
"""
# Dataframe of our training data (shooting stats from perimeter players drafted between 2010-2018).
NBA_shooting_stats = pd.read_csv("3PT Prediction Model Training Stats.csv")

# Dataframe of the players (and their respective stats) that we want to predict on (2019-20 rookie class).
rookies2019_20_stats = pd.read_csv("rookies.csv")


"""
Establish the statistics we want to use as features for our Random Forests model, XGB model, and K-Means Clustering.
"""
# The features used to predict 3P% improvement for the Random Forest model:
rf_predictors = ['Draft Position', 'Age', '3P% 1st Season', 'FT% 1st Season', '3P% College Improve',
              'Pre All Star - Post All Star 3P%', '3PA per 100 College', "3pr 1st season", "FT difference"]

# The features used to predict 3P% improvement for the XGBoost model:
xgb_predictors = ['Draft Position', 'Age', '3P% 1st Season', 'FT% 1st Season', '3P% College Improve',
                  'Pre All Star - Post All Star 3P%', '3PA per 100 College', '3PA per 100 1st season', "FT difference",
                  "3pr 1st season"]

# The features used to find similar historic shooters to a player via K-Means Clustering.
similarity_stat_predictors = ["3P% 1st Season", "3P% College", "3PA per 100 1st season", "3pr 1st season", "FT% 1st Season"]


"""
Preprocess the Data
"""
# Imputer used to fill in any blank entries with the respective column's mean values.
imputer = SimpleImputer(strategy="mean")

# Standard Scaler to scale all the data into standard units.
standardizer = preprocessing.StandardScaler()

# Impute and standardize all the data.
training_stats = NBA_shooting_stats.drop(columns=["Name"])
imputed_training_stats = pd.DataFrame(imputer.fit_transform(training_stats))
standardized_training_stats = pd.DataFrame(standardizer.fit_transform(imputed_training_stats.values))
imputed_training_stats.columns, standardized_training_stats.columns = training_stats.columns, training_stats.columns

# Get the features (X) from our training data for each of our models.
rf_X = standardized_training_stats[rf_predictors]
xgb_X = standardized_training_stats[xgb_predictors]
similarity_X = standardized_training_stats[similarity_stat_predictors]

# The stat (y) we are predicting for: change in 3P% from an NBA player's first to second season.
y = standardized_training_stats["3P% Improve"]

#The Random Forest and XGB Boost models.
rf_model = RandomForestRegressor(n_estimators=700, random_state=1)
xgb_model = XGBRegressor(random_state=0, n_estimators=600, learning_rate= .03)

def xgb_error():
    """
    Prints out the MAE and Median Absolute Error of the XGBoost Model using 5-fold cross validation.
    """
    mean_score = -1 * cross_val_score(xgb_model,xgb_X, y, cv=5, scoring='neg_mean_absolute_error')
    median_score = -1 * cross_val_score(xgb_model, xgb_X, y, cv=5, scoring='neg_median_absolute_error')

    # Converts MAE and Median Absolute error back to its original units.
    mean_score_reg_units = ((mean_score.mean() * np.std(imputed_training_stats["3P% Improve"])) + np.mean(imputed_training_stats["3P% Improve"])) * 100
    median_score_reg_units = ((median_score.mean() * np.std(imputed_training_stats["3P% Improve"])) + np.mean(imputed_training_stats["3P% Improve"])) * 100
    print("XGB cross-val mean absolute error:", mean_score_reg_units)
    print("XGB cross-val median absolute error", median_score_reg_units)

def rf_error():
    """
    Prints out the MAE and Median Absolute Error of the Random Forests Model using 5-fold cross validation.
    """
    mean_score = -1 * cross_val_score(rf_model, rf_X, y, cv=5, scoring='neg_mean_absolute_error')
    median_score = -1 * cross_val_score(rf_model, rf_X, y, cv=5, scoring='neg_median_absolute_error')

    # Converts MAE and Median Absolute error back to its original units.
    mean_score_reg_units = ((mean_score.mean() * np.std(imputed_training_stats["3P% Improve"])) + np.mean(imputed_training_stats["3P% Improve"])) * 100
    median_score_reg_units = ((median_score.mean() * np.std(imputed_training_stats["3P% Improve"])) + np.mean(imputed_training_stats["3P% Improve"])) * 100
    print("RF cross-val mean absolute error:", mean_score_reg_units)
    print("RF cross-val median absolute error:", median_score_reg_units)

def rf_predict_curr(name):
    """
    A function for predicting the change in 3P% from an NBA player's first season to second season using a Random
    Forests model.

    :param name: A string of the rookie's name.
    Prints out the predicted increase/decrease in 3P%.
    """

    # Get the inputted rookie's stats as a row from the rookie stat dataframe.
    specific_rookie_stats = rookies2019_20_stats.loc[rookies2019_20_stats["Name"] == name, :]

    # Iterate through the columns of the rookie's row of stats.
    for col in specific_rookie_stats:
        # If any stat is empty (likely due to an unusable sample size), replace it with the mean value of
        # that stat from our training dataframe.
        if (specific_rookie_stats[col].isnull().any()) & (col != ("3P% Improve")) & (col != ("3P% 2nd season")) & (col != "3PA per 100 2nd season"):
            mean = NBA_shooting_stats.loc[:, col].mean()
            specific_rookie_stats.loc[rookies2019_20_stats["Name"] == name, col] = mean
        elif ((col == "Name") | (col == "3P% Improve") | (col == "3P% 2nd season") | (col == "3PA per 100 2nd season")):
            continue
        else:
            # Put all of the rookie's stats in the standard units relative to the training data.
            specific_rookie_stats[col] = (specific_rookie_stats[col] - np.mean(imputed_training_stats[col])) / np.std(imputed_training_stats[col])

    # Fit the model on past rookies.
    rf_model.fit(rf_X, y)

    # Predict 3P% improvement based on the inputted rookie's stats.
    rookie_prediction = rf_model.predict(specific_rookie_stats[rf_predictors])
    # Convert the prediction into the original units.
    rookie_prediction = ((rookie_prediction * np.std(imputed_training_stats["3P% Improve"])) + np.mean(imputed_training_stats["3P% Improve"])) * 100
    if rookie_prediction >= 0:
        print("The Random Forests model predicts that " + name + "'s 3P% will increase by " + str(rookie_prediction)[1:-1] + "% next season.")
    else:
        print("The Random Forests model predicts that " + name + "'s 3P% will decrease by " + str(rookie_prediction)[1:-1] + "% next season.")

def xgb_predict_curr(name):
    """
    A function for predicting the change in 3P% from an NBA player's first season to second season using an XGBoost
    model.

    :param name: A string of the rookie's name.
    Prints out the predicted increase/decrease in 3P%
    """

    # Get the inputted rookie's stats as a row from the rookie stat dataframe.
    specific_rookie_stats = rookies2019_20_stats.loc[rookies2019_20_stats["Name"] == name, :]

    for col in specific_rookie_stats:
        # If any stat is empty (likely due to an unusable sample size), replace it with the mean value of
        # that stat from our training dataframe.
        if (specific_rookie_stats[col].isnull().any()) & (col != ("3P% Improve")) & (col != ("3P% 2nd season")) & (col != "3PA per 100 2nd season"):
            mean = NBA_shooting_stats.loc[:, col].mean()
            specific_rookie_stats.loc[rookies2019_20_stats["Name"] == name, col] = mean
        elif ((col == "Name") | (col == "3P% Improve") | (col == "3P% 2nd season") | (col == "3PA per 100 2nd season")):
            continue
        else:
            # Put all of the rookie's stats in the standard units relative to the training data.
            specific_rookie_stats[col] = (specific_rookie_stats[col] - np.mean(imputed_training_stats[col])) / np.std(imputed_training_stats[col])

    # Fit the model on past rookies.
    xgb_model.fit(xgb_X, y)

    # Predict 3P% improvement based on the inputted rookie's stats.
    rookie_prediction = xgb_model.predict(specific_rookie_stats[xgb_predictors])
    # Convert the prediction into the original units.
    rookie_prediction = ((rookie_prediction * np.std(imputed_training_stats["3P% Improve"])) + np.mean(imputed_training_stats["3P% Improve"])) * 100
    if rookie_prediction >= 0:
        print("The XGB model predicts that " + name + "'s 3P% will increase by " + str(rookie_prediction)[1:-1] + "% next season.")
    else:
        print("The XGB model predicts that " + name + "'s 3P% will decrease by " + str(rookie_prediction)[1:-1] + "% next season.")

def find_num_clusters():
    """
    Used for finding the optimal number of clusters to partition players via K-Means Clustering.

    Finds the sum of square distances (SSE) between centroids and the respective nodes of their clusters.
    Then plots the SSE per number of clusters to find where the drop-off in SSE occurs.
    """

    k_num = range(1, 20)
    sse = []
    for k in k_num:
        km = KMeans(n_clusters=k)
        km.fit(similarity_X)
        sse.append(km.inertia_)
    plt.plot(k_num, sse)
    plt.show()

def similarities(name):
    """
    A function that uses K-Means Clustering to find past NBA players that had similar early-career
    (college & rookie season) shooting stats to an inputted rookie.
    :param name: A string of the rookie's name.
    Prints out a list of the similar past NBA players.
    """

    # Get the inputted rookie's stats as a row from the rookie stat dataframe.
    specific_rookie_stats = rookies2019_20_stats.loc[rookies2019_20_stats["Name"] == name, :]

    for col in specific_rookie_stats:
        # If any stat is empty (likely due to an unusable sample size), replace it with the mean value of
        # that stat from our training dataframe.
        if (specific_rookie_stats[col].isnull().any()) & (col != ("3P% Improve")) & (col != ("3P% 2nd season")) & (col != "3PA per 100 2nd season"):
            mean = NBA_shooting_stats.loc[:, col].mean()
            specific_rookie_stats.loc[rookies2019_20_stats["Name"] == name, col] = mean
        elif ((col == "Name") | (col == "3P% Improve") | (col == "3P% 2nd season") | (col == "3PA per 100 2nd season")):
            continue
        else:
            # Put all of the rookie's stats in the standard units relative to the training data.
            specific_rookie_stats[col] = (specific_rookie_stats[col] - np.mean(imputed_training_stats[col])) / np.std(imputed_training_stats[col])

    # Get the features for the inputted rookie.
    rookie_similarity_stats = specific_rookie_stats[["Name", "3P% 1st Season", "3P% College", "3PA per 100 1st season",
                                                     "3pr 1st season", "FT% 1st Season"]]

    # Get the features from our training data that we'll be using to identify the similar players.
    similarity_training_stats = similarity_X
    similarity_training_stats["Name"] = NBA_shooting_stats["Name"]

    # Set up our K-means clusters.
    num_clusters = 6
    km = KMeans(n_clusters=num_clusters, max_iter=300, random_state=2)

    # Add the inputted rookie to the training dataframe of all the past rookies.
    training_data_with_rookie = similarity_training_stats.append(rookie_similarity_stats)
    player_names = training_data_with_rookie["Name"]
    training_data_with_rookie = training_data_with_rookie.drop(columns=["Name"])

    # Cluster players.
    y_predict = km.fit_predict(training_data_with_rookie)
    training_data_with_rookie['Cluster'] = y_predict

    # Find the cluster that the inputted rookie is located.
    training_data_with_rookie["Name"] = player_names
    cluster = training_data_with_rookie.loc[training_data_with_rookie["Name"] == name][["Cluster"]].values[0]

    # Get a list of players in that same cluster and print them out.
    similar_players_list = (training_data_with_rookie.loc[training_data_with_rookie["Cluster"] == int(cluster)][["Name"]].values[:-1])
    new_list = []
    for x in similar_players_list:
        new_list.append(str(x)[2:-2])
    print(new_list)

player_name = input("Enter a NBA player's name: ")
if player_name in rookies2019_20_stats.Name.values:
    # Predicts a Rookie's Shooting improvements using a Random Forests Model.
    rf_predict_curr(player_name)
    xgb_predict_curr(player_name)
    similarities(player_name)
else:
    print("Player does not exist. Check for spelling")
    exit()
