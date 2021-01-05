# NBA Rookie 3P% Improvement Model
The following code predicts an inputted NBA rookie's increase in three-point percentage from their first season to
their second season using both a Random Forests model and an XGBoost model. The features used in these models are draft
age, draft position, and college & rookie season shooting stats. 

Additionally, by using K-Means Clustering, this code also lists previous NBA players who had similar early-career shooting statistics to the inputted player.

# Features used (in standard units):
- Draft Position
- Draft Age
- Rookie season 3P%
- Rookie season FT%
- Change in a player's final collegiate season 3P% to their rookie season 3P%.
- Change in a player's final collegiate season FT% to their rookie season FT%.
- Change in a player's Pre-ASB 3P% to their Post-ASB 3P%
- 3PA per 100 possessions in college.
- 3PA per 100 possessions in rookie season.
- 3Pr in first season of the NBA.

# Example results:
Ja Morant: RF Model predicts: decrease by .05 3P%. XGBoost model predicts: increase by 2.5% 3P%

R.J. Barrett: RF Model predicts: increase by 1.7% 3P%. XGBoost model predicts: Increase by 3.0% 3P%.

