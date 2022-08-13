# ADLx22FinalProject - PUBG Placement Model
I cannot upload the dataset to github, but they are inside of the notebook on Kaggle: https://www.kaggle.com/code/aleksanderziobro/pubgplacementmodelforproject
## Introduction
The objective of this project was originally to demonstrate using a deep learning model on a dataset of statistics dumped from an API for the once popular battle royale game PlayerUnknown's Battlegrounds. However we hit roadblocks on the way and this quickly became a research project on how deep learning neural networks are not always the answer to your data set, and that sometimes going back to the basics with regression is the better choice. In our case, it would be gradient boosting regressors that do best, while our neural network performed poorly. 
## Selection of Data
We conducted preprocessing of the data using Pandas and Scikit on a dataset of about 65,000 matches of PUBG which equated to nearly 6.3 million individual player data entries. This contained a large number of features to work with from the distance players walked, the amount of kills they got, the amount of healing items used etc. Many of these features had little correlation to the placement of the placement of players in matches, while others had large correlations, so we were able to cut the data down to features we felt were most important in identifying how players would place in their matches. This would turn out to be the various distances traveled (swimming, walking, driving), the amount of kills, the amount of damage dealt to other players, and the amount of weapons acquired throughout the match. 
## Data Preview
Raw Data
![image](https://user-images.githubusercontent.com/54987160/184511168-59b846b2-522b-47b3-a9cd-754c2db9b754.png)
Once input into a Dataframe
![image](https://user-images.githubusercontent.com/54987160/184511180-4493f72f-7195-4987-8ba9-f9b94d6033fc.png)
Our group first preprocessed the data to shave off variables that were not well correlated to our goal/label we wanted to predict using data analysis such as heatmaps and graphs. Normalized our data since it is good practice, and then sent it through multiple different models to see what would score best. Initially our plan was just to use a Dense NN and tune it to get good results, but it seemed no amount of tuning was producing anything more than entry level results on the project, and this is where exploration into using gradient boosted regressors began. 
## Methods
Tools: -Scikit-learn, NumPy, Pandas, Tensorflow-Keras, Kaggle Kernels.
Scikit-learn features: Linear Regression, Histogram-based Gradient Boosting Regression Tree.  
## Results
![image](https://user-images.githubusercontent.com/54987160/184516797-faadf9f4-df5c-4203-af90-219c8a072fb8.png)
My submitted model using a Histogram-based Gradient Boosting Regression Tree with a basic level of data manipulation and preparation scored an MAE of 0.10597 which put us at approximately 1366/1529 on the leaderboard if the competition was still running.
![image](https://user-images.githubusercontent.com/54987160/184516854-5b2b2c9f-7a05-4bec-a8d9-df71547d19e1.png)
It's not a partiularly impressive entry, but our dense model scored almost in last place with a 0.16610, so just changing our approach was able to move us up over 100 places on the board. 
## Discussion
So what happened? Why is this the case? Shouldn't a simple tabular dataset like this get totally crushed by a complex neural network? Well... not really, and this ends up having some historical results both in the past and today. Neural Networks are great at producing results out of non-linear data, like vision problems. In those instances neural networks can excel. However, our PUBG dataset is tabular and has many factors with linear relationships. Linear relationships are where regressors are king and have had many optimized implementations over the years to improve results on data sets exactly like this. 
## Summary
## References
