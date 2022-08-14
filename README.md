# ADLx22FinalProject - PUBG Placement Model
I cannot upload the dataset to github, but they are inside of the notebook on Kaggle: https://www.kaggle.com/code/aleksanderziobro/pubgplacementmodelforproject
## Introduction
The objective of this project was originally to demonstrate using a deep learning model on a dataset of statistics dumped from an API for the once popular battle royale game PlayerUnknown's Battlegrounds. However we hit roadblocks on the way and this quickly became a research project on how deep learning neural networks are not always the answer to your data set, and that sometimes going back to the basics with regression is the better choice. In our case, it would be gradient boosting regressors that do best, while our neural network performed poorly. 
## Selection of Data
We conducted preprocessing of the data using Pandas and Scikit on a dataset of about 65,000 matches of PUBG which equated to nearly 6.3 million individual player data entries. This contained a large number of features to work with from the distance players walked, the amount of kills they got, the amount of healing items used etc. Many of these features had little correlation to the placement of the placement of players in matches, while others had large correlations, so we were able to cut the data down to features we felt were most important in identifying how players would place in their matches. This would turn out to be the various distances traveled (swimming, walking, driving), the amount of kills, the amount of damage dealt to other players, and the amount of weapons acquired throughout the match. The dataset for this can be found here. https://www.kaggle.com/competitions/pubg-finish-placement-prediction/data
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
Our submitted model using a Histogram-based Gradient Boosting Regression Tree with a basic level of data manipulation and preparation scored an MAE of 0.10597 which put us at approximately 1366/1529 on the leaderboard if the competition was still running.
![image](https://user-images.githubusercontent.com/54987160/184516854-5b2b2c9f-7a05-4bec-a8d9-df71547d19e1.png)
It's not a partiularly impressive entry, but our dense model scored almost in last place with a 0.16610, so just changing our approach was able to move us up over 100 places on the board. 
## Discussion
So what happened? Why is this the case? Shouldn't a simple tabular dataset like this get totally crushed by a complex neural network? Well... not really, and this ends up having some historical results both in the past and today. Neural Networks are great at producing results out of non-linear data, like vision problems. In those instances neural networks can excel. However, our PUBG dataset is tabular and has many factors with linear relationships. Linear relationships are where regressors are king and have had many optimized implementations over the years to improve results on data sets exactly like this. The most commonly brought up approach in the discussions for this dataset were decision tree based regressors such as XGBoost and LightGBM, which are models tuned for this exact kind of data, and scale incredibly well with data that is meticulously prepared. Tabular data is often time best represented using deterministic models like a tree than the probalistic model of a neural network. 

In gradient boosting decision trees, the behavior is similar to that of a random forest at face value, you generate many trees that are weaker models to combine them into a stronger model. Where the two differ is that for Gradient Boosted Trees, you build your trees in series, and tweak your future trees using the errors of the trees before them. This is done by training future trees using the residual errors of the trees before them as the labels for the new data. 

If we had more time to tackle this we would take more steps in the data preprocessing area. Namely, we would try to reintroduce the "matchType" categorical variable as dummy values to use in the regression as we think this feature is still very important to the accuracy of predictions. Additionally, there may be ways to optimize the redundancy of features namely: Boosts vs health, the three types of distances, the multiple types of kills, to better make predictions. 
## Summary
Could our dense network have done even better than it did? Likely, but it probably would have required a significantly larger amount of data manipulation than would be required to get good scores using these gradient boosted decision trees. In the interest of time required for training and the amount of setup to get decent results, if your data is heterogenous and tabular, look towards gradient boosting decision tree models first, you may be surprised by their capabilities. 
## References
[PUBG Finish Placement Prediction (Kernels Only)] https://www.kaggle.com/competitions/pubg-finish-placement-prediction/

[When and Why Tree-Based Models (Often) Outperform Neural Networks
] https://towardsdatascience.com/when-and-why-tree-based-models-often-outperform-neural-networks-ceba9ecd0fd8


[Boost then Convolve: Gradient Boosting Meets Graph Neural Networks] https://deepai.org/publication/boost-then-convolve-gradient-boosting-meets-graph-neural-networks#:~:text=Graph%20neural%20networks%20%28GNNs%29%20are%20powerful%20models%20that,learning%20methods%20when%20faced%20with%20heterogeneous%20tabular%20data.


[XGBoost] https://www.geeksforgeeks.org/xgboost/
