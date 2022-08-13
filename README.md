# ADLx22FinalProject - PUBG Placement Model
I cannot upload the dataset to github, but they are inside of the notebook on Kaggle: https://www.kaggle.com/code/aleksanderziobro/pubgplacementmodelforproject
##Introduction
The objective of this project was originally to demonstrate using a deep learning model on a dataset of statistics dumped from an API for the once popular battle royale game PlayerUnknown's Battlegrounds. However we hit roadblocks on the way and this quickly became a research project on how deep learning neural networks are not always the answer to your data set, and that sometimes going back to the basics with regression is the better choice. In our case, it would be gradient boosting regressors that do best, while our neural network performed poorly. 
##Selection of Data
We conducted preprocessing of the data using Pandas and Scikit on a dataset of about 65,000 matches of PUBG which equated to nearly 6.3 million individual player data entries. This contained a large number of features to work with from the distance players walked, the amount of kills they got, the amount of healing items used etc. Many of these features had little correlation to the placement of the placement of players in matches, while others had large correlations, so we were able to cut the data down to features we felt were most important in identifying how players would place in their matches. This would turn out to be the various distances traveled (swimming, walking, driving), the amount of kills, the amount of damage dealt to other players, and the amount of weapons acquired throughout the match. 
##Data Preview
Raw Data
![image](https://user-images.githubusercontent.com/54987160/184511168-59b846b2-522b-47b3-a9cd-754c2db9b754.png)
Once input into a Dataframe
![image](https://user-images.githubusercontent.com/54987160/184511180-4493f72f-7195-4987-8ba9-f9b94d6033fc.png)

##Methods
##Discussion
##Summary
##References
