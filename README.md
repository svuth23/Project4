# Project4_Group3
# Stroke Prediction Model Comparition
### Date : 
21/03/2024
# Data Source: 
https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

# File description:
## IMPORTANT NOTE - Run files on google colab ##
1) "Data" folder - contains data from https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
2) "Kmeans_240319.ipynb" - python code for Kmeans clustering
3) "Kmeans_Prac_Himali.ipynb" - python code for Kmeans clustering trained only on two features
4) "Kmeans_PCA_240319.ipynb" - python code for Kmeans clustering after reducing the number of features in the dataset
5) "keras_model.ipynb" - python code for neural network - Keras model
6) "AlexF_Project4.ipynb" - Python code for Logistric regression model & optimisation of model
7) "UsingRandomforestmodel (4).ipynb" - Python code for Random forrest model .
Python: The programming language used for data analysis and modeling.


# Prerequisite:

- Pandas: A powerful library for data manipulation and analysis.

- NumPy

- Scikit-learn: A machine learning library providing simple and efficient tools for data mining and data analysis.

- Matplotlib.

- Seaborn.

 - Google Colab: Interactive computing environment .....etc
 
# General description:

“Globally, the World Health Organisation estimates that there are at least 62 million stroke survivors. In Australia, 50,000 people suffer a stroke each year, leaving them with physical and mental disabilities that create an enormous emotional, social, and financial burden on our community.” (https://qbi.uq.edu.au/brain/brain-diseases/stroke/stroke-facts#:~:text=Strokes,financial%20burden%20on%20our%20community).

# Aim:
Our group aims to utilize a well documented dataset from https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset to generate models to predict wether a patient likely to have stroke or not based on vary features such as age, average glucose level, bmi, etc. To do that, we had explored unsupervised (Kmeans, PCA) and supervised (neural network, logistic regression, random forest) machine learning methods. 


# Objectives:

(1) Data import and pre-processing (PySpark, Pandas) by remove outlier, null value, non-informative features.

(2) Building predicting models (K-means, PCA, Neural Network,Random forest model …)

(3) Compare and suggest the best model


# Objective 1 - Data import and pre-processing
- Import data using Spark
- Inspect data
- Features: Id, gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status
- Label: stroke
  
  <img width="787" alt="image" src="https://github.com/AlexFeeney/Project4_Group3/assets/136966712/4332f9c9-b8fc-461b-ae26-6d274e97beb5">


## Data Pre-processing
- Remove row with “Other” value from “gender” feature
- Remove non-informative feature “Id”
- Remove rows with “N/A” value in the “bmi”
- Remove missing value from “stroke”


# Objective 2 - Building predicting models  
## K-MEANS clustering:

<img width="767" alt="image" src="https://github.com/AlexFeeney/Project4_Group3/assets/136966712/471f9aea-a47f-448d-a248-eac4d54ff24d">

  
<img width="674" alt="image" src="https://github.com/AlexFeeney/Project4_Group3/assets/136966712/91666c7d-3355-4323-b21f-8df20d6da7fb">


K-mean clustering method appears to cluster the patients differently when compared to actual stroke record from patients. Therefore, using K-mean clustering might not be a good option for this dataset.

## K-means clustering trained only against "BMI" and "Average glucose levels" 

![kmeans](https://github.com/AlexFeeney/Project4_Group3/assets/145959658/769387ee-92d5-4773-a45b-aff3228e97e4)
Figure 1: Visualisation of true datapoints K-means trained only against "BMI and average glucose levels"

![scatterplot](https://github.com/AlexFeeney/Project4_Group3/assets/145959658/cbf5f933-d47f-4531-b7f8-606be680d269)
FIgure 2: Scatter plot displaying stroke status 



Both figures show the true data points but figure 1 visualises 2 centroids  of the the clusters k means identified when trained only against BMI and average glucose levels.The colours assigned to each cluster is based on the individual data point's distance from the centroid it’s closest to.

Figure 2 is a scatter plot showing the same data points but whether they have had a stroke or not. 

When you compare the two, the yellow cluster seems to contain more people who have had strokes, represented by the blue dot concentration in the bottom right corner of figure 2. You can infer if you have a low BMI and higher glucose levels, you are more likely to have had a stroke. However, there are so many red dots due to the skewed nature of the dataset, making it difficult to see if there are more blue dots in the purple cluster. 

![vy](https://github.com/AlexFeeney/Project4_Group3/assets/145959658/e0ddcb42-3290-41bd-a559-0cff5c7bfc82)


If you refer to the scatter plot visualising the true stroke results of the previously trained K-means you can see more strokes present towards the bottom left corner, infering if you have low BMI and lower glucose levels, you are more likely to have had a stroke. 

Due to the skewed nature of this dataset,  K-means clustering is not an good option to use as a predictive tool. 





## K-mean clustering with PCA
-Image shows k-mean clustering with PCA

<img width="747" alt="image" src="https://github.com/AlexFeeney/Project4_Group3/assets/136966712/6f0f2dbd-9a1e-47fe-a558-d438916fa65d">


## Random Forest Model:
-To achive this model,convert data to fit in to model
-All values converted in to binary values except age, bmi, avg_glucose level.

<img width="448" alt="image" src="https://github.com/AlexFeeney/Project4_Group3/assets/136966712/009ca13e-5427-40e2-b5f2-86ee99bc6c69">

- define X and y: X is features or independent variables, Y is targeted variable 

-  Split data into training and testing set by 80%:20%
   image shows spliting data and shiffling data samples code :

   <img width="587" alt="image" src="https://github.com/AlexFeeney/Project4_Group3/assets/136966712/3b5349fe-71fe-42d5-8126-e29cc92692cd">



### Create random forest classifier and train Model

- Creating  and train random forest classifier using code below:

<img width="464" alt="image" src="https://github.com/AlexFeeney/Project4_Group3/assets/136966712/f8b6eca5-6ce1-457d-b4bf-1224710b055c">



- Accuracy is 93-94%

  <img width="382" alt="image" src="https://github.com/AlexFeeney/Project4_Group3/assets/136966712/56737509-c3fe-4de1-97cf-e95d9bfef884">


  -Below  table shows 1-11 itterations  of RFM :
  
 <img width="413" alt="image" src="https://github.com/AlexFeeney/Project4_Group3/assets/136966712/42c9507d-32af-4e58-8c8e-df87fbd4baf6">
 

- important Features of data : using this data we can train model and achive goal with high accuracy level 
 
 <img width="509" alt="image" src="https://github.com/AlexFeeney/Project4_Group3/assets/136966712/61dcddf2-4526-457d-bc9e-ded649d5a1b5">

 ![image](https://github.com/AlexFeeney/Project4_Group3/assets/136966712/e1353f18-8498-4d09-9d20-582d312cd8f7)

 

 - Further investigation and improvement are necessary and by using to features and training model ,we can achieve goal
   

# Logistic Regression:

- Build Model using numeric features
  Age, Hypertension, Heart Disease (True or False), Average Glucose Level

  <img width="418" alt="image" src="https://github.com/AlexFeeney/Project4_Group3/assets/136966712/31ac4dd1-87bc-4408-9699-cf040ed3395e">


  - Image shows, High accuracy score with given features to predict stroke status.

     <img width="431" alt="image" src="https://github.com/AlexFeeney/Project4_Group3/assets/136966712/e23ba502-e46b-463e-a895-9a49ada666e6">

# Neural network - Keras model 

<img width="757" alt="image" src="https://github.com/AlexFeeney/Project4_Group3/assets/136966712/448a7b07-0755-43b9-be10-6bd73619d8f8">

- model structure
  
  <img width="386" alt="image" src="https://github.com/AlexFeeney/Project4_Group3/assets/136966712/79633426-5e2c-454d-8d8d-d6220ca7c743">

  - Model evaluation on test dataset : Acuracy is 96%. Changing hyperparameters seem to not further improve the model.
 
  <img width="561" alt="image" src="https://github.com/AlexFeeney/Project4_Group3/assets/136966712/4998c3ac-87c7-4e4b-95e2-10968155fbaf">

# Conclution:

Our conclusion is supervised machine learning approaches have a great potential to predict wether a person likely to have stroke or not with the accuracy can be up to 96% in neural network - Keras model. 

# About:
-These projects were completed as part of Data Analysis Bootcamp.

  By :
  Alex Feeney,
  Vy Nguyen, 
  Swapna, 
  Himali

