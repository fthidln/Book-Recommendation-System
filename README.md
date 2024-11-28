# **Recommendation System: Automating Book's Suggestion using Collaborative Filtering**

By    : Muhammad Fatih Idlan (faiti.alfaqar@gmail.com)

This project was done to fulfil the *Machine Learning Terapan* 2nd assignment submission on Dicoding. The domain used in this project is book recommendation.

## Project Domain
In today’s digital age, the volume of content and choices available to users across platforms is overwhelming. Recommender systems play an indispensable role in navigating this vast landscape, ensuring users discover relevant and engaging content without being inundated by irrelevant options. By personalizing user experiences, these systems have become a cornerstone in industries like e-commerce, entertainment, and education, boosting user satisfaction, retention, and revenue. This project delves into the development of a book recommendation system, leveraging collaborative filtering techniques and the Nearest Neighbors algorithm to match users with books they are most likely to enjoy. Collaborative filtering, a widely used approach, relies on user-item interactions to uncover patterns and provide recommendations. Furthermore, Collaborative Filtering can make unforeseen recommendations, which means it might offer items that are relevant to the user even if the information is not in the user's profile [[ 1 ]](https://doi.org/10.1016/j.eij.2015.06.005). The system implementation is aimed to demonstrate how machine learning can be harnessed to create a seamless and personalized user experience in the context of literature discovery.

## Business Understanding
### Problem Statement
Starting with explanation from the background above, core problems that this project aims to solve are:

* How to develop a machine learning-based recommendation system for books?
* How are the results when data with and without normalization is compared using the same algorithm?

### Objectives
According to problem statement above, this project has several objectives too, that are:

* Develop a machine learning-based recommendation system for books
* Determining high performance model with variation of data preparation method

### Solution
To achive the objectives, we need to perform several things such as:

* Using Nearest Neighbour through variation of data with and without normalization to selecting high performance corresponding to evaluation metrics (Euclidean Distance)

## Data Understanding
![Data Understanding](/Assets/Kaggle.png "Data Understanding")
The dataset that used in this project is Book Recommendation Dataset, which can be accessed through kaggle [[ 2 ]](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset). This dataset consist of 3 csv files, Books.csv (271360 rows with 8 columns), Ratings.csv (1149780 rows with 3 columns), and Users.csv (27885 rows with 3 columns), also has 3 png file which irrelevant in this project. The explanation for each column can be seen below:

For Books.csv, the column are consist of:
* ISBN = International Standard Book Number of the books inside obtained from Amazon Web Services
* Book-Title = Title of the books obtained from Amazon Web Services
* Book-Author = The Author of the books obtained from Amazon Web Services
* Year-Of-Publication = Publication year of the books obtained from Amazon Web Services
* Publisher = The Publisher of the books obtained from Amazon Web Services
* Image-URL-S = URL for small sized Book's cover images point to the Amazon web site
* Image-URL-M = URL for medium sized Book's cover images point to the Amazon web site
* Image-URL-L = URL for large sized Book's cover images point to the Amazon web site

For Ratings.csv, the column are consist of:
* User-ID = Anonymized user identification in integers
* ISBN = International Standard Book Number of the books inside obtained from Amazon Web Services
* Book-Rating = Rating of the books, expressed on a scale from 1-10 (higher values denoting higher appreciation) in an explicit way, or expressed by 0 in implicit way

For Users.csv, the column are consist of:
* User-ID = Anonymized user identification in integers
* Location = Region of the reader in form of city, country
* Age = Age of the readers

### Exploratory Data Analysis (EDA)
Conducting exploratory data analysis, including statistical properties review with describe method and building correlation matrix for each variables to identify what variables are strongly related to the target variable.

#### Statistical Properties

|index|DEGC1PV|DEGC2PV|DEGC3PV|DEGC4PV|DEGC5PV|DEGC6PV|DEGC1SV|DEGC2SV|DEGC3SV|DEGC4SV|DEGC5SV|DEGC6SV|NM3/H\.1PV|NM3/H\.2PV|NM3/H\.3PV|NM3/H\.4PV|NM3/H\.5PV|NM3/H\.6PV|NM3/H\.1SV|NM3/H\.2SV|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|count|3840\.0|3840\.0|3840\.0|3840\.0|3840\.0|3840\.0|3840\.0|3840\.0|3840\.0|3840\.0|3840\.0|3840\.0|3840\.0|3840\.0|3840\.0|3840\.0|3840\.0|3840\.0|3840\.0|3840\.0|
|mean|939\.2851041666668|952\.687421875|1112\.672109375|1089\.26828125|1064\.5071875|1044\.1438802083333|1010\.6500000000001|1070\.0|1214\.9348958333333|1201\.3463541666667|1147\.6432291666667|1182\.91015625|2644\.1302083333335|3062\.5533854166665|3509\.8098958333335|7790\.544270833333|2049\.137760416667|5026\.225|2609\.134375|3013\.773958333333|
|std|135\.45595304145127|154\.76821993607717|178\.98477503964477|173\.77740241218478|158\.93625657797813|177\.66221850870477|14\.950441706365089|0\.0|10\.169618062337365|10\.608466161522355|25\.667546218462046|61\.20395788273972|1179\.3317961947675|1464\.637077204743|1600\.8417586831647|4059\.337145401351|841\.6380473776278|1323\.413675538872|1198\.357081977947|1487\.8200100610572|
|min|474\.1|455\.9|551\.6|493\.0|525\.7|496\.1|1005\.0|1070\.0|1180\.0|1180\.0|1100\.0|1100\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|977\.0|1488\.0|
|25%|860\.6500000000001|868\.2249999999999|993\.875|983\.275|1010\.95|965\.775|1005\.0|1070\.0|1210\.0|1200\.0|1140\.0|1130\.0|2090\.0|2296\.75|2536\.0|4227\.75|1538\.0|4268\.75|2139\.0|2214\.0|
|50%|995\.9|1029\.7|1215\.2|1188\.2|1142\.6|1136\.4|1005\.0|1070\.0|1215\.0|1200\.0|1140\.0|1180\.0|2447\.5|2882\.0|2982\.5|6931\.5|1897\.0|5017\.5|2495\.0|2897\.0|
|75%|1026\.8|1058\.4|1226\.725|1203\.7|1160\.4|1162\.725|1005\.0|1070\.0|1220\.0|1210\.0|1150\.0|1245\.0|2998\.5|3470\.0|4058\.0|11146\.75|2245\.25|6170\.5|2981\.5|3450\.5|
|max|1156\.0|1164\.9|1314\.6|1260\.4|1264\.7|1287\.4|1050\.2|1070\.0|1265\.0|1240\.0|1260\.0|1245\.0|12590\.0|15302\.0|12955\.0|15630\.0|6536\.0|9406\.0|12114\.0|14855\.0|

#### Multivariate Analysis
#### Correlation Matrix
![Correlation Matrix](Assets/CorrMat.png "Correlation Matrix")

#### Important Key Points from EDA
*   All DEGC2SV variable values are stagnant at 1070, so they have no impact on the target
*   Each variable has quite a lot of outlier values, but it is still retained because it can represent noise in real time
*   From correlation matrix above, we can conclude that NM3/H.1PV, NM3/H.2PV, NM3/H.1SV, and NM3/H.2SV is the most influencial variables to source input pressure, so we can drop the other unnacessary variables

## Data Preparation
Before model development step, it is inevitable to skip data preparation. This section is important, preparing data so the data that enter model development stage is not generating a trash model. It is start with data cleaning which removing duplicated data using pandas data frame method, drop_duplicates(). Later, principal component analysis is conducted to simplify dimension which removing redundance information. To fit the data into machine learning algorithm, splitting data into train and test set is necessary. This project use train_test_split from sklearn model selection using 37 as the random state, so  each time the code is run, it does not generate different splitting. The last thing to do is value standardization of principal component to perform efficiently by ensuring that different variables are treated on a common scale, since this project use an algorithm that rely in distance metrics (K-Nearest Neighbour). 

### Principal Component Analysis
This step is important, Principal Component Analysis (PCA) helps to eliminate redundancy by transforming the original features into a smaller set of uncorrelated variables (principal components), making the data easier to analyze by the model. Turns out that the most influencial principal component variance is 0.978, followed by 0.012 and 0.009. We can ignore the last two dimension because it has a very small variance corresponding to the first one [[ 3 ]](https://www.sciencedirect.com/science/article/pii/S1877050919321507). Thus simplify the problem that the models try to solve [[ 4 ]](https://royalsocietypublishing.org/doi/10.1098/rsta.2015.0202). 

### Spliting Dataset into Train and Test Set
To initiate the model development, splitting the data into train and test set is necessary. Moreover, this project using supervised learning. The train set serve as learning agent while test set serve as evaluating agent.

### Standardization
In order to scaling the dataset value, we can use standardization method. It transform the dataset in such a way to have a mean of 0 and standard deviation of 1. Moreover, standardization method is the superior scaling technique for medium and large dataset [[ 5 ]](https://ieeexplore.ieee.org/document/10681438).

## Model Development
To conduct model development, we have to divide variables in dataset into independent variable (y) and dependent variables (x). This project target variable (independent variable) is source input pressure (mmH2O) and dependent variables air flowrate also desired air flowrate in zone 1 and 2 (NM3/H.1PV, NM3/H.2PV, NM3/H.1SV, NM3/H.2SV). After that, fit the independent and dependent variable into each machine lerning algorithm and set several hyperparameter (if applicable). In this step, the algorithm used for model developments are K-Nearest Neighbour, Linear Regression, and Dense Neural Network.

* K-Nearest Neighbour = KNN is a simple, instance-based learning algorithm. It predict the target value for a new data point by averaging the target values of the K-nearest neighbors. To build machine learning model using K-Nearest Neighbour for this project, we need to decide the hyperparameter first. For K-Nearest Neighbour, the hyperparameters that will be set is the value of K which is 5 and brute algorithm employed to building model. While the rest of the hyperparameters are left by default.
 * Pros
   * Simple to understand and implement
   * No explicit training phase (lazy learning)
 * Cons
   * Computationally expensive for large datasets (due to distance calculations)
   * Sensitive to irrelevant or unscaled features
   * Performance depends on the choice of K and distance metric
* Linear Regression = Linear regression models the relationship between a dependent variable (target) and one or more independent variables (features) by fitting a linear equation to the data. This machine learning algorithm is the simplest model among other model used in the project. It requires no hyperparameters to set, because it is just fitting the data points into linear straight line.
 * Pros
   * Simple, interpretable model
   * Works well when there is a linear relationship between features and the target
 * Cons
   * Limited to linear relationships
   * Sensitive to outliers
   * Assumes no multicollinearity between features (when using multiple features)
* Dense Neural Network = A dense neural network (DNN) consists of layers of neurons where each neuron in one layer is connected to every neuron in the next layer (hence the term "fully connected"). Compare to other machine learning algorithm used in the project, this algorithm is the most complex. It is build using Sequential() model from tensorflow with 9 consecutive dense layer. The notation architecture for this model can be seen below:
![DNN_Figure](Assets/DNN_Figure.png "DNN_Figure")
 * Pros
   * Can model highly complex relationships between input and output
   * Scalable to large datasets and tasks like image recognition, natural language processing, etc
 * Cons
   * Requires a large amount of data and computational resources to train effectively
   * Prone to overfitting, especially with small datasets
   * Difficult to interpret compared to simpler models like linear regression

## Evaluation
The metrics evaluation used for this step is Mean Squared Error<br><br>
$$MSE(y, x) = \frac{\sum_{i=0}^{N - 1} (y_i - x_i)^2}{N}$$
<br>

Where:

* N = Amount of the data
* i = Index of the data
* y = Actual value
* x = Predicted value

MSE is a metric used to measure the average squared difference between the predicted values and the actual values in the dataset. It is calculated by taking the average of the squared residuals, where the residual is the difference between predicted value and the actual value for each data point [[ 6 ]](https://www.geeksforgeeks.org/mean-squared-error/). A lower MSE indicates that the model's predictions are closer to the actual values signifying better accuracy. While, a higher MSE suggests that the model's predictions deviate further from true values indicating the poorer performance.

### Performance of Each Machine Learning Algorithm
![HistPerform](Assets/HistPerform.png "HistPerform")

|index|train|test|
|---|---|---|
|KNN|1731\.4725651041665|4193\.04046875|
|Linear Regression|5030\.204432958909|5404\.623603896129|
|ANN|2773\.7929275104184|4416\.747012630459|

From metric evaluation table above, we can conclude that K-Nearest Neighbour algorithm is the most desired algortihm because has the lowest MSE value in train and test set, followed by Dense Neural Network, and the last is Linear Regression.

### Model Prediction
This step is carried out to see how each machine learning algorithm predicting the target data (source pressure).

|index|y\_true|dimension|LR|KNN|ANN|
|---|---|---|---|---|---|
|770|601|0\.25246995242719095|580\.21152119611|590\.6|577\.8556518554688|

![Prediction Scatter](Assets/RealPredScatter.png "Prediction Scatter")

From the figure above, we can compare how prediction data and real data from each machine learning algorithm (K-Nearest Neighbour, Linear Regression, Dense Neural Network). Clearly, Linear Regression generated data point in a straight line. K-Nearest Neighbour generated data points that gather in one area. Then Dense Neural Network seems to struggle with its predictions forming a smoother but lower curve that doesn't capture the wide spread of real data.

### Conclusion
After building this project, we can answer the problem statement and fulfil our objectives that we set before in business understanding section. Also, by implementing solution statement, we can easily achieve them (the answer of problem statements and fulfil the objectives), due to heatmap visualization of correlation matrix to understand interaction of each variables and the usage of MSE metrics to identify the best model from several machine learning algorithm (K-Nearest Neighbour, Dense Neural Network, and Linear Regression).
* From correlation matrix visualization using heatmap, we can see that among all variables that do not have a strong influencial to dependent variable, source input pressure (mmH2O), there is several variables have a correlation point up to 0.5 indicating high influential presence, that are Air flowrate in zone 1 (NM3/H.1PV), Air flowrate in zone 2 (NM3/H.2PV), Desired air flowrate in zone 1 (NM3/H.1SV), and Desired air flowrate in zone 2 (NM3/H.2SV).
* With correaltion point up to 0.5 for NM3/H.1PV, NM3/H.2PV, NM3/H.1SV, and NM3/H.2SV indicating that all this variables is positively related with our target variable. It means that the bigger value of independent variables, the bigger value for dependent variable is generated.
* Using MSE metrics, we can conclude that the K-Nearest Neighbour algorithm is the best algorithm for this project with MSE value for train set is 1731.47 and test set is 4193.04 making it the lowest. Followed by Dense Neural Network with MSE value for train set is 2773.79 and test set is 4416.75 and the last algorithm is Linear Regression with MSE value for train set is 5030.2 and test set is 5404.62.

## Reference

*   [ 1 ] J. Conradt, “A comparison between a traditional PID controller and an Artificial Neural Network controller in manipulating a robotic arm,” 2019. Accessed: Oct. 22, 2024. [Online]. Available: https://www.semanticscholar.org/paper/A-comparison-between-a-traditional-PID-controller-a-Conradt/efb1c57c0dbc3b88cd35085f677869104fce5474

*   [ 2 ] “Smart Pressure Control Prediction.” Accessed: Oct. 23, 2024. [Online]. Available: https://www.kaggle.com/datasets/guanlintao/smart-pressure-control-prediction

*   [ 3 ] N. Salem and S. Hussein, “Data dimensional reduction and principal components analysis,” Procedia Computer Science, vol. 163, pp. 292–299, Jan. 2019, doi: 10.1016/j.procs.2019.12.111.

*   [ 4 ] I. T. Jolliffe and J. Cadima, “Principal component analysis: a review and recent developments,” Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences, vol. 374, no. 2065, p. 20150202, Apr. 2016, doi: 10.1098/rsta.2015.0202.

*   [ 5 ] K. Mahmud Sujon, R. Binti Hassan, Z. Tusnia Towshi, M. A. Othman, M. Abdus Samad, and K. Choi, “When to Use Standardization and Normalization: Empirical Evidence From Machine Learning Models and XAI,” IEEE Access, vol. 12, pp. 135300–135314, 2024, doi: 10.1109/ACCESS.2024.3462434.

*   [ 6 ] “Mean Squared Error | Definition, Formula, Interpretation and Examples,” GeeksforGeeks. Accessed: Oct. 23, 2024. [Online]. Available: https://www.geeksforgeeks.org/mean-squared-error/
