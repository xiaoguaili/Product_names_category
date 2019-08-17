# Machine Learning Data Scientist Test

## Part 1: Order Item Name
### Outline:
In order to predict the categories of products, a reference dataset (Amazon product dataset ../website or reference#) is adopted to train the model. The reference dataset is divided into training set and test set. Using a naive bayes model, 44 categories are predicted for the test set, with 75% accuracy. (The result of XGBoost model is coming soon.)

### 1. Goal: 
For the online transactions, we have the information about the product items that people buy,
such as the table below shows. There are about 12K product item names in the attached dataset
( ecommerce_product_names.csv ).

a.) If we want to understand which catalog (such as clothing, shoes, accessories, beauty,
jewelry etc.) each item is, how will you make that happen?

b.) How can you extract the additional information from the item names, such as the color,
style, size, material, gender etc. if there is any?

### 2. Methods and Modeling Process: 

Since training set is not provided in the spreadsheet, an Amazon product dataset[1, 2] is adopted to help train machine learning model. Over 9.4 million products are collected from May 1996 to July 2014. The dataset contains name, description, price, sales-rank, brand info, and co-purchasing link of each product. Other reference datasets could also be used in model training, which will be discussed in Discussion Section.

#### Step 1. Clean reference dataset

1.1 Clean unuseful data: e.g. sales-rank, brand info, and co-purchasing links; the categories that don't appear in target dataset; rare categories

1.2 Filter 'macro' categories from category tree

1.3 Split 'Clothing, Shoes & Jewelry' category into individual categories, as expected by our goal

Codes: cleanData.ipynb

Input: input/metadata.json.gz

Output: output/selectedAmazonProductData_final.csv

#### Step 2. Predict categories of target dataset

2.1 Load data; convert each category name in reference dataset to a numerical variable

2.2 Represent text as numerical data: build vocabulary with all product data; convert text into matrix by word appearance

2.3 Train naive bayes model/XGBoost classifier to predict categories, which achieves 75% accuracy score (naive bayes) in test set. 

2.4 Categories of product in target dataset are exported.

Codes: predictCategories.ipynb

Input: output/selectedAmazonProductData_final.csv

Output: output/ProductWithCategories.csv

### 3. Results: 

The reference dataset is divided into training set and test set. Using a naive bayes model, 44 categories are predicted for the test set, with 75% accuracy.

### 4. Discussion and Future Work: 


1. This model serves as an outline for this problem due to limited time, where details could be further improved. Naive_bayes was implemented as the main base estimator. Other models are still worthwhile to be trained. For instance, random forest classifier, SVM classifier, logistic classifier or neural network. Sepecially, bagging and boosting could be applied to achieve higher accuracy. Different models need to be compared in training time, space complexicity, accuracy, etc.

2. Amazon product dataset is used as the reference dataset in this model. However, the target dataset seems related to Indian retail e-commerce, so there exists some discrepancies in languague preferences. For instance, 'ballies' in the target dataset refers to 'ballet flat' or 'ballerina flat' in US markets generally, which causes touble in category prediction. The idea case is that the reference dataset shares same languague preferences with the target dataset.  

3. As for the additional information such as the color, style, size, material, gender etc., the easiest way to implement it is to build several dictionaries of these properties respectively. For example, dictionary 'color' includes words like 'red', 'blue', etc. As the text is already converted to numerical matrix, it is not expensive to filter these properties from the matrix. The dictionaries are small, as the common descriptions of these properties are limited.

4. In this model, 44 categories were considered, where the categories are well broken down. In real application, coarser categories may be required, for higher accuracy. The target dataset was labeled 24 categories, and 20 categories were not predicted by the model. Therefore, removing the data in the 20 unused categories during model training may help improve efficiency and accuracy. 

5. Cross validation is needed as future work.

6. More metrics could be used to quantify performace of the model.

7. Other methods and packages to deal with text could also be considered in this problem. For instance, NLTK. 


### 5. Summary: 
Naive bayes model is trained with a reference dataset and achieved 75% accuracy in the test set. Products in the target dataset are well classified into 24 out of 44 categories and exported.

### References:   
 [1] Ups and downs: Modeling the visual evolution of fashion trends with one-class collaborative filtering,  R. He, J. McAuley, WWW, 2016

 [2] Image-based recommendations on styles and substitutes, J. McAuley, C. Targett, J. Shi, A. van den Hengel, SIGIR, 2015


## Part 2: Credit Card Fraud Detection Model

### Q a.) How are you going to use these variables for training a machine learning model in order to detection fraud? Any variables can you derive from these variables?

First, feature engineering should be done, as the text data should be transformed to numerical data and new features should be generated.

New features: 

Consistency between shipping and billing information / Consistency between billing and recorded consumer information:

1. whether shipping address is same with billing address
2. whether shipping name is same with billing name
3. whether shipping phone number is same with billing phone number

4. whether shipping address is same with recorded consumer address
5. whether shipping name is same with recorded consumer name
6. whether shipping phone number is same with recorded consumer number

7. whether shipping address is valid
8. whether the billing address is valid



### Q b.)  What additional information / variables / features do you want to collect (assuming we will get the user’s permission) in order to better catch the fraudulent transactions, and why? How do you plan to use these extra variables you are going collection?

Transaction history of the card: 

1. whether shipping city ever appears in transaction history
2. whether this transction is consistent with the consumer's consumption habits (common transaction time during a day/categories) 
3. transaction frequency (past week/month/year)
4. time interval between the 'fraud' transaction and most recent transaction in this credit card
5. income (detect large amount fraud)
6. transaction amount of the 'fraud' transaction

Data of high risk phone number: whether a lot of fraud happened to the shipping phone number in history (See from fraud history data of the phone number)

Data of confirmed fraud transaction in nearby states within a rerent period: data of the above all features

#### Postprocessing

All 'whether' features labels variables as 1 or 0. The other numerical features are used as variables directly. Furether, the historical data of the card are labeled as non-fraud in the following model training, while data of confirmed fraud transaction is labeled as fraud. 

### Q c.)  What kind of machine learning model do you want to build to detect the fraud credit card transactions? More specifically, what machine learning algorithms and techniques are you going to use, and why?

The efficiency of different algorithms may vary according to different dataset. For general cases, logistic regression and XGBoost classifier would be first considered to train the model. Logistic regression is known to be reasily interpreted, not expensive, robust to small noises, not sensitive to collinearity. In [1], compared to LDA/KNN/RF/SVM, logistic regression and XGBoost classifier are shown to have good predictability in a similar fraud detect problem. Moreover, XGBoost classifier takes special care of the sparse matrix, which possibly happens in fraud detect.

For unbalanced dataset, SMOTE+ENN is suggested by [2], where the precision could be penalized and the recall score could be increased. Recall score may be more significant in fraud detection, because the increase in false positive rate may potentially cause losing clients due to wrongfully classifying transactions as fraud, as well as increasing operational costs for cancelling credit cards, printing new ones and posting them to the clients.

Moreover, well-developed techniques are reported in literature. As summirized by [3],  Whitrow et al. (2009)[4] describe in great length how they have derived data attributes through aggregation. Bhattacharyya et al. (2010)[5] report data attributes including primary attributes and derived attributes. Bahnsen et al. (2016) expand the transaction aggregation strategy, and create features based on the periodic behavior of the time of a transaction using the von Mises distribution. These studies then apply supervised learning techniques: support vector machines (SVM) and random forests (RF) etc. in detecting credit card fraud. Sundarkumar & Ravi (2015) [6] list the data attributes in their banking and insurance fraud detection application. Vlasselaer et al. (2017) [7] develop their network-based features by examining the network of credit card holders and merchants and deriving a time-dependent suspiciousness score for each network object.

### References:  
[1] https://towardsdatascience.com/detecting-credit-card-fraud-using-machine-learning-a3d83423d3b8

[2] https://towardsdatascience.com/detecting-financial-fraud-using-machine-learning-three-ways-of-winning-the-war-against-imbalanced-a03f8815cce9

[3] https://medium.com/@Dataman.ai/how-to-create-good-features-in-fraud-detection-de6562f249ef

[4] Whitrow, Christopher, et al. "Transaction aggregation as a strategy for credit card fraud detection." Data mining and knowledge discovery 18.1 (2009): 30-55.

[5] Bhattacharyya, Siddhartha, et al. "Data mining for credit card fraud: A comparative study." Decision Support Systems 50.3 (2011): 602-613.

[6] Sundarkumar, G. Ganesh, and Vadlamani Ravi. "A novel hybrid undersampling method for mining unbalanced datasets in banking and insurance." Engineering Applications of Artificial Intelligence 37 (2015): 368-377.

[7] Van Vlasselaer, Véronique, et al. "APATE: A novel approach for automated credit card transaction fraud detection using network-based extensions." Decision Support Systems 75 (2015): 38-48.

  
