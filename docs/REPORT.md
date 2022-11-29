# Augmenta: House Prices Project

## Abstract

In this report, we created a linear regression model using the Jax python library to predict the sale price of homes in the Ames Iowa Housing Dataset. First, we preprocessed the data by imputing missing values and encoding categorical values. Next, we applied a polynomial feature transformation and hyper-parameter optimization to improve the performance of our model. Then, we compared the performance of multiple pre-built models to our Jax linear regression, including, but not limited to, XGBRegressor which uses gradient boosted random forests, in order to determine which models worked best under a given set conditions. We concluded that the polynomial feature transformation produced a significant improvement in the jax linear regression, however, XGBRegressor performed the best out of our models regardless of data processing. 

## Introduction 

The problem in focus is predicting the sale price of residential homes in Ames, Iowa based on their features including square footage, construction material, age, condition, and location among many others. This problem is of great interest since the complexity of the training data offers an array of preprocessing strategies, which significantly impact the performance of machine learning algorithms. Although the data is complex, it can be learned by a variety of algorithms ranging over several levels of sophistication, which makes it a helpful tool for analyzing different implementations of linear regression. 

## Related Work

1. https://coax.readthedocs.io/en/latest/examples/linear_regression/jax.html
2. https://github.com/pantelis-classes/PRML/blob/master/prml/linear/_linear_regression.py
3. https://xgboost.readthedocs.io/en/stable/python/python_api.html
4. https://www.kaggle.com/code/ryanholbrook/feature-engineering-for-house-prices
This introductory google collab was used as a reference in order to implement our data processing and test different encoding methods such as label, one-hot, and a combination of both.
5. https://www.youtube.com/watch?v=aOsZdf9tiNQ&t=635s
This linear regression technique performed using Jax was used as a reference in order to implement our linear regression using transformations of polynomial degree = 2 for testing.
6. https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview

In [Incorporating Multiple Linear Regression in Predicting the House Prices Using a Big Real Estate Dataset with 80 Independent Variables](https://www.scirp.org/journal/paperinformation.aspx?paperid=115003), author Azad Adbulhafedh attempts to create a model with multiple regression that accurately predicts the prices of home in the Ames Housing Dataset. One of our approaches will be similar to this approach because we will utilize a random forest in XGBRegressor.

The paper [Housing Price Prediction Based on Multiple Linear Regression](https://www.hindawi.com/journals/sp/2021/7678931/) by Qingqi Zhang attempts to predict house prices for a dataset of homes sold in Boston using multiple linear regression. Our approach will differ from this approach because we are using a single linear regression.

## Data

The data that we are working with is a collection of numeric values and categorical features describing an individual house in Ames, Iowa. Numeric features include Lot square footage, pool area, etc. Categorical features include house style, garage condition, etc. The dataset contains 79 features about a specific home, and 1460 rows of house data. The data was collected in 2011. 

The dataset needed preprocessing in order to be used for regression. Preprocessing of the dataset was broken up into three main parts: cleaning, categorizing, and handling missing values. 

During the cleaning phase of preprocessing, the description of the dataset was analyzed and compared to the actual data in order to check for any inconsistencies in our training data. Analysis revealed that the dataset contained some typos that differed from the specification of the data.

After fixing any typos or inconsistencies, the dataset was categorized. During this phase of the preprocessing, the categorical feature-set was split into nominative features and ordinal features. This process was done by-hand in order to increase the effectiveness of the encoding that will be done later. 

In the final phase of preprocessing, the dataset was prepped to handle missing values. Developing a strategy for handling missing values requires understanding which features are missing values and how they should be treated. The following graph shows the top 6 features missing values.

   ### MissingValues
   | ![missing values](https://github.com/hahdookin/cs301/blob/main/images/MissingValues.png) |
   |:--:|
   | **Figure 1**: Shows the data fields with missing values |

After understanding why these features are missing values by reading their descriptions in the data description, appropriate action was taken to handle these missing values.

## Methods

Housing prices were predicted using three different linear regression models: 
* [1] implementation of linear regression in Jax `jreg()`
* [2] implementation of linear regression from the pmrl library as described in “Pattern Recognition and Machine Learning” `LinearRegression()`
* [3] `XGBRegressor()` from the XGBoost library

Jax was chosen for our own linear regression implementation for its ease of use and accelerated linear algebra compiler, which was shown to reduce model training time by half when compared to an accelerated regression. The LinearRegression() model from prml was chosen as a baseline (“standard”) model to compare our implementations with. Finally, `XGBRegressor()` was chosen since it is extremely powerful and provides a reasonable upper bound on the performance of our model for a given data processing strategy. 

To predict housing prices from the data set taken from Kaggle, we first process our data by imputing missing values and encoding categorical data types into numerical values, which is an essential step since linear regression can only be trained with numeric data. To achieve such, we take two separate approaches: label encoding and one-hot encoding. Label encoding involves assigning unique integers to a corresponding value of a categorical feature, whereas one-hot encoding splits each categorical feature’s possible values into separate columns and assigns them a binary value. When passing the parameter “both” through the data initialization both methods are used, in which label encoding was applied to ordinal features and one-hot encoding was applied to nominative features.

After the data has been encoded we then can choose to take a subset of the features with highest correlation to sale price and/or transform it using the `PolynomialFeature` function from the prml library. Taking only a subset of the features helps ensure the model does not overfit the training data, and it was shown that the difference between the minimized training loss and test loss decreased with less features. A polynomial transformation was chosen to capture the nonlinear relationships between the model features and was shown to improve the performance of the jax linear regression model. To determine which combinations of models and processing strategies performed best, we tested each linear regression model across different processing methods. 

The last step in building each model is tuning their hyperparameters, for `JReg()` this mainly involves changing the learning rate and number of iterations to produce a training loss with exponential decay resembling the graph below. 

   ### Loss over epochs
   | ![loss vs epoch](https://github.com/hahdookin/cs301/blob/main/images/LossVsEpoch.png) |
   |:--:|
   | **Figure 2**: Loss decreasing over epochs |

When tuning XGBRegressor there are a slew of hyperparameters to optimize, to accomplish such we use the hyperopt python library to find the best combination of [colsample_bytree, learning rate, max tree depth, number of estimators, and subsample] to minimize the test loss from XGBRegressor. 

## Experiments

   ### Preliminary Result: Cross-Validation and Linear Regression Using the Training Data
   | ![cross validation](https://github.com/hahdookin/cs301/blob/main/images/Cross-ValidatedPreliminary_trainingdata.png) |
   |:--:|
   | **Figure 3**: Determining data fit using linear Regression. After plotting the training data and applying linear regression, it was determined that lower priced houses were better fit than houses with a higher price |

Because of the types of data present in our dataset, encoding methods need to be used in order for our regression to utilize the entire dataset. The following pie chart shows the distribution of data types present in our dataset. Where the columns correspond to the number of fields in the dataset.

   ### Data Distributions
   | ![Data type distribution](https://github.com/hahdookin/cs301/blob/main/images/DataDistribution.png) |
   |:--:|
   | **Figure 4**: Data was found to take three types of value that included integer, float, and categorical type |

After viewing the pie chart, it is evident that over half of our dataset consists of non-numeric columns, which will not be used in a regression algorithm. It is clear that simply ignoring columns that contain categorical data will severely hinder the success of our model. Multiple encoding techniques were considered. After much consideration, a combination of label and one-hot encoding was used. Label encoding was applied to the ordinal features and one-hot encoding was applied to the nominative features. We then tested each encoding strategy with each model implementation, and recorded the performance of each model within a single encoding strategy (for figures below, “label” encoding was used). 

   ### Linear Regression Models
   | ![Different Linear Regression Models](https://github.com/hahdookin/cs301/blob/main/images/DifferentMachineLearningResults.png) |
   |:--:|
   | **Figure 5**: A visualization of predictions from three linear regression models plotted against the corresponding ground truth home sale prices. (Figures from left to right are from sources [1], [2], [3] respectively). | Comparing rmse between jax linear regression and xgboost

   ### Results of Encoding Techniques
   | ![Encoding Techniques](https://github.com/hahdookin/cs301/blob/main/images/DifferentEncodingResults.png) |
   |:--:|
   | **Figure 6**: Comparing rmse between our implementation versus other kaggle |

This data shows a clear supremacy for XGBRegressor(), but we will now show how a polynomial transformation of a subset impacts its performance as well as the jax regression implementation. 

   ### Dimensionality and Model Performance
   | ![Performance](https://github.com/hahdookin/cs301/blob/main/images/model-transforms.PNG) |
   |:--:|
   | **Figure 7**: Comparing the performance of models with different dimensionalities |

This data shows that reducing the dimensionality of the data improves the performance of `JReg()`, but hurts the prediction accuracy of `XGBRegressor()`. For both models however, the transformation of data into a 2nd degree polynomial decreased the test loss, but by an order of magnitude less than by reducing the number of features. Test loss was further reduced in the `XGBRegressor()` model by tuning the hyperparameters, however it did not make a significant difference when compared to the impact of changing the feature subset. 


## Conclusion

Through the models created and different feature engineering methods, we concluded that XGBRegressor with feature-engineering performed the best, producing the lowest RMSE. We learned that feature-engineering was far more important than hyper-parameter optimization for our specific problem, as it increased the effectiveness of XGBRegressor by a higher order of magnitude than hyper-parameter optimization.

Future improvements to solving this problem would include using a multiple linear regression model as well as more complex machine learning techniques such as ridge regression. Additionally, the performance of the linear regression models tested could likely be improved by encoding categorical data types by their frequency and by transforming the data set with basis functions to specific features, as opposed to. 

In summary, attempting this problem was challenging but enjoyable and served as a good introduction to working with popular machine learning libraries and their applications to real-world problems. 
