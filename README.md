| Member           | Github Profile |
|------------------|--------|
| Christopher Pane | https://github.com/hahdookin |
| David Salazar    | https://github.com/NJSal |
| Dylan Lederman   | https://github.com/DylanLederman |

The Kaggle competition for this problem can be found [here](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview).

## The Problem
The problem that we will be investigating is predicting the sales price of residential homes in Ames, Iowa based on their features including house type, construction material, age, condition, and location among many others. 

This problem is interesting because it utilizes advanced machine learning algorithms to recreate how humans evaluate home prices artificially. Such technology could be of great use for determining fair market prices for homes and for predicting future home prices. 

## Reading Material
The reading material that we will use to understand the context and background of this problem:
+ [“Predicting property prices with machine learning algorithms”](https://www.tandfonline.com/doi/full/10.1080/09599916.2020.1832558)
+ [“Gradient Boosted Trees with XGBoost and SciKit Learn”](http://s3.amazonaws.com/MLMastery/xgboost_with_python_sample.pdf)

## The Data
The data that we will be using is the Ames Housing dataset. This dataset consists of 80 features of houses in Ames, Iowa sold between 2006 and 2010. Examples of these features include how many stories the house has, the shape of the property, etc.

## The Method
Our proposed method of solving this problem is by using a linear regression algorithm accelerated by gradient boosting. Most solutions posted utilize either a regression or a neural network; our implementation will be similar with an added focus on optimization of algorithm efficiency and accuracy using XGBoost.

## Evaluating the Results

### Qualitative
We will evaluate the results of our model qualitatively with use of plots to see how well our hypothesis fits the data.

### Quantitative
To evaluate our results we will use the test.csv data posted under the Kaggle competition page to quantify an MSE between our model predictions and test data. We will then compare this value against other submissions posted to the Kaggle page. The training duration will also be used to quantify how quickly our algorithm can learn from training data and will serve as a key metric for the performance of our code.
