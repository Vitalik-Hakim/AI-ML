# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 09:31:50 2022

@author: Vitalik Hakim
"""

`Algorithm` -- A set of rules and statistics techiniques used
to learn patterns from data

`Model` -- A Model is trained using a ml algorithm

`Predicted Variable` -- A feature of data that can be used to predict the output.
 `response Variable` -- it is a feature or the output variable that needs to be predicted by using the predictor models
 
 `Training Data` -- The machine learning model is built using training data
 
 `Testing Data` -- The ML model is evaluated using the testing data
 
## Machine learning process

The machine learning process involves building a predictive model that can be used to find a solution for a problem statement

1. Define obejective
2. Data gathering
3. Preparing Data
4. Data Exploration
5. Building a Model
6. Model Evaluation
7. Predictions
 
# Example one (Predict the weather in local area)
## Step 1. To predict the possibility of rain by studying the weather conditions
Questions to ask
*  What are we trying to predict?
* What are the target features?
* What is the input data?
* What kind of problem are you facing
ie. Binary classification or clustering

## Step 2. Data Gathering

Data such as weather conditions humidity level temperature pressure etc
either collected manually or scraped from the web
Online resuources such as Kaggle


## Step 3. Data preparing (Data cleaning)

This method involves getting rid of inconsistencies in data such as missing values or redundan variables

So you need to:

Transform data into desired format

Data cleaning:
    Missing Values
    Corrupted Data
    Remove unnecessary data

## Step 4. Exploratory Data Analysis

Data exploration involves underatanding the patterns and trends
in the data. At this stage all the useful insights are drawn and correlations
between the variables are understood.

## Step 5. Building a machine learning model

At this stage a preditive model is built using machine learning algorithms such as 
 LInear regression Decision trees etc
 
 Machine learning is built using the training data set
 The model is the machine learning algorithm that predicts the output by using the data fed into it

## Step 6. Model Evaluation & Optimization

The effienciency of the model is evaluated and any further improvement in the model are implemented

    Machine learning model is evaluated using the testing data set
    The accuracy of the model is calculated 
    Further improvement in the model are done by using techniques like PARAMETER TUNING

## Step 7. Predictions

The final outcome is predicted after performing parameter tuning and improving the accuracy model



# Types of Machine Learning

## Supervised Learning

This is a techique in which we teach or train the machine
using data which is well labelled.
 
 Using lablelled data to train a machine learning algorithm 

Types of problems:
    Regression and classification
Popular Algorithms:
    Linear regression
    Logistic regression
    Support Vector Machine
    KNN.
 
## Unsupervised learning

This is the training of machine using information that is unlabeled and allowing the algorithm to act on that
information without guidance.

Using unlabeled data to train a ml algorithm by finding similarities and grouping them in clusters

Types of problems:
    Association and clustering
Popular algorithms:
    K-Means
    C-Means

## Reinforcement Learning

This is a part of machine learning where an agent is put in an evnvironment and he learns to behave in this environment
by performing certain actions and observing the rewards which it gets from those actions

Types of problems:
    Reward based
Popular algorithms:
    Q-learning
    SARSA
    
# Types of problems solved using Machine Learning

## Regression
* Supervised learning
* Output is a continous Quantity
* Main aim is to forecast or predict
* Example is predict stock market price
* Algorithm: Linear Regression

## Classification
* Supervised Learning
* Output is a catergorical quantity
* Main aim is to compute category of the data
* Eg. Classify emails as spam or non-Spam
* Algorithm: Logistic regression
 
## Clustering
 * Unsupervised Learning
 * Assigns data points into clusters
 * Main aim is to group similar items into clusters
 * Eg. Find all transactions which were fraudulent in nature
 * Algorithm: K-means



# Supervised Learning Algorithms

## Linear Regression

Linear regression is a method used to predict dependent variable (Y) based on the value of independent variables(X)
It can be used for cases where we wanrt predict some continous quantity.

  Dependent variable (Y) (Continous)
      This is the response value that needs to be predicted
      
  Independent Variable(X) (Discrete or continous) 
      The predictor Variable used to predict the response variable
    
    Linear Regression Equation
        Y = B0 + B1X + e
        Y = is the Y variable or dependent variable
        B0(Beta-zero) = Y intercept
        B1(Beta-one) = Slope(Gradient)
        X = Independent variable or X variable
        E = Error
 [Y = mx + c]
        
        
We can find a relationship by using a linear fit line

## Logistic Regression

This is a method used to predict a dependent variable given a set of independent variables
 such that the dependent variable is categorical.

  Represent a relationship between p(X)=Pr(Y=1/X) and X?
  
  We take the exponetial of the equation to make the whole thing positive
  Since the exponential of any value is a positive number
  Secondly a number divided by itself + 1 will always be less than 1
  eg 45/45+1
  
  Hence the formula = P(X) = e(B0 + B1x) //
                             e(B0 + B1x) + 1 
    Logic Function
    P(X) = e(B0 + B1x) //
           e(B0 + B1x) + 1 
          Cross multiply
    p(e(B0 + B1x) + 1) = e(B0 + B1x)
    p.e(B0 + B1x) + p = e(B0 + B1x)
    p = e(B0 + B1x) -  p.e(B0 + B1x) 
    p = e(B0 + B1x) (1-p)
    p//(1-p) = e(B0 + B1x) 
    ln[p//(1-p)] = (B0 + B1x)
 Used for classification
 The output variable is always a categorical variable
 
## Decision Tree

This is a supervised machine learning algorithm which looks
like an inverted tree wherein each node represents a predictor variable
the link between the nodes represents a decision and each leaf node represents an outcome

Structure of a decision tree

Root Node: This is the starting point of a tree. At this point the first split is performed

Internal Nodes: Each internal node represents a decision point(predictor variable) 
that eventually leads to the prediction of the outcome

Leaf / Terminal Nodes: Leaf nodes represent the final class of the outcome and therefore they're' are also called
Termating nodes

Branches: Branches are connections between nodes 
 they are represented as arrows. Each branch represents
 a responce such as yes or no.
 
### Steps to make decision tree

1. Select the best attribute A
The data with the best information gain
best divides the data into the desired output classes

So we need to calculate the following measures:

1. Entropy
This measures  the impurity or uncertainity present in the data

2. Information Gain
This indicates how much 'information' a particular feature / variable gives about the final outcome

Example calculate the IG of parent node (Speed of a car)

Find out the fraction of the two classess (slow and fast) present in the parent node:

P(slow) -) fraction of 'slow' outcomes in the parent node

P(fast) -) fraction of 'fast' outcomes in the parent node

The formula to calculate P(slow) is 
= number of 'slow' outcomes in the parent node/ total number of outcomes

The formula to calculate P(fast) is 
= number of 'fast' outcomes in the parent node/ total number of outcomes


Calculate the Entropy
Entropy(parent) = -sigmaP(slow)log2(Pslow) + P(fast)log2(Pfast)

Calculating the IG of the child npde (Road type)

information Gain = entropy(parent) - [weighted average] * entropy(children)
 
2. Assign A as a decision variable for the root node
3. For each value of A. Build a descendant of the node
4. Assign classification labels to the leaf node
5. If data is correctly classified. Stop
6. Else iterate over the tree

 
 
## Random Forest

This is a collection of multiple decision trees ( calleda forest) and glues them together to get a more
accurate and stable prediction.

It is:
    More accurate
    Avoid Overfitting
    Bagging
    
Bootstraping is an estimation method used to make predictions on a data set
by re-sampling it.

## Naive Bayes

This is based on Bayes Theorem that is used to solve classification problems by
following a probabilistic approach

* P(A|B): Conditional probability of event A occuring given the event B
* P(A): Probability of event A occuring
* P(B): Probabiity of event B occuring
* P(B|A): Condttional probability of event B occuring given the event A


## KNN (K Nearest Neighbour)

This is a supervised learning algorithm that classifies a new data point into the target class depending on the features
of its neigbouring data points



    
