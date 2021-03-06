

File 1: Assignment.ipynb

This file contains the jupyter notebook, where the assignment is explained in a detailed manner.
   step 1: Preprocessing data
   step 2: Splitting data into 70:30 ratio
   step 3: Create bag of words models: 
           Algorithms used: MultinomialNB, BernoulliNB, LogisticRegression, RandomForest, GradientBoosting, XGBoost

           Best model to go forward: LogisticRegression with 88% F1-score in validation set 
                                                             85.536% in Kaggle Public Leadeboard.

   step 4: Using Tf-Idf Vectorizer and setting maximum features to be used at 10,000 features

           F1-score in validation set -> 89%
           Kaggle Public Leadeboard   -> 87.724%

   step 5: N-Gram Analysis with tf-idf vactorizer and 10,000 features
           N-grams used: Bi-gram(2,2), Tri-gram(3,3), Uni-bi-gram(1,2)

           Best model to go forward: Uni-bi-gram (1,2) with 89% F1-score in validation set
                                                            88.056% in Kaggle Public Leadeboard.


   step 6: Cross validation on the last model with the full training set

   step 7: Hyper-parameter tuning with cv=5
           Best model:  LogisticRegression(C=2.0, class_weight=None, dual=False, fit_intercept=True,
                                           intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                                           penalty='l2', random_state=None, solver='sag', tol=0.001,
                                           verbose=0, warm_start=False)

           Kaggle Public Leadeboard   -> 88.464%




File 2: Assignment.py

This file contains the code to this problem from start to end that has the same steps as file 1.
The best model (which gave a score of 88.464% in Kaggle Public Leadeboard) in the previous file is used here to predict on the test data.

Libraries that needs to be installed and imported
1. pandas
2. scikit-learn
3. re
4. string
5. nltk


How to fetch the data?
Just enter the location of the train and test file in the main_ml() function.



File 3: Kaggle_submissions.png

This is the screenshot of all the submissions done in Kaggle.