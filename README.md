This is presents a comprehensive evaluation of machine learning approaches for 
multiclass classification of human activities using smartphone accelerometer and gyroscope 
sensor data. Using the UCI Human Activity Recognition (HAR) dataset comprising 10,299 
samples with 561 engineered time and frequency domain features across six activity 
classes, we implement and systematically compare three classification algorithms: Logistic 
Regression, Random Forest, and Support Vector Machine (SVM) with RBF kernel. 
Key methodological contributions include: (1) standardised preprocessing using scikit
learn Pipelines to prevent data leakage, (2) comprehensive exploratory data analysis 
including PCA and t-SNE visualisations revealing class separability patterns, (3) 
hyperparameter tuning via GridSearchCV and RandomizedSearchCV with 5-fold stratified 
cross-validation, and (4) detailed analysis of why tuning produces modest improvements on 
expertly-engineered features. 
Results demonstrate that all three models achieve high classification accuracy: Logistic 
Regression (95.52%), Random Forest (92.81%), and SVM (95.18%). After hyperparameter 
tuning, Logistic Regression achieves the best performance at 95.52% accuracy (C=10, gamma='scale'), 
representing a +0.24% improvement. The modest gains from tuning are attributed to the 
high-quality feature engineering in the UCI HAR dataset, near-optimal default parameters in 
scikit-learn, and near-ceiling baseline performance. The primary driver of classification 
success is the expertly-crafted 561-dimensional feature space rather than hyperparameter 
optimisation.
