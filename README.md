# FindIT

The digital world is increasingly accessible to children, with mobile apps playing a significant role in their entertainment, education, and social interaction. However, this increased access also brings potential risks, particularly regarding the collection and use of children's personal information. The Children's Online Privacy Protection Act (COPPA) in the United States, and similar regulations globally, aim to protect children's privacy online by requiring app developers to obtain parental consent before collecting data from users under 13.

This competition challenges you to develop a machine learning model that can predict whether a mobile app is likely to be at risk of violating COPPA. By identifying potentially non-compliant apps, we can help app stores, developers, and parents create a safer online environment for children. Your model will analyze a variety of app characteristics, including genre, target audience (implied by download ranges), privacy policy features, and developer information, to assess the likelihood of COPPA non-compliance.

# Problem Statement
The core objective is a binary classification task:
Target Variable: coppaRisk (boolean: true or false) - Predict whether an app is at risk of violating COPPA. true indicates a higher risk of non-compliance, while false suggests a lower risk.

# Evaluation
Submissions will be evaluated based on a suitable classification metric, likely:

AUC (Area Under the ROC Curve)
AUC, or Area Under the ROC Curve, is a single metric that summarizes the overall effectiveness of a classifier. Its value ranges from 0 to 1, where 0.5 suggests the model performs no better than random guessing, and a value of 1 indicates perfect classification performance.
