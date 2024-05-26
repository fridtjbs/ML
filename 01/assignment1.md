### ​Assignment № 1

### ​

### ​Problem area definition

This assignment focuses on a binary classification problem, wherein for each class the data samples consist of only two attributes (features). You will find the corresponding testing and training data sets in the folder with the assignment as .csv files. With labels at the top, the line format of the training file is _[AttrX,AttrY,Class]_, where each attribute is separated by commas, and each sample separated by a line break. There are in total 20 labeled training samples and 4 labeled test samples.

### ​Features and Classification Basics

In this section, we utilize the training data to decide which linear boundary best separates the data, and which attributes are more useful.

### Create and show a scatter plot of the training data (a1\_train.csv), where the different classes have different colors.

  1. Add a few possible linear decision boundaries of the classes based on the training data samples. Recall, we are working with a binary classification problem. How many potential boundaries are there?
    1. Ultimately, which boundary do you think will generalize best to unseen data?

1.3 Looking at the scatter plot and your chosen decision boundaries, which feature do you believe to be more useful for binary classification (that is, which attribute can discriminate classes more easily)? Why do you believe this feature is better? Using Weka, apply at least two attribute evaluators that can rank features (InfoGainAttributeEval, GainRatioAttributeEval, etc) of the training dataset to test your claim, and report their feature rankings in comparison to your results.

### ​

### ​

### ​

### ​

### ​Classification Evaluation and Classification Models

In this section, we evaluate our decision boundary and trained models using Weka on the testing data.

2.1 Using what you believe to be your best decision boundary from Part 1, predict which classes the test samples from a1\_test.csv belong to. Using these results, create a confusion matrix.

2.1.a What is the accuracy of your classification model (the results based on your linear decision boundary)?

2.2. Using Weka, try testing at least two different classification algorithms, and test how well the models perform given the training and testing data. What were the accuracies of the trained models applied on the test data?

Notes: Suggested classifiers NaïveBayes, LogisticRegression, IBk (K-Nearest Neighbors), LibSVM, MultilayerPerceptron, Decision Trees (J48). No need to update hyperparameters (done so by left clicking the classifier after selecting it).

2.3 Using Weka, use the J48 Decision Tree algorithm to create a decision tree classifier based on your training data (if you do not have it available, try downloading it through the Weka Package manager). Once it is trained, you can right click the model in the results list to visualize the generated tree. Draw or paste the decision tree below. What is the meaning of this decision tree?

2.4 What would be some of the benefits of increasing the number of samples in our training dataset, assuming the added data was not erroneous?

### ​Assignment Delivery Details (FAQ)

_ **Grading** __: Each assignment is graded and represents 10% of the final grade. Students have to upload their report with associated material until the deadline. Please, put all information regarding the assignment in a single archive package with name_ _ **assignN.zip** _ _(N being the respective assignment number: 1, 2, …)__ **.** _

_ **Delivery Format** __: The assignment delivery should include responses to the questions, source code, and any other necessary additional material/attachments. The written portion of the delivery may be a Word Document, PDF, or Jupyter Notebook with code included, etc. Code may be written in any language of the student's choosing, but Python, C++, or Matlab is preferred._

_ **Delivery Length:** _ _There is no mandatory length of the written assignment. So long as your response to the question supported with sufficient evidence or analysis then you have the potential to receive full credit. Past students who have received A's have submitted assignments anywhere between 2-10 pages._ _Different assignments are expected to be of different lengths._

_ **References** __: Within the assignment, cite references as necessary (both within the written assignment, and the coding portions). Include a list of references at the end of the assignment._

_ **Use of Libraries:** _ _Programming libraries are allowed unless specified. However, if we ask you to program a method (such as a simple classifier, or optimization method), we expect that you do so, where the library does not solve the task for you. Libraries may be used to assist this end (for example, NumPy to assist linear algebra, or libraries to split your data)._

_ **Plagiarism** __: The assignment delivery must your INDIVIDUAL WORK AND ANALYSIS. This includes the written part, and the programming portion. Use of any code found on the internet or elsewhere and used in the assignment should be explicitly referenced, and comments should be added to explain what the code is doing or why it is added to the assignment._

_If significant portions of code are found in the submission that are identical to code online, without a reference or comments from the student, we may give the particular assignment a 0. If plagiarism is common in the submissions or is deemed extensive, we may report it as plagiarism._

_If the student is unable to program a specific portion of the assignment, they may write pseudocode, with comments explaining the pseudocode (maximum credit per question with pseudocode is 50%)._

### ​

**References**

1. Lecture materials.

2. Machine Learning and Data Mining: Introduction to Principles and Algorithms (Paperback) by Igor Kononenko and Matjaz Kukar.

3. Pattern Recognition and Machine Learning (Information Science and Statistics) by C.M. Bishop.