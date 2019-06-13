#import decision tree from scikit-learn
from sklearn import tree
#import support vector machine from scikit-learn
from sklearn import svm
#import stochastic gradient descent from scikit-learn
from sklearn.linear_model import SGDClassifier

#[height, weight, shoe size]
x = [[181,80,44], [177,70,43], [160,60,38], [154,54,37], [166,65,40],
[190,90,47], [175,64,39], [177,70,40], [159,55,37], [171,75,42], [181,85,43]]

y = ['male', 'female', 'female', 'female', 'male','male', 'male', 
'female', 'male', 'female', 'male']

#creating classifier and making it a decision tree classifier
clfTree = tree.DecisionTreeClassifier()
#creating classifier and making it a support vector classifier
clfSVC = svm.SVC(gamma='scale')
#creating classifier and making it a SGD classifier
clfSGD = SGDClassifier()

#train data
clfTree = clfTree.fit(x,y)
clfSVC = clfSVC.fit(x,y)
clfSGD = clfSGD.fit(x,y)

#set prediction to what the model predicts the entered data to be
#correct answer is 'female' and only Tree data predicts properly and consistently
prediction = clfTree.predict([[160,60,38]])
prediction1 = clfSVC.predict([[160,60,38]])
prediction2 = clfSGD.predict([[160,60,38]])

print(prediction)
print(prediction1)
print(prediction2)
