import pandas as pd
df = pd.read_csv('/iris')
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm 
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB 
lr = LogisticRegression()
rf = RandomForestClassifier(random_state=1)
gbc = GradientBoostingClassifier(n_estimators=10)
dt = DecisionTreeClassifier(random_state=0)
sv=svm.SVC()
nn = MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(5, 2),random_state=0)
gnb = GaussianNB()  
mnb = MultinomialNB()
x = df.drop('variety',axis=1)
print(x)
y = df['variety']
print(y)
**LOGISTIC REGRESSION**
train = lr.fit(X_train, Y_train)
y_pred = lr.predict(X_test)
print(accuracy_score(Y_test, y_pred))
**RANDOM FOREST**
train_rf = rf.fit(X_train, Y_train)
rf_pred = rf.predict(X_test)
print(accuracy_score(Y_test, rf_pred))
**GRADIENT BOOSTING**
train_gbc =gbc.fit(X_train, Y_train)
gbc_pred = gbc.predict(X_test)
print(accuracy_score(Y_test, gbc_pred))
**NAIVE BAYES GAUSSIAN**
train_gnb = gnb.fit(X_train, Y_train)
gnb_pred = gnb.predict(X_test)
print(accuracy_score(Y_test, gnb_pred))
**NAIVE BAYES MULTINOMIAL**
train_mnb = mnb.fit(X_train, Y_train)
mnb_pred = mnb.predict(X_test)
print(accuracy_score(Y_test, mnb_pred))
**DECISION TREE**
train_dt = dt.fit(X_train, Y_train)
dt_pred = dt.predict(X_test)
print(accuracy_score(Y_test, dt_pred))
**NEURAL NETWORKS**
train_nn = nn.fit(X_train, Y_train)
nn_pred = nn.predict(X_test)
print(accuracy_score(Y_test, nn_pred))
**SUPPORT VECTOR**
train_sv = sv.fit(X_train, Y_train)
sv_pred = sv.predict(X_test)
print(accuracy_score(Y_test, sv_pred))
