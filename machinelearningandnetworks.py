sb.pairplot(df,hue='Species',palette='hls')

DBSCAN for Outlier Detection
eps and min_samples - model parameters for dbscan
import sklearn
from sklearn.cluster import DBSCAN
from collections import Counter

model=DBSCAN(eps=0.8,min_samples=19).fit(data)
print model

outliers_df=pd.DataFrame(data)
print Counter(model.labels_)
print outliers_df(model.labels_==-1) #getting outliers
#less than 5% can be outliers

K means clusturing
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale
import sklearn.metrics as sm
from sklearn import datasets
from sklearn.metrics import confusion_matrix, classification_report

load dataset iris
clustering=KMeans(n_clusters=3,random_state=5)
clustering.fit(x)

you get summary of the model

plt.subplot(1,2,1)
plt.scatter(x=iris_df.Petal_Length,y=iris_df.Petal_Width,c=color_theme[iris.target],s=50)
plt.title('Ground Truth Classification')

plt.subplot(1,2,2)
plt.scatter(x=iris_df.Petal_Length,y=iris_df.Petal_Width,c=color_theme[clustering.labels_],s=50)
plt.title('K-Means Classification')

relabel=np.choose(clustering.labels_,[2,0,1]).astype(np.int64)
plt.subplot(1,2,1)
plt.scatter(x=iris_df.Petal_Length,y=iris_df.Petal_Width,c=color_theme[iris.target],s=50)
plt.title('Ground Truth Classification')

plt.subplot(1,2,2)
plt.scatter(x=iris_df.Petal_Length,y=iris_df.Petal_Width,c=color_theme[clustering.labels_],s=50)
plt.title('K-Means Classification')

print(classification_report(y,relabel))
to evaluate the results

Hierarchical Clustering
Hierarchical Clustering Dendrogram
from scipy.cluster.hierarchy import dendrogram,linkage
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
import sklearn.metrics as sm

np.set_printoptions(precision=4,suppress=True)

Z=linkage(x,'ward') #clustering
dendrogram(Z,truncate_mode='lastp',p=12,leaf_rotation=45.,leaf_font=15.,show_contracted=True)
plt.axhline(y=500)
plt.axhline(y=150)
plt.show()
k=2

Hclustering=ApplomerativeClustering(n_clusters=k,affinity='euclidean',linkage='ward')#linkage can be ward,complete,average and affinity can be euclidean,manhattan and take the best accuracy
Hclustering.fit(x)

sm.accuracy_score(y,Hclustering.labels_)

K-NN K nearest neighbor
avoid large datasets
import urlib
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from skleaen import metrics
x=preprocessing.scale(x_prime)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.33,random_state=17)
clf=meighbors.KNeighborsClassifier()
clf.fit(X-train,y_train)
print(clf)

y_expect=y_test
Y_pred = clf.predict(X_test)
print(metrics.classification_report(y_expect,y_pred))

NETWORK ANALYSIS
import numpy pandas seaborn matplotlib
import networkx as nx

G=nx.Graph()
nx.draw(G)

G.add_node(1)
nx.draw(G)

G.add_nodes_from([2,3,4,5,6,8,12,..])
G.add_edges_from([(2,4),(2,6),(2,8)...])
nx.draw(G)

nx.draw_circular(G, node_color='bisque',with_labels=True)
G.remove_node(1)
nx.draw_spring(G)

sum_stats=nx.info(G)
print sum_stats
print nx.degree(G)
G = nx.complete_graph(25)
nx.draw(G,node_color=''..)
G = nx.gnc_graph(7,seed=25)
ego_G = nx.ego_graph(G,3,radius=5)
nx.draw(G)

SIMULATE A SOCIAL NETWORK
DG = nx.gn_graph(7,seed=25)
for line in nx.generate_edgelist(DG, data=False): print(line)
print DG.node(0)
DG.node[0]['name']='Alice'
print DG.node(0)
DG.node[1]['name']='somthing'
...
DG.add_nodes_from([(0,{'age':25}),(1,{'age':18})........])#adding ages for all nodes
print DG.node(0)
DG.node[0]['gender']='m'
...
nx.draw_circular(DG, with_labels=True)
labeldict = {0:'Alice',1:'Bob',...}
nx.draw(DG,labels=labeldict,with_labels=True)
G = DG.to_undirected()
nx.draw_spectrum(G,labels=labeldict,with_labels=True)

print nx.info(DG)
DG.degree()
nx.draw_circular(DG,node_color='bisque',with_labels=True)
DG.successors(3)
DG.neighbors(4)
G.neighbors(4)

Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
from collections import Counter
#enroll forecast dataset
enroll.pd.read_csv(address)
enroll.colums = ['year','roll'..]
enroll.head()
sb.pairplot(enroll)
print enrool.corr()#correlation
enroll_data=enroll.ix[:,(2,3)].values
enroll_target=enroll.ix[:,1].values
enroll_data_names=['unem','hgrad']
x,y=scale(enroll_data),enroll_target
missing_values=x==np.NAN
x[missing_values==True]
LinReg=LinearRegression(normalize=True)
LinReg.fit(x,y)
print LinReg.score(x,y)

Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn import preprocessing

#using cars dataset
cars.head()
cars_data=cars.ix[:,(5,11)].values
cars_data_names=['drat','carb']
y=cars.ix[:,9].values
sb.regplot(x='drat',y='carb',data=cars,scatter=True)
drat=cars['drat']
carb=cars['carb']
spearmanr_coefficient,p_value=spearmanr(drat,carb)
print 'Spearman Rank Correlation Coefficient %0.3f'%(spearmanr_coefficient)
cars.isnull().sum()
sb.countplot(x='am',data=cars,palette='hls')
cars.info()
X=scale(cars_data)
LogReg=LogisticRegression()
LogReg.fit(X,y)
print LogReg.score(X,y)
y_pred=LogReg.predict(x)
from sklearn.metrics import classification_report
print (classificarion_report(y,y_pred))

NAIVE BAYES CLASSIFIERS
import urllib
import sklearn
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score

url = "database url here"
raw_data = urllib.urlopen(url)
dataset=np.loadtxt(raw_data,delimiter=',')
print dataset(0)
x=dataser[:,0:48]
y=dataser[:,-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.33,random_state=17)
BernNM=BernoulliNB(binarize=True)
BernNM.fit(x_train,y_train)
print(BernNB)
y_expect=y_test
y_pred=BernNB.pred(x_test)
print accuracy_score(y_expect,y_pred)

MultiNB=MulyinomialNB()
MultiNB.fit(x_train,y_train)
print(MultiNB)
y_pred=MultiNB.predict(x_test)
print accuracy_score(y_expect,y_pred)

same way for GausNB use GaussianNB()
and print accuracy results
make binarize=0.1 from above it get better results


