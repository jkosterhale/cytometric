# # K MEANS


def k_means_optimized(testdata, nrange=range(2,12), scale=False):
    '''Returns trained k-means model that optimizes silhouette score
    
    Args:
        data (ndarray): data to cluster
        n_clusters_range (iterable of ints): values of n_clusters (k) to try
        
    Returns:
        fitted sklearn.cluster.KMeans
    '''
    if scale:
        scaler = StandardScaler()
        testdata = scaler.fit_transform(testdata)
    scores = {} # scores mapped to n_clusters (float --> int)
    for n_clusters in nrange:
        model = KMeans(n_clusters=n_clusters)
        model.fit(testdata)
        score = silhouette_score(testdata, model.labels_, sample_size=2000+int(testdata.shape[0]**.5))
        #score = silhouette_score(testdata, model.labels_)
        del model
        scores[score] = n_clusters
    best_score = max(scores.keys())
    best_n_clusters = scores[best_score]
    best_model = KMeans(n_clusters=best_n_clusters)
    best_model.fit(testdata)
    return best_model, scores
    

#check out just the target cells -- how we well do we do? 
#d[(d['x']>2) & (d['y']>7)]
cols = DADcols
temp = subset[(subset['cell_type']=="blast") | (subset['cell_type']=="healthy")]
kmeans_temp, scores_temp = k_means_optimized(temp[cols].as_matrix(),nrange = (2,10), scale=False)    

temp['kmeans_temp'] = kmeans_temp.labels_
print kmeans_temp
print scores_temp
plt.bar(range(len(scores_temp)), scores_temp.keys(), align='center')
plt.xticks(range(len(scores_temp)), scores_temp.values())

plt.show()
print 'KMEANS TESTING NMI:', adjusted_mutual_info_score(temp['cell_type'], kmeans_temp.labels_)
table = temp.groupby(['cell_type',"kmeans_temp"]).count()

plt.plot(kmeans_temp.cluster_centers_[0],label="0")
plt.plot(kmeans_temp.cluster_centers_[1],label="1")
#plt.plot(kmeans_temp.cluster_centers_[2],label="2")
plt.legend()
plt.xticks(range(0,len(cols)), cols, rotation='vertical')
table['FSC-A']




#cluster DAD
kmeans_DAD, scores_DAD = k_means_optimized(subset[DADcolsscaled].as_matrix())    
    
#how did we do at DAD?
subset['kmeans_DAD'] = kmeans_DAD.labels_
print kmeans_DAD
print scores_DAD
plt.bar(range(len(scores_DAD)), scores_DAD.keys(), align='center')
plt.xticks(range(len(scores_DAD)), scores_DAD.values())

plt.show()
print 'KMEANS DAD NMI:', adjusted_mutual_info_score(subset['cell_type'], kmeans_DAD.labels_)
table = subset.groupby(['cell_type',"kmeans_DAD"]).count()
table['FSC-A']

#actual labels 
f, ax = plt.subplots(figsize=(15, 15))
x = subset.groupby(['cell_type']).mean()
y = x[DADcolsscaled]
a = y.iloc[0,:]
b = y.iloc[1,:]
c = y.iloc[2,:]
a.plot(label="blast",rot=0)
b.plot(label="healthy")
c.plot(label="nontarget",rot=90)
plt.xlabel = DADcolsscaled
plt.legend()


# cluster labels 
plt.plot(kmeans_DAD.cluster_centers_[0],label="0")
plt.plot(kmeans_DAD.cluster_centers_[1],label="1")
plt.plot(kmeans_DAD.cluster_centers_[2],label="2")
#plt.plot(kmeans_DAD.cluster_centers_[3],label="3")
plt.xticks(range(0,len(DADcolsscaled)), DADcolsscaled, rotation='vertical')
plt.legend()
DADcolsscaled

#now get just the remaining  get only good cells -- cluster TYPE 
x = subset.groupby(['cell_type',"kmeans_DAD"]).count()
subset_TYPE = subset[subset['kmeans_DAD']==0]

# In[323]:

#now cluster just the remaining 
kmeans_TYPE, scores_TYPE = k_means_optimized(subset_TYPE[cancercols].as_matrix(),n_clusters_range=range(3,12))


# In[324]:

#how did it do? 
subset_TYPE['kmeans_TYPE'] = kmeans_TYPE.labels_
print kmeans_TYPE
print scores_TYPE
plt.bar(range(len(scores_TYPE)), scores_TYPE.keys(), align='center')
plt.xticks(range(len(scores_TYPE)), scores_TYPE.values())

print 'KMEANS TYPE NMI:', adjusted_mutual_info_score(subset_TYPE['cell_type'], kmeans_TYPE.labels_)
plt.show()
subset_TYPE.groupby(['cell_type',"kmeans_TYPE"]).count()



# In[308]:

kmeans_TYPE.cluster_centers_[0]


# In[325]:

plt.plot(kmeans_TYPE.cluster_centers_[0],label="0")
plt.plot(kmeans_TYPE.cluster_centers_[1],label="1")
plt.plot(kmeans_TYPE.cluster_centers_[2],label="2")
#plt.plot(kmeans_TYPE.cluster_centers_[3],label="3")
#plt.plot(kmeans_TYPE.cluster_centers_[4],label="4")
plt.xticks(range(0,len(bcscaledcols)), bcscaledcols, rotation='vertical')
plt.legend()
colsOfInterest


# In[328]:

#now get just the remaining  get only good cells -- cluster TYPE 
#x = subset.groupby(['cell_type',"kmeans_TYPE"]).count()
subset_FINAL = subset_TYPE[subset_TYPE['kmeans_TYPE']==2]


# In[329]:

#now cluster just the remaining 
kmeans_FINAL, scores_FINAL = k_means_optimized(subset_FINAL[bcscaledcols].as_matrix())

#how did it do? 
subset_FINAL['kmeans_FINAL'] = kmeans_FINAL.labels_
print kmeans_FINAL
print scores_FINAL
plt.bar(range(len(scores_FINAL)), scores_FINAL.keys(), align='center')
plt.xticks(range(len(scores_FINAL)), scores_FINAL.values())

print 'KMEANS TYPE NMI:', adjusted_mutual_info_score(subset_FINAL['cell_type'], kmeans_FINAL.labels_)
plt.show()
subset_FINAL.groupby(['cell_type',"kmeans_FINAL"]).count()




# In[330]:

plt.plot(kmeans_FINAL.cluster_centers_[0],label="0")
plt.plot(kmeans_FINAL.cluster_centers_[1],label="1")
plt.plot(kmeans_FINAL.cluster_centers_[2],label="2")
#plt.plot(kmeans_TYPE.cluster_centers_[3],label="3")
#plt.plot(kmeans_TYPE.cluster_centers_[4],label="4")
plt.xticks(range(0,len(bcscaledcols)), bcscaledcols, rotation='vertical')
plt.legend()
colsOfInterest


# In[332]:

#now cluster just the remaining 
#now get just the remaining  get only good cells -- cluster TYPE 
x = subset.groupby(['cell_type',"kmeans_DAD"]).count()
subset_TYPE = subset[subset['kmeans_DAD']==2]

kmeans_TYPE, scores_TYPE = k_means_optimized(subset_TYPE[cancercols].as_matrix())

#how did it do? 
subset_TYPE['kmeans_TYPE'] = kmeans_TYPE.labels_
print kmeans_TYPE
print scores_TYPE
plt.bar(range(len(scores_TYPE)), scores_TYPE.keys(), align='center')
plt.xticks(range(len(scores_TYPE)), scores_TYPE.values())

print 'KMEANS TYPE NMI:', adjusted_mutual_info_score(subset_TYPE['cell_type'], kmeans_TYPE.labels_)
plt.show()



plt.plot(kmeans_TYPE.cluster_centers_[0],label="0")
plt.plot(kmeans_TYPE.cluster_centers_[1],label="1")
#plt.plot(kmeans_TYPE.cluster_centers_[2],label="2")
#plt.plot(kmeans_TYPE.cluster_centers_[3],label="3")
#plt.plot(kmeans_TYPE.cluster_centers_[4],label="4")
plt.xticks(range(0,len(cancercols)), cancercols, rotation='vertical')
plt.legend()
subset_TYPE.groupby(['cell_type',"kmeans_TYPE"]).count()
#subset_TYPE.groupby(['cell_type',"kmeans_TYPE"]).mean()


# In[ ]:

z = 2
index = np.array(range(0,len(colsOfInterestFlow)))
plt.barh(index,kmeans_TYPE.cluster_centers_[z],label="z", color=cmap[z])
plt.yticks(index+ .3, colsOfInterest)
#plt.plot(kmeans_TYPE.cluster_centers_[1],label="1")
#plt.plot(kmeans_TYPE.cluster_centers_[2],label="2")
#plt.plot(kmeans_TYPE.cluster_centers_[3],label="3")
#plt.plot(kmeans_TYPE.cluster_centers_[4],label="4")
#plt.yticks(range(0,len(colsOfInterestFlow))+.25, colsOfInterestFlow, rotation='horizontal')
#plt.legend()
#subset_TYPE.groupby(['cell_type',"kmeans_TYPE"]).count()
#subset_TYPE.groupby(['cell_type',"kmeans_TYPE"]).mean()


# In[ ]:

index[:] + .25


# In[ ]:

kmeans_TYPE.cluster_centers_[0]


# In[ ]:

sns.set(font_scale=1.6)
subset.loc[subset['cell_type'] =="blast" ,'cell_type_num'] = 1
subset.loc[subset['cell_type'] =="healthy" ,'cell_type_num'] = 2
subset.loc[subset['cell_type'] =="nontarget" ,'cell_type_num'] = 3

g = sns.lmplot('FSC-H_scaled', 'SSC-H_scaled', data=subset, hue="kmeans_DAD", legend=True, fit_reg=False,scatter_kws={'alpha':0.4},size=8)
g.set(ylim=(-1, 5))
g.set(xlim=(-1, 5))
#from scipy.cluster.hierarchy import dendrogram, linkage
#X = subset[colsOfInterestSubFlow].as_matrix()
#plt.scatter(X[:,0], X[:,1],c=subset['cell_type_num'], cmap=["blue","red","green"])
#plt.show()
#colsOfInterestSubFlow



# In[ ]:




# In[ ]:

colsOfInterest


# In[ ]:

sns.set(font_scale=1.6)

cmap = sns.cubehelix_palette(3, start=2, rot=0, dark=0, light=.95, reverse=True)  
cmap = sns.dark_palette("lightgreen",4, reverse=True)
#cmap = [  0.34986544,  0.53490196,  0.34986544,  1.      ],[   0.34986544,  0.53490196,  0.34986544,  1.       ],[  0.34986544,  0.53490196,  0.34986544,  1.     ]
#
g = sns.lmplot('CD66B H : CD19 H : CD3 H : FITC H_log', 'CD34 H : BV605 H_log', data=subset_TYPE, legend=False,palette = cmap, hue="kmeans_TYPE", fit_reg=False,scatter_kws={'alpha':0.8},size=8)
g.set(ylim=(7, 14))
g.set(xlim=(7, 14))

g = sns.lmplot('FSC-H_scaled', 'SSC-H_scaled', data=subset_TYPE, legend=True,palette = cmap, hue="kmeans_TYPE", fit_reg=False,scatter_kws={'alpha':0.4},size=8)
g.set(ylim=(-1, 5))
g.set(xlim=(-1, 5))

#from scipy.cluster.hierarchy import dendrogram, linkage
#X = subset[colsOfInterestSubFlow].as_matrix()
#plt.scatter(X[:,0], X[:,1],c=subset['cell_type_num'], cmap=["blue","red","green"])
#plt.show()
#colsOfInterestSubFlow


# In[ ]:

cmap[0]


# -------------




# In[779]:

cancercols


# In[ ]:




# In[782]:

plotcols= np.append(cancercols, 'kmeans_'+str(CL-1))
sns.pairplot(subdata[plotcols], hue = 'kmeans_'+str(CL-1));


# In[761]:

cancercols


# In[ ]:




# In[ ]:




# 
# 
# # Hierarchical Clustering

# In[ ]:




# In[ ]:

plt.show()


# In[ ]:

X = subset_TYPE[colsOfInterestSubFlow].as_matrix()
plt.scatter(X[:,0], X[:,1])
plt.show()


# In[ ]:

Z = linkage(X, 'ward')
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

c, coph_dists = cophenet(Z, pdist(X))
c


# In[ ]:

Z


# In[ ]:

subset_TYPE.loc[(subset_TYPE['cell_type']=='healthy'),"color"] = "blue"
subset_TYPE.loc[(subset_TYPE['cell_type']=='blast'),"color"] = "red"
subset_TYPE.loc[(subset_TYPE['cell_type']=='nontarget'),"color"] = "green"


# In[ ]:

xlabels = subset_TYPE["cell_type"].tolist()


# In[ ]:

f, ax = plt.subplots(figsize=(40, 40))

plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
d = dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels,
    get_leaves = True,
    color_threshold=max(Z[:,2]),
    count_sort = True,
    orientation = "left",
    labels=xlabels
    #link_color_func = xcolor
)

x = subset.iloc[d["leaves"],subset.columns.get_loc("color")] 
xcolors = list(x.values)

for xtick, color in zip(ax.get_xticklabels(), xcolors):
    xtick.set_color(color)

for ytick, color in zip(ax.get_yticklabels(), xcolors):
    ytick.set_color(color)
    
    


# In[ ]:


y =  ax.get_xticklabels()
y[[1]]


# In[ ]:

from scipy.cluster.hierarchy import fcluster
max_d = 4
clusters = fcluster(Z, max_d, criterion='distance')
clusters


# In[ ]:

k=3
clusters = fcluster(Z, k, criterion='maxclust')


# In[ ]:

colsOfInterest[13]


# In[ ]:


plt.figure(figsize=(10, 8))
plt.scatter(X[:,12], X[:,3], c=clusters, cmap='prism')  # plot points with cluster dependent colors
plt.show()


# In[ ]:

subset.loc[subset['cell_type'] =="blast" ,'cell_type_num'] = 1
subset.loc[subset['cell_type'] =="healthy" ,'cell_type_num'] = 2
subset.loc[subset['cell_type'] =="nontarget" ,'cell_type_num'] = 3


# In[ ]:


plt.figure(figsize=(10, 8))
plt.scatter(X[:,12], X[:,3], c=subset['cell_type_num'], cmap='prism')  # plot points with cluster dependent colors
plt.show()


# # DBSCAN

# In[ ]:

from sklearn.cluster import DBSCAN


# In[ ]:

# check k distance for dbscan
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=int(len(subset)**.5))
knn.fit(subset[colsOfInterest], [1]*len(subset))
distances = np.array(knn.kneighbors_graph(n_neighbors=int(len(subset)**.25), mode='distance').max(axis=1).todense().T)[0]
distances.sort()
plt.plot(distances)
#plt.ylim((0, 1e6))


# In[ ]:




# In[ ]:

# fit dbascn
dbscan = DBSCAN(eps=2e5, min_samples=1000)
#scaler = RobustScaler()
#scaled_data = scaler.fit_transform(testdata[relevant_columns])
dbscan.fit(subset[colsOfInterest])
subset['dbscan'] = dbscan.labels_
# look at class balances
from collections import Counter
counter = Counter(dbscan.labels_)
print counter
# evaluate
dbscan_plots = plotting.pairwise_plots(subset, colsOfInterest, 'dbscan', max_points=1000, opacity=.75)
print 'DBSCAN NMI:', normalized_mutual_info_score(subset['cell_type'], dbscan.labels_)
for p in dbscan_plots: iplot(p)


# # HIERARCHICAL K MEANS

# In[ ]:

# # use original k means object from above
# data['kmeans2'] = None
# for cluster_label in data['kmeans'].unique():
#     model = k_means_optimized(data[data['kmeans']==cluster_label][relevant_columns].as_matrix())
#     data.loc[data['kmeans']==cluster_label,'kmeans2'] = model.labels_.astype(str) + \
#                                                         data.loc[data['kmeans']==cluster_label,'kmeans'].astype(str)


# In[ ]:

# kmeans2_plots = plotting.pairwise_plots(data, relevant_columns, 'kmeans2', max_points=1000, opacity=.75)


# In[ ]:

#for p in kmeans2_plots: iplot(p)


# # NMF

# In[ ]:

from sklearn.decomposition import NMF
nmf = NMF(n_components=3)
nmf_transformed_data = nmf.fit_transform(data[relevant_columns].as_matrix())
data['nmf'] = np.argmax(nmf_transformed_data, axis=1)
print 'NMF NMI:', normalized_mutual_info_score(data['cell_type'], data['nmf'])
#nmf_plots = plotting.pairwise_plots(data, relevant_columns, 'nmf', max_points=1000, opacity=.75)
#for p in nmf_plots: iplot(p)


# # K Means + PCA

# In[ ]:

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# In[ ]:

# scale
scaler = RobustScaler()
data_scaled = scaler.fit_transform(data[relevant_columns])

# pca
pca = PCA(n_components=3)
data_pca_transformed = pca.fit_transform(data_scaled)


# In[ ]:

# kmeans_with_pca = KMeans(n_clusters=5)
# kmeans_with_pca.fit(data_pca_transformed)
kmeans_with_pca = k_means_optimized(data_pca_transformed, scale=False)
data['kmeans_with_pca'] = kmeans_with_pca.labels_
print 'KMEANS+PCA NMI:', normalized_mutual_info_score(data['cell_type'], data['kmeans_with_pca'])
#kmeans_with_pca_plots = plotting.pairwise_plots(data, relevant_columns, 'kmeans_with_pca', max_points=1000, opacity=.75)
#for p in kmeans_with_pca_plots: iplot(p)


# # DBSCAN WITH PCA

# In[ ]:

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# scale
scaler = StandardScaler() #RobustScaler()
data_scaled = scaler.fit_transform(data[relevant_columns])

# pca
pca = PCA(n_components=3)
data_pca_transformed = pca.fit_transform(data_scaled)


# In[ ]:

# check k distance for dbscan
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=int(len(data)**.5))
knn.fit(data_pca_transformed, [1]*len(data))
distances = np.array(knn.kneighbors_graph(n_neighbors=int(len(data_pca_transformed)**.1), mode='distance').max(axis=1).todense().T)[0]
distances.sort()
py.plot(distances)
py.ylim((0, 1))


# In[ ]:

dbscanwithpca = DBSCAN(eps=.1, min_samples=len(data)**.5)
dbscanwithpca.fit(data_pca_transformed)
data['dbscan_with_pca'] = dbscanwithpca.labels_
from collections import Counter
counter = Counter(dbscanwithpca.labels_)
print counter


# In[ ]:

print 'DBSCAN+PCA NMI:', normalized_mutual_info_score(data['cell_type'], data['dbscan_with_pca'])
#dbscanwithpca_plots = plotting.pairwise_plots(data, relevant_columns, 'dbscan_with_pca', max_points=10000, opacity=.25)
#for p in dbscanwithpca_plots: iplot(p)


# In[ ]:

print("\n" * 100)


# In[ ]:


## plot it with one color 
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


x = subset[DADcolsscaled[0]]
y = subset[DADcolsscaled[1]]

# Calculate the point density
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

# Sort the points by density, so that the densest points are plotted last
#idx = z.argsort()
#x, y, z = x[idx], y[idx], z[idx]

fig, ax = plt.subplots()
ax.scatter(x, y, c=z, cmap="Blues_r", s=50, edgecolor='')

ax.set_xlim([-1, 3])
ax.set_ylim([-1, 3])
plt.show()





# In[ ]:








--------------------------------------------------------------------------------------------------------------------------------------------------------
# Plot what are already have! 

import plotly as py
plotly.tools.set_credentials_file(username='jokoho', api_key='kveygo902q')

import plotly.plotly as py
import plotly.tools as plotly_tools
from plotly.graph_objs import *

py.sign_in(username='jokoho', api_key='kveygo902q')



#cluster DAD
CL=0
print("this cluster: top level")

fig, ax = plt.subplots(figsize=(10, 10))
teals = sns.light_palette("#18cfed", as_cmap=True, reverse=True)
brightgreens = sns.light_palette("#03f90c", as_cmap=True, reverse=True)

cmaps = ("Blues_r","Reds_r","Greens_r","Purples_r","Reds_r","Greys_r","Blues_r","Reds_r","Greens_r","Oranges_r","Purples_r","Greys_r")


sns.cubehelix_palette(as_cmap=True,reverse=True,start=.1, rot=0, dark=.3)


fig = plt.figure()
ax = fig.add_subplot(221)
#fig, ax = plt.subplots(figsize=(10, 10))
s = 121
trace = list()
for n in range(0, nclusters):
    
    x = subset.loc[subset['kmeans_0']==n, DADcolsscaled[0]]
    y = subset.loc[subset['kmeans_0']==n, DADcolsscaled[1]]

    # Calculate the point density
    xy = np.vstack([x,y])
    z1 = gaussian_kde(xy)(xy)
    
    cb1 = ax.scatter(x, y, c=z1, cmap=cmaps[n], s=50, edgecolor=None,edgecolors=None,alpha=1)
    plt.colorbar(cb1, ax=ax)


ax.set_xlim([-1, 2])
ax.set_ylim([-1, 2])
ax.set_xlabel(DADcolsscaled[0])
ax.set_ylabel(DADcolsscaled[1])
ax.set_axis_bgcolor('white')
ax.grid(b=True, which='both', color="gray",linestyle='-',alpha=.1)
plt.tight_layout(pad=0.8, w_pad=0.8, h_pad=1.0)
fig = plt.gcf()
py.plot_mpl(fig, filename="mpl-colormaps-simple")
plt.show()

layout = dict(title = 'Styled Scatter',
              yaxis = dict(zeroline = False),
              xaxis = dict(zeroline = False)
             )

fig = dict(data=data, layout=layout)
py.iplot(fig, filename='styled-scatter', colorscale = cmaps)
