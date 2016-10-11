
# coding: utf-8

# In[54]:

get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')


# In[55]:


#DAD or TYPE 
get_ipython().magic(u'matplotlib inline')
classification = "TYPE"

import pandas as pd
import numpy as np

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_mutual_info_score

if classification == "DAD":
    get_ipython().magic(u"cd '/Users/jorie/Dropbox (Personal)/Insight_Personal/Analyses/ActiveCode/DAD'")
    import settings
elif classification == "TYPE":
    get_ipython().magic(u"cd '/Users/jorie/Dropbox (Personal)/Insight_Personal/Analyses/ActiveCode/TYPE'")
    import settings
    
    
get_ipython().magic(u"cd '/Users/jorie/Dropbox (Personal)/Insight_Personal/Analyses/ActiveCode'")
import processing
import plotting
import fcsparser as fcs


from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode()
import matplotlib.pylab as plt


# In[56]:

if classification == "DAD":
    relevant_columns = ['FSC-H', 'SSC-H', 'DAPI H', 'FSC-A', 'SSC-A', 'DAPI A']
elif classification == "TYPE":
    type_relevant_columns = ['FSC-H', 'SSC-H']


# In[57]:

settings.DATA_LOCATION


# In[58]:

#get the data with the column names labeled by compound 
if classification == "DAD":
    compound_data_uncat = processing.load_and_process_data(template = "*originallabeled.fcs", features_to_scale=None)
elif classification == "TYPE":
    compound_data_uncat = processing.load_and_process_data(template = "training_set/*.fcs", features_to_scale=None)
    
compound_data = pd.concat(compound_data_uncat, join='outer', ignore_index=True)
compound_data.index = compound_data[settings.EVENT_IDENTIFYING_COLUMNS]
compound_data


# In[59]:

#get the data with the column names labeled by compound 
if classification == "DAD":
    labeled_data_uncat = processing.load_and_process_data(template = "*.fcs", features_to_scale=None)
elif classification == "TYPE":
    labeled_data_uncat = processing.load_and_process_data(template = "screen_525_cell_plate_1_labeled/*.fcs", features_to_scale=None)
    
labeled_data = pd.concat(labeled_data_uncat, join='outer', ignore_index=True)
labeled_data.index = labeled_data[settings.EVENT_IDENTIFYING_COLUMNS]

labeled_data


# In[60]:

#check out your data
x = compound_data.count(0)
x.sort_values()


# In[61]:

#check out more of your data
x = labeled_data.count(0)
x.sort_values()


# In[62]:

#merge the labeled and unlabeled data (effectively, adding labels to complete dataset)
try:
    labeled_comp_data = processing.add_labeled_columns(labeled_data, compound_data)
    labeled_comp_data
except:
    labeled_comp_data = processing.add_labeled_columns(labeled_data, compound_data)
    labeled_comp_data
    
labeled_comp_data


# In[63]:

#clean up the NaNs in the unlabled data 
labeled_comp_data.loc[labeled_comp_data['cell_type_y'].isnull(),'cell_type_y'] = "nontarget"
labeled_comp_data['cell_type'] = labeled_comp_data['cell_type_y']



# In[64]:

#make the cell type label numeric for further modeling 
labeled_comp_data.loc[pd.isnull(labeled_comp_data['is_blast']) ,'is_blast'] = False
labeled_comp_data.loc[pd.isnull(labeled_comp_data['is_healthy']) ,'is_healthy'] = False
labeled_comp_data.loc[pd.isnull(labeled_comp_data['is_live']) ,'is_live'] = False
labeled_comp_data.loc[pd.isnull(labeled_comp_data['is_debris']) ,'is_debris'] = False
labeled_comp_data.loc[pd.isnull(labeled_comp_data['is_dead']) ,'is_dead'] = False
                      
labeled_comp_data['is_live']= pd.to_numeric(labeled_comp_data['is_live']*1)
labeled_comp_data['is_blast']= pd.to_numeric(labeled_comp_data['is_blast']*1)
labeled_comp_data['is_healthy']= pd.to_numeric(labeled_comp_data['is_healthy']*1) 
labeled_comp_data['is_dead']= pd.to_numeric(labeled_comp_data['is_dead']*1)
labeled_comp_data['is_debris']= pd.to_numeric(labeled_comp_data['is_debris']*1)



# In[65]:

#grab the columns of interest, and log and scale them 
x = labeled_comp_data.filter(regex='H')
x.drop('FSC-H',1)
x.drop('SSC-H',1)
scalenames = list(x.columns.values)

processing.scale_data(labeled_comp_data,scalenames)
processing.log_data(labeled_comp_data,scalenames)
list(labeled_comp_data.columns.values)


# In[66]:

#replace NAs with median 
labeled_comp_data_backup = labeled_comp_data
names = labeled_comp_data._get_numeric_data().columns.values
#labeled_comp_data[names].fillna(labeled_comp_data[names].mean())
#print(np.where(pd.isnull(labeled_comp_data[names])))
#print labeled_comp_data[names].iloc[3,55]
#x = labeled_comp_data[names].median()



# In[67]:

x = labeled_comp_data[names].fillna(labeled_comp_data[names].median())
labeled_comp_data[names] = x 
labeled_comp_data[names].isnull().sum()


# In[68]:

#check out that you have reasonable data: how many cell types
labeled_comp_data.groupby('cell_type').count()


# In[69]:

#check out that you have reasonable data: how many screens
labeled_comp_data.groupby('screen_number').count()


# In[70]:

#check out that you have reasonable data: how many wells
labeled_comp_data.groupby('well_number').count()


# In[427]:

#grab your subset of data
screenTarget = "525"
wellTarget = "c16" 

subset = labeled_comp_data.loc[(labeled_comp_data.screen_number==screenTarget)&(labeled_comp_data.well_number==wellTarget)]
subset


# In[428]:

#make sure there is something to analyze
subset.groupby('cell_type').count()


# # Explore me!
# 

# In[429]:

import seaborn as sns
sns.set(context="paper", font="monospace")


# In[430]:

logcols = labeled_comp_data.filter(regex='log|is').columns.values
logcols[logcols =='FSC-H_log'] = 'FSC-H'
logcols[logcols =='SSC-H_log'] = 'SSC-H'
logcols


# In[431]:

#re order for plotting 
cols = labeled_comp_data.columns.tolist()

b = [labeled_comp_data.columns.get_loc("is_blast"),labeled_comp_data.columns.get_loc("is_healthy"), 
     labeled_comp_data.columns.get_loc("is_live"),labeled_comp_data.columns.get_loc("is_dead"),
     labeled_comp_data.columns.get_loc("is_debris")]
b.sort()
a = [ cols[i] for i in b]

cols[b[0]:b[len(b)-1]+1] = []
cols[0:0] = a
cols
ld = labeled_comp_data[cols]




# In[432]:

# Load the datset of correlations between cortical brain networks
corrmat = ld[logcols].corr()

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(40, 40))


mask = np.zeros_like(corrmat, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=.8, square=True, mask=mask)


# In[433]:

ld_subset = ld.loc[(ld.screen_number==screenTarget)&(ld.well_number==wellTarget)]

# Load the datset of correlations between cortical brain networks
corrmat = ld_subset[logcols].corr()

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(20, 20))


mask = np.zeros_like(corrmat, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=.8, square=True, mask=mask)


# In[434]:

x=corrmat[["is_blast"]]
#x.sort_index(by=['is_blast'], ascending=[False])
x = x.sort_values('is_blast')
x


# In[435]:


x[x<1].plot(figsize=(20, 20),kind="bar")


# In[436]:

colsOfInterest = ["CD34 H : BV605 H_log","VL5-H_log", "FSC-H_scaled",] #"KIT H : BV421 H_log"]
colsOfInterest2 = colsOfInterest
plotme = ld_subset[colsOfInterest]


# In[81]:

#processing.scale_data(plotme,colsOfInterest,overwrite=True)


# In[438]:


f, ax = plt.subplots(figsize=(10, 10))

sns.distplot(plotme[[0]], label = plotme.columns.values[0])
sns.distplot(plotme[[1]],label = plotme.columns.values[1])
sns.distplot(plotme[[2]],label = plotme.columns.values[2])
#sns.distplot(plotme[[3]],label = plotme.columns.values[3])
plt.legend();
#plt.ylim([0,.000004])
#plt.xlim([-1,1])


# In[83]:

#processing.scale_data(ld_subset,["KIT H : BV421 H_log"],overwrite=False)


# In[84]:

#sns.distplot(ld_subset["KIT H : BV421 H_log_scaled"], label = "FSC-H")
#sns.distplot(ld_subset["KIT H : BV421 H_log"], label = "FSC-H")

f, ax = plt.subplots(figsize=(10, 10))
sns.distplot(ld.loc[(ld.cell_type=='live'),"DAPI A"],label = "live",color="lightgreen")
sns.distplot(ld.loc[(ld.cell_type=='dead'),"DAPI A"],label = "dead", color="green")
sns.distplot(ld.loc[(ld.cell_type=='debris'),"DAPI A"],label = "debris", color = "darkgreen")
plt.legend();
plt.ylim([0,.00004])
plt.xlim([-100000,400000])f, ax = plt.subplots(figsize=(10, 10))
sns.distplot(ld.loc[(ld.cell_type=='live'),"FSC-H"],label = "live",color="lightblue")
sns.distplot(ld.loc[(ld.cell_type=='dead'),"FSC-H"],label = "dead", color="teal")
sns.distplot(ld.loc[(ld.cell_type=='debris'),"FSC-H"],label = "debris", color = "blue")
plt.legend();
plt.ylim([0,.000003])
plt.xlim([-100000,4000000])ld.loc[(ld.cell_type=='live'),"DAPI A"]def get_median_filtered(signal, threshold=3):
    signal = signal.copy()
    difference = np.abs(signal - np.median(signal))
    median_difference = np.median(difference)
    if median_difference == 0:
        s = 0
    else:
        s = difference / float(median_difference)
    mask = s > threshold
    signal[mask] = np.median(signal)
    return signallabeled_comp_data['FSC-H'].plot(use_index=False,color="red")

labeled_comp_data["index"] =labeled_comp_data.index 
labeled_comp_data["wellscreen"] =labeled_comp_data["well_number"] + "_" + labeled_comp_data["screen_number"]
fg = sns.FacetGrid(data=labeled_comp_data, hue="wellscreen",size=12)
fg.map(plt.scatter, 'index', "FSC-H").add_legend()df = testdata[colsOfInterest]
x = df.sort_values('DAPI A')
x[[3]].plot(use_index=False)


df = testdata[colsOfInterest]
x = df.sort_values('DAPI H')
x[[2]].plot(use_index=False)


df = testdata[colsOfInterest]
x = df.sort_values('FSC-A')
x[[0]].plot(use_index=False)


df = testdata[colsOfInterest]
x = df.sort_values('FSC-H')
x[[1]].plot(use_index=False)

df = testdata[colsOfInterest]
figsize = (10,10)
kw = dict(marker='o', linestyle='none', color='r', alpha=0.3)

df['FSC-A_medf'] = get_median_filtered(df['FSC-A'].values, threshold=10)

outlier_idx = np.where(df['FSC-A_medf'].values != df['FSC-A'].values)[0]

fig, ax = py.subplots(figsize=figsize)
df['FSC-A'].plot()
df['FSC-A'][outlier_idx].plot(**kw)

# #  Clean the Data

# In[ ]:




# 
# # LABELED

# In[440]:

#all of the lasers + scatter
colsOfInterest = labeled_comp_data.filter(regex='log').columns.values
colsOfInterest[colsOfInterest =='FSC-H_log'] = 'FSC-H_scaled'
colsOfInterest[colsOfInterest =='SSC-H_log'] = 'SSC-H_scaled'


#all of the lasers, no scatter
colsOfInterestFlow = colsOfInterest
i= np.where(colsOfInterestFlow=='SSC-H')
colsOfInterestFlow = np.delete(colsOfInterestFlow,i)
i= np.where(colsOfInterestFlow=='FSC-H')
colsOfInterestFlow = np.delete(colsOfInterestFlow,i)
colsOfInterestFlow
#colsOfInterestFlow = temp.filter(regex='log').columns.values


#most correlated lasers 
colsOfInterestSub = ["CD34 H : BV605 H_log","VL5-H_log"] #, "FSC-H","KIT H : BV421 H_log"]
colsOfInterestSubFlow = ["CD34 H : BV605 H_log","VL5-H_log"] #,"KIT H : BV421 H_log"]

#scatter only 
colsOfScatter = type_relevant_columns
colsOfScatter
colsOfInterest


# In[441]:

labeled_plots = plotting.pairwise_plots(subset, colsOfInterest, 'cell_type', opacity=.5)
#for p in labeled_plots: iplot(p)


# In[87]:

labeled_plots = plotting.pairwise_plots(labeled_comp_data, colsOfInterest, 'cell_type', max_points=int(len(labeled_data)**.75), opacity=.5)
#for p in labeled_plots: iplot(p)


# # K MEANS

# In[460]:

def k_means_optimized(testdata, n_clusters_range=range(3,12), scale=True):
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
    for n_clusters in n_clusters_range:
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



# In[447]:

#check out just the target cells 
#d[(d['x']>2) & (d['y']>7)]
temp = subset[(subset['cell_type']=="blast") | (subset['cell_type']=="healthy")]
kmeans_temp, scores_temp = k_means_optimized(temp[colsOfInterest].as_matrix(),scale=True)    

temp['kmeans_temp'] = kmeans_temp.labels_
print kmeans_temp
print scores_temp
plt.bar(range(len(scores_temp)), scores_temp.keys(), align='center')
plt.xticks(range(len(scores_temp)), scores_temp.values())

plt.show()
print 'KMEANS DAD NMI:', adjusted_mutual_info_score(temp['cell_type'], kmeans_temp.labels_)
temp.groupby(['cell_type',"kmeans_temp"]).count()


# In[448]:

#check out just the target cells forced k=2
kmeans_temp = KMeans(2)
kmeans_temp.fit(temp[colsOfInterestFlow].as_matrix())
temp['kmeans_temp2'] = kmeans_temp.labels_
print kmeans_temp
print scores_temp
plt.bar(range(len(scores_temp)), scores_temp.keys(), align='center')
plt.xticks(range(len(scores_temp)), scores_temp.values())

plt.show()
print 'KMEANS DAD NMI:', adjusted_mutual_info_score(temp['cell_type'], kmeans_temp.labels_)
temp.groupby(['cell_type',"kmeans_temp2"]).count()


# In[449]:

plt.plot(kmeans_temp.cluster_centers_[0],label="0")
plt.plot(kmeans_temp.cluster_centers_[1],label="1")
#plt.plot(kmeans_temp.cluster_centers_[2],label="2")
plt.legend()
plt.xticks(range(0,len(colsOfInterestFlow)), colsOfInterestFlow, rotation='vertical')
colsOfInterestSub


# In[452]:

#cluster DAD
kmeans_DAD, scores_DAD = k_means_optimized(subset[colsOfScatter].as_matrix())    
    
    
#how did we do at DAD?
subset['kmeans_DAD'] = kmeans_DAD.labels_
print kmeans_DAD
print scores_DAD
plt.bar(range(len(scores_DAD)), scores_DAD.keys(), align='center')
plt.xticks(range(len(scores_DAD)), scores_DAD.values())

plt.show()
print 'KMEANS DAD NMI:', adjusted_mutual_info_score(subset['cell_type'], kmeans_DAD.labels_)
subset.groupby(['cell_type',"kmeans_DAD"]).count()



# In[453]:

x = subset.groupby(['cell_type',"kmeans_DAD"]).mean()
x[colsOfInterest]


# In[454]:

f, ax = plt.subplots(figsize=(15, 15))
x = subset.groupby(['cell_type']).mean()
y = x[colsOfInterest]
a = y.iloc[0,:]
b = y.iloc[1,:]
c = y.iloc[2,:]
a.plot(label="blast",rot=0)
b.plot(label="healthy")
c.plot(label="nontarget",rot=90)
plt.xlabel = colsOfInterest
plt.legend()


# In[455]:

# plot DAD
plt.plot(kmeans_DAD.cluster_centers_[0],label="0")
plt.plot(kmeans_DAD.cluster_centers_[1],label="1")
plt.plot(kmeans_DAD.cluster_centers_[2],label="2")
#plt.plot(kmeans_DAD.cluster_centers_[3],label="3")
plt.xticks(range(0,len(colsOfInterest)), colsOfInterest, rotation='vertical')
plt.legend()
colsOfInterest


# In[456]:

#now get just the remaining  get only good cells -- cluster TYPE 
x = subset.groupby(['cell_type',"kmeans_DAD"]).count()
subset_TYPE = subset[subset['kmeans_DAD']==2]


# In[457]:

#now cluster just the remaining 
kmeans_TYPE, scores_TYPE = k_means_optimized(subset_TYPE[colsOfInterestFlow].as_matrix())


# In[458]:

colsOfInterestSubFlow


# In[459]:

#how did it do? 
subset_TYPE['kmeans_TYPE'] = kmeans_TYPE.labels_
print kmeans_TYPE
print scores_TYPE
plt.bar(range(len(scores_TYPE)), scores_TYPE.keys(), align='center')
plt.xticks(range(len(scores_TYPE)), scores_TYPE.values())

print 'KMEANS TYPE NMI:', adjusted_mutual_info_score(subset_TYPE['cell_type'], kmeans_TYPE.labels_)
plt.show()
subset_TYPE.groupby(['cell_type',"kmeans_TYPE"]).count()



# In[423]:

kmeans_TYPE.cluster_centers_[0]


# In[424]:

plt.plot(kmeans_TYPE.cluster_centers_[0],label="0")
plt.plot(kmeans_TYPE.cluster_centers_[1],label="1")
plt.plot(kmeans_TYPE.cluster_centers_[2],label="2")
plt.plot(kmeans_TYPE.cluster_centers_[3],label="3")
#plt.plot(kmeans_TYPE.cluster_centers_[4],label="4")
plt.xticks(range(0,len(colsOfInterestSubFlow)), colsOfInterestSubFlow, rotation='vertical')
plt.legend()
colsOfInterest


# In[469]:

#now cluster just the remaining 
#now get just the remaining  get only good cells -- cluster TYPE 
x = subset.groupby(['cell_type',"kmeans_DAD"]).count()
subset_TYPE = subset[subset['kmeans_DAD']==2]

kmeans_TYPE, scores_TYPE = k_means_optimized(subset_TYPE[colsOfInterestFlow].as_matrix())

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
plt.xticks(range(0,len(colsOfInterestFlow)), colsOfInterestFlow, rotation='vertical')
plt.legend()
subset_TYPE.groupby(['cell_type',"kmeans_TYPE"]).count()
#subset_TYPE.groupby(['cell_type',"kmeans_TYPE"]).mean()


# In[470]:

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


# In[398]:

index[:] + .25


# In[373]:

kmeans_TYPE.cluster_centers_[0]


# In[464]:

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




# In[341]:

colsOfInterest


# In[468]:

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


# In[407]:

cmap[0]


# 
# 
# # Hierarchical Clustering

# In[221]:

plt.show()


# In[ ]:

X = subset_TYPE[colsOfInterestSubFlow].as_matrix()
plt.scatter(X[:,0], X[:,1])
plt.show()


# In[102]:

Z = linkage(X, 'ward')
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

c, coph_dists = cophenet(Z, pdist(X))
c


# In[103]:

Z


# In[104]:

subset_TYPE.loc[(subset_TYPE['cell_type']=='healthy'),"color"] = "blue"
subset_TYPE.loc[(subset_TYPE['cell_type']=='blast'),"color"] = "red"
subset_TYPE.loc[(subset_TYPE['cell_type']=='nontarget'),"color"] = "green"


# In[105]:

xlabels = subset_TYPE["cell_type"].tolist()


# In[1]:

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
    
    


# In[179]:


y =  ax.get_xticklabels()
y[[1]]


# In[157]:

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


# In[181]:


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




# In[ ]:



