
## -- get set up ------------------------------------------------------------------------------------------------------------------------------------

get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')
get_ipython().magic(u'load_ext rmagic')

#DAD or TYPE 
get_ipython().magic(u'matplotlib inline')
classification = "TYPE"

import pandas as pd
import numpy as np

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_mutual_info_score


# pull the right settings 
if classification == "DAD":
    get_ipython().magic(u"cd '/Users/jorie/Dropbox (Personal)/Insight_Personal/Analyses/ActiveCode/DAD'")
    import settings
elif classification == "TYPE":
    get_ipython().magic(u"cd '/Users/jorie/Dropbox (Personal)/Insight_Personal/Analyses/ActiveCode/TYPE'")
    import settings
    

# get in the rest of the analysis code 
get_ipython().magic(u"cd '/Users/jorie/Dropbox (Personal)/Insight_Personal/Analyses/ActiveCode'")
import processing
import plotting
import fcsparser as fcs

# final packages (plot should go last)
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode()
import matplotlib.pylab as plt


## Grab the data ------------------------------------------------------------------------------------------------------------------------------------

# which columns? 

if classification == "DAD":
    relevant_columns = ['FSC-H', 'SSC-H', 'DAPI H', 'FSC-A', 'SSC-A', 'DAPI A']
elif classification == "TYPE":
    type_relevant_columns = ['FSC-H', 'SSC-H']

import glob
import os

#check you are getting what you want 
location = settings.DATA_LOCATION
filename_template = "screen_*_cell_plate_1_labeled/*.fcs"
filename_template = "training_set/*.fcs"

filenames = glob.glob(os.path.join(location, filename_template))
filenames


#%debug
#get the data with the column names labeled by compound 
if classification == "DAD":
    compound_data_uncat = processing.load_and_process_data(template = "*originallabeled.fcs", features_to_scale=None)
elif classification == "TYPE":
    compound_data_uncat = processing.load_and_process_data(template = "training_set/*.fcs", features_to_scale=None)
    
compound_data = pd.concat(compound_data_uncat, join='outer', ignore_index=True)
compound_data.index = compound_data[settings.EVENT_IDENTIFYING_COLUMNS]
compound_data

compound_data['cell_type'].unique()



#get the data with the column names labeled by compound 
if classification == "DAD":
    labeled_data_uncat = processing.load_and_process_data(template = "*.fcs", features_to_scale=None)
elif classification == "TYPE":
    labeled_data_uncat = processing.load_and_process_data(template = "screen_*_cell_plate_1_labeled/*.fcs", features_to_scale=None)
    
labeled_data = pd.concat(labeled_data_uncat, join='outer', ignore_index=True)
labeled_data.index = labeled_data[settings.EVENT_IDENTIFYING_COLUMNS]
labeled_data
labeled_data['cell_type'].unique()


#check out your data
x = compound_data.count(0)
print(x.sort_values())

#check out more of your data
with pd.option_context('display.max_rows', 999, 'display.max_columns', 900):
    x = labeled_data.count(0)
    print(x.sort_values())

cols = list(compound_data.columns.values)
cols = filter(lambda x:'H' in x, cols)
cols

labeled_data.groupby('screen_number').size()

#labeled_data[['DAPI A']].groupby(['screen_number']).agg(['mean', 'count'])
table = labeled_data.groupby(['screen_number'])[cols].count()

## merge the data ------------------------------------------------------------------------------------------------------------------------------------

#merge the labeled and unlabeled data (effectively, adding labels to complete dataset)
try:
    labeled_comp_data = processing.add_labeled_columns(labeled_data, compound_data)
    labeled_comp_data
except:
    labeled_comp_data = processing.add_labeled_columns(labeled_data, compound_data)
    labeled_comp_data
    
print(labeled_comp_data)

labeled_comp_data['cell_type_y'].unique()


#clean up the NaNs in the unlabled data 
labeled_comp_data.loc[labeled_comp_data['cell_type_y'].isnull(),'cell_type_y'] = "nontarget"
labeled_comp_data['cell_type'] = labeled_comp_data['cell_type_y']


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


#check out that you have reasonable data: how many cell types
labeled_comp_data.groupby('cell_type').count()


#check out that you have reasonable data: how many screens
#labeled_comp_data.groupby('screen_number').count()

#check out that you have reasonable data: how many wells
#labeled_comp_data.groupby('well_number').count()

#labeled_comp_data.to_csv("labeled_comp_data.csv")

## ----- SQL ------------------------------------------------------------------------------------------------------------------------------------
# from sqlalchemy import *
#  
# connection_string = "mysql://root:password@ireland-mysql-instance1.abcdefg12345.eu-west-1.rds.amazonaws.com:3306/DatabaseName"
#  
#  
# db = create_engine(connection_string)

# #connection_string = "mysql://root:password@ireland-mysql-instance1.abcdefg12345.eu-west-1.rds.amazonaws.com:3306/DatabaseName"
# 
# 
# from sqlalchemy import *
# engine = create_engine('postgresql://jorie:pancakes@autocytedata.cdxamlee3sna.us-west-2.rds.amazonaws.com:5432/AutoCyte')
# #labeled_comp_data.to_sql('labeled_comp_data', engine)
# 
# labeled_comp_data.to_sql("labeled_comp_data", engine)
# 
# 
# 

## ----- Data SubSet ------------------------------------------------------------------------------------------------------------------------------------

#grab your subset of data
screenTarget = "525"
wellTarget = "c12" 

subset = labeled_comp_data.loc[(labeled_comp_data.screen_number==screenTarget)&(labeled_comp_data.well_number==wellTarget)]
subset

#drop any empty columns
notempty = subset.isnull().sum() != subset.shape[0]
#subset.loc[notempty]
notempty.shape
subset = subset.loc[:,notempty]


#grab the columns of interest, and log and scale them 
x = subset.filter(regex='H')
#x.drop('FSC-H',1)
#x.drop('SSC-H',1)
scalenames = list(x.columns.values)

#processing.scale_data(labeled_comp_data,scalenames)
processing.log_data(subset,scalenames)
processing.boxcox_data(subset,scalenames)
list(subset.columns.values)

#are there any missing values? 
numeric_cols = subset._get_numeric_data().columns.values
subset[numeric_cols].isnull().sum().sum()

#if so, replace NAs with median 
numeric_cols = subset._get_numeric_data().columns.values
x = subset[numeric_cols].fillna(subset[numeric_cols].median())
subset[numeric_cols] = x 
subset[numeric_cols].isnull().sum()


#make sure there is something to analyze
subset.groupby('cell_type').count()


## ----- Explore the Data ------------------------------------------------------------------------------------------------------------------------------------

import seaborn as sns

# different data formats
logcols = subset.filter(regex='log').columns.values
logcols

bccols = subset.filter(regex='_bc_').columns.values
bccols

hcols = subset.filter(regex='H$').columns.values
hcols

# scale the bccols because they are over the place 
processing.scale_data(subset,bccols)
bcscaledcols = subset.filter(regex='scaled').columns.values

# how skew are we? 
from scipy.stats import skew, skewtest
skness = skew(subset[bcscaledcols])
sknessp = skewtest(subset[bcscaledcols])

print(bccols)
print(skness)
print(sknessp[1])
sum(sknessp[1]<.01)



# check out the distributions 
x = 5
sns.distplot(subset[bccols[x]], label = bccols[x], kde=False)
sns.distplot(subset[logcols[x]], label = logcols[x], kde=False)

## ------- Semantic Knowledge ----------------------------------------------------------------------------------------------------------------
    
# *CD34
#     CD117
# CD38
#     HLA-DR
#     CD13
# *CD33
#     CD15
#     MPO
# CD14
#     CD64
#     CD36
#     CD235a
#     CD71
#     CD41
#     CD61
# (CD19)
#     CD79a
#     CD10
# (CD3)
#     
# -----
# 
# CD11
# CD16
# CD163
# 
# 
# *CD45 -- mid/low
# *CD66b -- healthy
# 
# *KIT
# 
# 
# http://www.cytometry.org/public/educational_presentations/Cherian.pdf page 13

## ------- select the content-based columns ----------------------------------------------------------------------------------------------------------------

# cancer
x = zip(*np.where(np.char.find(bcscaledcols.astype('str'), 'CD') > -1))
#x = pd.DataFrame(x)
try:
    y = zip(*np.where(np.char.find(bcscaledcols.astype('str'), 'KIT') > -1))
except:
    pass
x.append(y[0])
x = pd.DataFrame(x)
cancercols = bcscaledcols[x[0]]
cancercols

# DAD 
x = zip(*np.where(np.char.find(hcols.astype('str'), 'SC') > -1))
#x = pd.DataFrame(x)
y = zip(*np.where(np.char.find(hcols.astype('str'), 'DAPI') > -1))
try:
    x.append(y[0])
except:
    pass
x = pd.DataFrame(x)
DADcols = hcols[x[0]]
DADcols

processing.scale_data(subset,DADcols)
subset


#which cols?  
DADcolsscaled = DADcols+'_scaled'

#histogram of the cancer cols 
my_colors = sns.hls_palette(13, l=.3, s=1)
ax = subset[cancercols].plot.hist(alpha=.5,bins=100,color=my_colors, figsize=(15,15))
ax.set_xlim( -5,5 )
ax.set_ylim( 0,1500 )

#histogram of the DAD cols 
my_colors = sns.hls_palette(13, l=.3, s=1)
ax = subset[DADcolsscaled].plot.hist(alpha=.5,bins=100,color=my_colors, figsize=(15,15))
#ax.set_xlim(0,4000000)
ax.set_ylim( 0,1500 )

#histogram of ALL the columns 
bccols.size
plt.figure();
subset[logcols].plot.hist(bins=50, alpha=.3)

#hows the outlier situation? 
subset[logcols].plot.box(figsize=(15,15))
f, ax = plt.subplots(figsize=(15, 15))
#all flow 
sns.boxplot(subset[bcscaledcols])
# just DAD 
sns.boxplot(subset[DADcolsscaled])

##  Clean the Data  ---------------------------------------------------------------------------------------------------------
# trim out the data over 4 standard deviations away 

#subset = subset_uncut
cutdata, cut = processing.outlier_data(subset, features_to_scale=DADcolsscaled, overwrite=False,threshold=3)
print(cut)
print(cutdata.shape)
print(subset.shape)

float(cutdata.shape[0])/float(subset.shape[0])

subset_uncut = subset
subset = cutdata

# double check the triming process 
#points = subset['CD16 H : APC H_bc_0.28_scaled']

#median = np.median(points)
#diff = (points - median)**2  #square it 
#diff = np.sqrt(diff)  #root it 
#med_abs_deviation = np.median(diff)
#modified_z_score = 0.6745 * diff / med_abs_deviation  # 75% of the data     
#x= modified_z_score < 2


#median = np.median(points)
#diff = (points - median)**2  #square it 
#diff = np.sqrt(diff)  #root it 
#med_abs_deviation = 1.4826 * np.median(diff)
#modified_z_score =  diff / med_abs_deviation 
#y= modified_z_score < 2



## PLOTTING ------------------------------------------------------------------------------------------------------------------------------------------------

#re order for plotting 

#all data
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

#subset data 

#cols = subset.columns.tolist()

b = [subset.columns.get_loc("is_blast"),subset.columns.get_loc("is_healthy"), 
     subset.columns.get_loc("is_live"),subset.columns.get_loc("is_dead"),
     subset.columns.get_loc("is_debris")]
b.sort()
a = [ cols[i] for i in b]

cols[b[0]:b[len(b)-1]+1] = []
cols[0:0] = a
cols
ld = subset[cols]


# name the log cols 
logcols = subset.filter(regex='log|is').columns.values
logcols

## all data 

# Load the datset of correlations between cortical brain networks
corrmat = ld[logcols].corr()

# Set up the matplotlib figure
mask = np.zeros_like(corrmat, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Draw the heatmap using seaborn
f, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corrmat, vmax=.8, square=True, mask=mask)

## subset data 

ld_subset = ld.loc[(ld.screen_number==screenTarget)&(ld.well_number==wellTarget)]

# Load the datset of correlations between cortical brain networks
corrmat = ld_subset[bcscaledcols].corr()

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(20, 20))


mask = np.zeros_like(corrmat, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=.8, square=True, mask=mask)


#what correlates with being a blast? 
#x=corrmat[["is_blast"]]
#x.sort_index(by=['is_blast'], ascending=[False])
#x = x.sort_values('is_blast')
#x =  x[x<1]
#x
#x.plot(figsize=(20, 20),kind="bar")
#corrcols  = list(x[x['is_blast']>.1].index)

#plotme = ld_subset[corcols]

#f, ax = plt.subplots(figsize=(10, 10))
#sns.distplot(plotme[[0]], label = plotme.columns.values[0])
#sns.distplot(plotme[[1]],label = plotme.columns.values[1])
#sns.distplot(plotme[[2]],label = plotme.columns.values[2])
#sns.distplot(plotme[[3]],label = plotme.columns.values[3])
#plt.legend();
#plt.ylim([0,.000004])
#plt.xlim([-1,1])

#f, ax = plt.subplots(figsize=(10, 10))
#sns.distplot(ld.loc[(ld.cell_type=='live'),"DAPI A"],label = "live",color="lightgreen")
#sns.distplot(ld.loc[(ld.cell_type=='dead'),"DAPI A"],label = "dead", color="green")
#sns.distplot(ld.loc[(ld.cell_type=='debris'),"DAPI A"],label = "debris", color = "darkgreen")
#plt.legend();
#plt.ylim([0,.00004])
#plt.xlim([-100000,400000])f, ax = plt.subplots(figsize=(10, 10))
#sns.distplot(ld.loc[(ld.cell_type=='live'),"FSC-H"],label = "live",color="lightblue")
#sns.distplot(ld.loc[(ld.cell_type=='dead'),"FSC-H"],label = "dead", color="teal")
#sns.distplot(ld.loc[(ld.cell_type=='debris'),"FSC-H"],label = "debris", color = "blue")
#plt.legend();
#plt.ylim([0,.000003])
#plt.xlim([-100000,4000000])ld.loc[(ld.cell_type=='live'),"DAPI A"]

#labeled_comp_data["index"] =labeled_comp_data.index 
#labeled_comp_data["wellscreen"] =labeled_comp_data["well_number"] + "_" + labeled_comp_data["screen_number"]
#fg = sns.FacetGrid(data=labeled_comp_data, hue="wellscreen",size=12)
#fg.map(plt.scatter, 'index', "FSC-H").add_legend()df = testdata[colsOfInterest]
#x = df.sort_values('DAPI A')
#x[[3]].plot(use_index=False)





# The columns we car eabout  

DADcolsscaled
bcscaledcols
cancercols 
#corrcols






# # KMEANS proper

# define it 

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




#cluster DAD
CL=0
print("this cluster: top level")
kmeans,scores = k_means_optimized(subset[DADcolsscaled].as_matrix(),nrange=range(2,12))

    
#how did we do at DAD?
subset['kmeans_'+str(CL)] = kmeans.labels_
nclusters = kmeans.n_clusters
 
print kmeans
print scores
plt.bar(range(len(scores)), scores.keys(), align='center')
plt.xticks(range(len(scores)), scores.values())

print 'KMEANS NMI:', adjusted_mutual_info_score(subset['cell_type'], kmeans.labels_)

table = subset.groupby(['cell_type',"kmeans_0"]).count()
print(table[["FSC-H"]])
print("has " + str(nclusters) + "subclusters")

CL = CL +  1
from scipy.stats import gaussian_kde

## plot it with one color 
import numpy as np
import matplotlib.pyplot as plt

x = subset[DADcolsscaled[0]]
y = subset[DADcolsscaled[1]]

# Calculate the point density
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]

fig, ax = plt.subplots()
ax.scatter(x, y, c=z, cmap="Blues_r", s=50, edgecolor='')

ax.set_xlim([-1, 3])
ax.set_ylim([-1, 3])
plt.show()


# make a graph with all the colors for KDAD

fig, ax = plt.subplots(figsize=(10, 10))
teals = sns.light_palette("#18cfed", as_cmap=True, reverse=True)
brightgreens = sns.light_palette("#03f90c", as_cmap=True, reverse=True)

cmaps = ("Blues_r","Reds_r","Greens_r","Purples_r","Reds_r","Greys_r","Blues_r","Reds_r","Greens_r","Oranges_r","Purples_r","Greys_r")


sns.cubehelix_palette(as_cmap=True,reverse=True,start=.1, rot=0, dark=.3)

for n in range(0, nclusters):
    
    x = subset.loc[subset['kmeans_0']==n, DADcolsscaled[0]]
    y = subset.loc[subset['kmeans_0']==n, DADcolsscaled[1]]

    # Calculate the point density
    xy = np.vstack([x,y])
    z1 = gaussian_kde(xy)(xy)

    
#    ax= sns.kdeplot(x, y,cmap=cmaps[n],shade=True,shade_lowest=False)
    ax.scatter(x, y, c=z1, cmap=cmaps[n], s=50, edgecolor='',alpha=1)


ax.set_xlim([-1, 2.5])
ax.set_ylim([-1, 2.5])
ax.set_xlabel(DADcolsscaled[0])
ax.set_ylabel(DADcolsscaled[1])
ax.set_axis_bgcolor('white')
ax.grid(b=True, which='both', color="gray",linestyle='-',alpha=.1)
#plt.show()


# Next Level 2 

CL = 1 
#cluster level 2 
for upperclusters in range(0,nclusters):
    print('--------')
    print("this cluster: " + str(upperclusters))
    subdata = subset[subset['kmeans_'+str(0)]==upperclusters]
    kmeans,scores = k_means_optimized(subdata[bcscaledcols].as_matrix(),nrange=range(3,12))    

    subset.loc[subset['kmeans_'+str(0)]==upperclusters,'kmeans_'+str(CL)] = kmeans.labels_
    subdata['kmeans_'+str(CL)] = kmeans.labels_
    #print kmeans
    #print scores
    #plt.bar(range(len(scores)), scores.keys(), align='center')
    #plt.xticks(range(len(scores)), scores.values())
    #plt.show()
    print 'KMEANS NMI:', adjusted_mutual_info_score(subset.loc[subset['kmeans_'+str(0)]==upperclusters,'cell_type'], kmeans.labels_)
    z=subdata.groupby(['cell_type','kmeans_'+str(CL)]).count()
    
    nclusters2 = kmeans.n_clusters
    print("has " +str(nclusters2) + " sunclusters")
    print(z[["FSC-H"]])
    
    
    fig, ax = plt.subplots(figsize=(10, 10))
    for n in range(0, nclusters2):

        x = subdata.loc[subdata['kmeans_'+str(CL)]==n, bcscaledcols[2]]
        y = subdata.loc[subdata['kmeans_'+str(CL)]==n, bcscaledcols[3]]

        # Calculate the point density
        xy = np.vstack([x,y])
        z1 = gaussian_kde(xy)(xy)


    #   ax= sns.kdeplot(x, y,cmap=cmaps[n],shade=True,shade_lowest=False)
        ax.scatter(x, y, c=z1, cmap=cmaps[n], s=50, edgecolor='',alpha=1)


    ax.set_xlim([-1, 2.5])
    ax.set_ylim([-1, 2.5])
    ax.set_xlabel(bcscaledcols[2])
    ax.set_ylabel(bcscaledcols[3])
    ax.set_axis_bgcolor('white')
    ax.grid(b=True, which='both', color="gray",linestyle='-',alpha=.1)
    ax.text(2.2,2.2, ('CL0= ' + str(upperclusters)))
    plt.show()

    


#cluster level 2 
CL = 2 
for upperclusters in range(0,nclusters):
    for midclusters in range(0,nclusters2):
        print('--------')
        print("this cluster: " + str(upperclusters) + str(midclusters))
        subdata = subset[(subset['kmeans_'+str(0)]==upperclusters) & (subset['kmeans_'+str(1)]==midclusters)]
        kmeans,scores = k_means_optimized(subdata[cancercols].as_matrix(),nrange=range(3,12))    
    
        subset.loc[(subset['kmeans_'+str(0)]==upperclusters) & (subset['kmeans_'+str(1)]==midclusters),'kmeans_'+str(CL)] = kmeans.labels_
        subdata['kmeans_'+str(CL)] = kmeans.labels_
        #print kmeans
        #print scores
        #plt.bar(range(len(scores)), scores.keys(), align='center')
        #plt.xticks(range(len(scores)), scores.values())
        #plt.show()
        print 'KMEANS NMI:', adjusted_mutual_info_score(subset.loc[(subset['kmeans_'+str(0)]==upperclusters) & (subset['kmeans_'+str(1)]==midclusters),'cell_type'], kmeans.labels_)
        z=subdata.groupby(['cell_type','kmeans_'+str(CL)]).count()
        
        nclusters3 = kmeans.n_clusters
        print("has " +str(nclusters3) + " sunclusters")
        print(z[["FSC-H"]])
        
        
        fig, ax = plt.subplots(figsize=(10, 10))
        for n in range(0, nclusters2):
    
            x = subdata.loc[subdata['kmeans_'+str(CL)]==n, cancercols[2]]
            y = subdata.loc[subdata['kmeans_'+str(CL)]==n, cancercols[3]]
    
            # Calculate the point density
            xy = np.vstack([x,y])
            z1 = gaussian_kde(xy)(xy)
    
    
        #   ax= sns.kdeplot(x, y,cmap=cmaps[n],shade=True,shade_lowest=False)
            ax.scatter(x, y, c=z1, cmap=cmaps[n], s=50, edgecolor='',alpha=1)
    
    
        ax.set_xlim([-1, 2.5])
        ax.set_ylim([-1, 2.5])
        ax.set_xlabel(cancercols[2])
        ax.set_ylabel(cancercols[3])
        ax.set_axis_bgcolor('white')
        ax.grid(b=True, which='both', color="gray",linestyle='-',alpha=.1)
        ax.text(1,1, ('CL0= ' + str(upperclusters) + ' CL1=' + str(midclusters)))
        plt.show()

    
CL = CL + 1





plotdata = subset[(subset['kmeans_'+str(0)]==0) & (subset['kmeans_'+str(1)]==1)]

f, ax = plt.subplots(figsize=(15, 15))
x = plotdata.groupby(['cell_type']).mean()
y = x[bcscaledcols]
a = y.iloc[0,:]
#b = y.iloc[1,:]
#c = y.iloc[2,:]
a.plot(label="blast",rot=0)
#b.plot(label="healthy")
c.plot(label="nontarget",rot=90)
plt.xlabel = bcscaledcols
plt.legend()

f, ax = plt.subplots(figsize=(15, 15))
x = plotdata.groupby(['kmeans_2']).mean()
y = x[bcscaledcols]
a = y.iloc[0,:]
#b = y.iloc[1,:]
#c = y.iloc[2,:]
a.plot(label="0",rot=0)
#b.plot(label="1")
#c.plot(label="2",rot=90)
plt.xlabel = bcscaledcols
plt.legend()





#--------------------------------------------------------------------------------------------------------------------------------------------------------
# Plot what are already have! 


CL=0


table = subset.groupby(['cell_type',"kmeans_0"]).count()
print(table[["FSC-H"]])
print("has " + str(nclusters) + "subclusters")

fig, ax = plt.subplots(figsize=(10, 10))
teals = sns.light_palette("#18cfed", as_cmap=True, reverse=True)
brightgreens = sns.light_palette("#03f90c", as_cmap=True, reverse=True)
cmaps = ("Blues_r","Reds_r","Greens_r","Purples_r","Reds_r","Greys_r","Blues_r","Reds_r","Greens_r","Oranges_r","Purples_r","Greys_r")

fig, ax = plt.subplots(figsize=(10, 10))
x = subset.loc[:, DADcolsscaled[0]]
y = subset.loc[:, DADcolsscaled[1]]

# Calculate the point density
xy = np.vstack([x,y])
z1 = gaussian_kde(xy)(xy)
ax.scatter(x, y, c=z1, cmap=cmaps[n], s=50, edgecolor='',alpha=1)

ax.set_xlim([-.5, 1.5])
ax.set_ylim([-1, 2])
ax.set_xlabel("Front Scatter (Z-scored)",fontsize=25)
ax.set_ylabel("Side Scatter (Z-scored)",fontsize=25)
ax.set_axis_bgcolor('white')
ax.grid(b=True, which='both', color="gray",linestyle='-',alpha=.1)
plt.show()
plt.savefig('C0.png', bbox_inches='tight', dpi=150)


fig, ax = plt.subplots(figsize=(10, 10))
for n in range(0, 3):
    
    x = subset.loc[subset['kmeans_0']==n, DADcolsscaled[0]]
    y = subset.loc[subset['kmeans_0']==n, DADcolsscaled[1]]

    # Calculate the point density
    xy = np.vstack([x,y])
    z1 = gaussian_kde(xy)(xy)

    
#    ax= sns.kdeplot(x, y,cmap=cmaps[n],shade=True,shade_lowest=False)
    ax.scatter(x, y, c=z1, cmap=cmaps[n], s=50, edgecolor='',alpha=1)


ax.set_xlim([-.5, 1.5])
ax.set_ylim([-1, 2])
ax.set_xlabel("Front Scatter (Z-scored)",fontsize=25)
ax.set_ylabel("Side Scatter (Z-scored)",fontsize=25)
ax.set_axis_bgcolor('white')
ax.grid(b=True, which='both', color="gray",linestyle='-',alpha=.1)
plt.show()
plt.savefig('C0.png', bbox_inches='tight', dpi=150)

 
f, ax = plt.subplots(figsize=(15, 15))
x = subset.groupby(['kmeans_0']).mean()
y = x[cols[0:2]]
ax = y.T.plot.barh(color=("#4392c6","#fb7757","#4aaf61"))
ax.set_axis_bgcolor('white')
ax.grid(b=True, which='both', color="gray",linestyle='-',alpha=.1)
ax.tick_params(axis='y', labelsize=10,direction=45)
plt.xticks(rotation=70)
plt.legend(title="", fancybox=True)
plt.savefig('C0_filter1.png', bbox_inches='tight',dpi=150)

f, ax = plt.subplots(figsize=(15, 15))
x = subset.groupby(['kmeans_0']).mean()
y = x[cols[2: len(cols)]]
ax = y.T.plot.barh(color=("#4392c6","#fb7757","#4aaf61"))
ax.set_axis_bgcolor('white')
ax.grid(b=True, which='both', color="gray",linestyle='-',alpha=.1)
ax.tick_params(axis='y', labelsize=10)
plt.xticks(rotation=70)
plt.legend(title="", fancybox=True)
plt.savefig('C0_filter2.png', bbox_inches='tight',dpi=150)



f, ax = plt.subplots(figsize=(10, 5))
x = subset.groupby(['kmeans_0']).mean()
y = x.loc[0,cols]
y[1] = y[1]/10
y[0] = y[0]/10
ax = y.T.plot.barh(color=("darkred", "red","orange","gold","yellow","greenyellow", "lightgreen","green","turquoise", "deepskyblue","blue","purple",'m',"violet", "pink"))
ax.set_axis_bgcolor('white')
ax.grid(b=True, which='both', color="gray",linestyle='-',alpha=.1)
ax.tick_params(axis='y', labelsize=10)
plt.xticks(rotation=70)








# Next Level 2 

CL = 1 
#cluster level 2 
nclusters = subset['kmeans_'+str(0)].max()+1

for upperclusters in range(0,nclusters):
    print('--------')
    print("this cluster: " + str(upperclusters))
    
    subdata = subset[subset['kmeans_'+str(0)]==upperclusters]
    nclusters2 = subdata['kmeans_'+str(1)].max()+1
    print("has " +str(nclusters2) + " sunclusters")
    z=subdata.groupby(['cell_type','kmeans_'+str(CL)]).count()
    print(z[["FSC-H"]])
    
    
    fig, ax = plt.subplots(figsize=(10, 10))
    x = subdata.loc[:, bcscaledcols[12]]
    y = subdata.loc[:, bcscaledcols[3]]
    xy = np.vstack([x,y])
    z1 = gaussian_kde(xy)(xy)
    ax.scatter(x, y, c=z1, cmap=cmaps[n], s=100, edgecolor='',alpha=1)
    #ax.set_xlim([-3, 4])
    #ax.set_ylim([-3, 4])
    ax.set_xlabel(bcscaledcols[12],fontsize=25)
    ax.set_ylabel(bcscaledcols[3],fontsize=25)
    ax.set_axis_bgcolor('white')
    ax.grid(b=True, which='both', color="gray",linestyle='-',alpha=.1)

    
    fig, ax = plt.subplots(figsize=(10, 10))
    for n in range(0, int(nclusters2)):

        x = subdata.loc[subdata['kmeans_'+str(CL)]==n, bcscaledcols[12]]
        y = subdata.loc[subdata['kmeans_'+str(CL)]==n, bcscaledcols[3]]

        # Calculate the point density
        xy = np.vstack([x,y])
        z1 = gaussian_kde(xy)(xy)


    #   ax= sns.kdeplot(x, y,cmap=cmaps[n],shade=True,shade_lowest=False)
        ax.scatter(x, y, c=z1, cmap=cmaps[n], s=100, edgecolor='',alpha=1)


    ax.set_xlim([-3, 4])
    ax.set_ylim([-3, 4])
    ax.set_xlabel(cols[9],fontsize=25)
    ax.set_ylabel(cols[3],fontsize=25)
    ax.set_axis_bgcolor('white')
    ax.grid(b=True, which='both', color="gray",linestyle='-',alpha=.1)
    name = "C1_" + str(upperclusters) + '.png'
    plt.savefig(name, bbox_inches='tight',dpi=150)


    #f, ax = plt.subplots(figsize=(15, 15))
    x = subdata.groupby(['kmeans_1']).mean()
    y = x[cols[0:2]]
    ax = y.T.plot.barh(color=("#4392c6","#fb7757","#4aaf61","#6e5aa8"))
    ax.set_axis_bgcolor('white')
    ax.grid(b=True, which='both', color="gray",linestyle='-',alpha=.1)
    ax.tick_params(axis='y', labelsize=10,direction=45)
    plt.xticks(rotation=70)
    #plt.legend(title="", fancybox=True)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=4, borderaxespad=0., title = "")
    name = "C1_" + str(upperclusters) + '_filter1.png'
    plt.savefig(name, bbox_inches='tight',dpi=150)
 
    
    #f, ax = plt.subplots(figsize=(15, 15))
    x = subset.groupby(['kmeans_1']).mean()
    y = x[cols[2: len(cols)]]
    ax = y.T.plot.barh(color=("#4392c6","#fb7757","#4aaf61","#6e5aa8"))
    ax.set_axis_bgcolor('white')
    ax.grid(b=True, which='both', color="gray",linestyle='-',alpha=.1)
    ax.tick_params(axis='y', labelsize=10)
    plt.xticks(rotation=70)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=4, borderaxespad=0., title = "")
    name = "C1_" + str(upperclusters) + '_filter2.png'
    plt.savefig(name, bbox_inches='tight',dpi=150)
    


#cluster level 2 
CL = 2 
for upperclusters in range(0,nclusters):
    subdata = subset[subset['kmeans_'+str(0)]==upperclusters]
    nclusters2 = subdata['kmeans_'+str(1)].max()+1
    for midclusters in range(0,int(nclusters2)):
        
        print('--------')
        print("this cluster: " + str(upperclusters) + str(midclusters))
        subdata = subset[(subset['kmeans_'+str(0)]==upperclusters) & (subset['kmeans_'+str(1)]==midclusters)]
        nclusters3 = subdata['kmeans_'+str(2)].max()+1
        
        z=subdata.groupby(['cell_type','kmeans_'+str(CL)]).count()
    
        print("has " +str(nclusters3) + " sunclusters")
        print(z[["FSC-H"]])
        
        
        fig, ax = plt.subplots(figsize=(10, 10))
        for n in range(0, int(nclusters3)): 
    
            x = subdata.loc[subdata['kmeans_'+str(CL)]==n, cancercols[2]]
            y = subdata.loc[subdata['kmeans_'+str(CL)]==n, cancercols[3]]
    
            # Calculate the point density
            xy = np.vstack([x,y])
            z1 = gaussian_kde(xy)(xy)
    
    
        #   ax= sns.kdeplot(x, y,cmap=cmaps[n],shade=True,shade_lowest=False)
            ax.scatter(x, y, c=z1, cmap=cmaps[n], s=50, edgecolor='',alpha=1)
    

        ax.set_xlim([-3, 4])
        ax.set_ylim([-3, 4])
        ax.set_xlabel(cols[9],fontsize=25)
        ax.set_ylabel(cols[3],fontsize=25)
        ax.set_axis_bgcolor('white')
        ax.grid(b=True, which='both', color="gray",linestyle='-',alpha=.1)
        name = "C2_" + str(upperclusters) + '.png'
        plt.savefig(name, bbox_inches='tight',dpi=150)
    
        #f, ax = plt.subplots(figsize=(15, 15))
        x = subdata.groupby(['kmeans_2']).mean()
        y = x[cols[0:2]]
        ax = y.T.plot.barh(color=("#4392c6","#fb7757","#4aaf61","#6e5aa8"))
        ax.set_axis_bgcolor('white')
        ax.grid(b=True, which='both', color="gray",linestyle='-',alpha=.1)
        ax.tick_params(axis='y', labelsize=10,direction=45)
        plt.xticks(rotation=70)
        #plt.legend(title="", fancybox=True)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=4, borderaxespad=0., title = "")
        name = "C2_" + str(upperclusters) + '_filter1.png'
        plt.savefig(name, bbox_inches='tight',dpi=150) 

        
        
        #f, ax = plt.subplots(figsize=(15, 15))
        x = subset.groupby(['kmeans_2']).mean()
        y = x[cols[2: len(cols)]]
        ax = y.T.plot.barh(color=("#4392c6","#fb7757","#4aaf61","#6e5aa8"))
        ax.set_axis_bgcolor('white')
        ax.grid(b=True, which='both', color="gray",linestyle='-',alpha=.1)
        ax.tick_params(axis='y', labelsize=10)
        plt.xticks(rotation=70)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=4, borderaxespad=0., title = "")
        name = "C2_" + str(upperclusters) + '_filter2.png'
        plt.savefig(name, bbox_inches='tight',dpi=150) 


z1=subset.groupby(['cell_type','kmeans_'+str(0),'kmeans_'+str(1),'kmeans_'+str(2)]).count()    
z1 = z1[["FSC-H"]]
z1 =pd.DataFrame(z1.to_records())


cancer_count = max(z1.loc[z1['cell_type'] =="blast","FSC-H"])
correct_cancer_count = sum(z1.loc[z1['cell_type'] =="blast","FSC-H"])
accuracy = float(cancer_count) / float(correct_cancer_count)


subset.loc[(subset['kmeans_0'] == cancerindex[['kmeans_0']]) & (subset['kmeans_1'] == cancerindex[['kmeans_1']]) & (subset['kmeans_2'] == cancerindex[['kmeans_2']])]

blastcells  = subset.loc[(subset['kmeans_0'] == int(cancerindex.ix[:,'kmeans_0'].values)) & (subset['kmeans_1'] == int(cancerindex.ix[:,'kmeans_1'].values)) & (subset['kmeans_2'] == int(cancerindex.ix[:,'kmeans_2'].values)) ]

cancerindex = z1.ix[z1["FSC-H"] == max(z1.loc[z1['cell_type'] =="blast","FSC-H"])]

z2=subset.groupby(['kmeans_'+str(0),'kmeans_'+str(1),'kmeans_'+str(2),'cell_type',]).count()    
z2 = z2[["FSC-H"]]
z2 =pd.DataFrame(z2.to_records())


pd.DataFrame(z1.to_records())






