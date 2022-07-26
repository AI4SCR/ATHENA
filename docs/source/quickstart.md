# Quickstart
ATHENA is an open-source computational framework written in Python that facilitates the visualization, processing and analysis of (spatial) heterogeneity from spatial omics data. ATHENA supports any spatially resolved dataset that contains spatial transcriptomic or proteomic measurements, including Imaging Mass Cytometry (IMC), Multiplexed Ion Beam Imaging (MIBI), multiplexed Immunohistochemisty (mIHC) or Immunofluorescence (mIF), seqFISH, MERFISH, Visium.

### **_Important_**
You are strongly advised to read the sections:
- Note on Phenotype Encodings
- Note on Segmentation Masks

at the end of this document.

### Requirements
To use all the capabilities of ATHENA one needs the (cell-) segmenation masks for a sample and the omics-profiles of the observations in the sample (as extracted from the high-dimensional images produced by different omics-technologies).

However, it is also possible to use ATHENA (with some limitations) without segmentation masks and the omics-profiles and just the single-observation coordinates and a classification of those observations (phenotypes).

### Further Resources
- A more comprehensive tutorial on IMC data can be found [here](https://ai4scr.github.io/ATHENA/source/tutorial.html)
- A tutorial on how to load your data can be found [here](https://ai4scr.github.io/ATHENA/source/introduction-spatialOmics.html)


```python
import athena as ath
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

pd.set_option("display.max_columns", 5)
print(ath.__version__)
```

    0.1.2



```python
so = ath.dataset.imc_quickstart()
so
```

    INFO:numexpr.utils:NumExpr defaulting to 8 threads.


    warning: to get the latest version of this dataset use `so = sh.dataset.imc(force_download=True)`





    
    SpatialOmics object with 5505 observations across 2 samples.
        X: 2 samples,
        spl: 2 samples,
            columns: ['pid', 'cell_count', 'immune_cell_count']
        obs: 2 samples,
            columns: ['meta_id', 'CellId', 'tumor_immune_id', 'x', 'y', 'meta_label', 'cell_type_id', 'phenograph_cluster', 'tumor_immune', 'cell_type', 'core', 'id']
        var: 2 samples,
            columns: ['channel', 'full_target_name', 'feature_type', 'target', 'fullstack_index', 'metal_tag']
        G: 0 samples,
            keys: []
        masks: 2 samples
            keys: [{'cellmasks'}]
        images: 0 samples



### Sample-level Meta Data
Sample-level meta data is stored in the `so.spl` attribute but not required to use ATHENA.


```python
so.spl.head(3) 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pid</th>
      <th>cell_count</th>
      <th>immune_cell_count</th>
    </tr>
    <tr>
      <th>core</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>SP43_75_X2Y5</th>
      <td>75</td>
      <td>2771</td>
      <td>940</td>
    </tr>
    <tr>
      <th>SP41_239_X11Y3_165</th>
      <td>239</td>
      <td>2734</td>
      <td>412</td>
    </tr>
  </tbody>
</table>
</div>



### Observation-level Meta Data
Observation-level meta data is stored in the `so.obs` attribute and required to use ATHENA. At least the coordinates of the 
observations along with some classification of those (phenotypes) are required. 


```python
spl = so.spl.index[0]
so.obs[spl].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>core</th>
      <th>meta_id</th>
      <th>...</th>
      <th>y</th>
      <th>x</th>
    </tr>
    <tr>
      <th>cell_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>SP43_75_X2Y5</td>
      <td>9</td>
      <td>...</td>
      <td>1.142857</td>
      <td>33.107143</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SP43_75_X2Y5</td>
      <td>3</td>
      <td>...</td>
      <td>4.868613</td>
      <td>43.693431</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SP43_75_X2Y5</td>
      <td>10</td>
      <td>...</td>
      <td>5.467391</td>
      <td>61.326087</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SP43_75_X2Y5</td>
      <td>8</td>
      <td>...</td>
      <td>2.353659</td>
      <td>79.987805</td>
    </tr>
    <tr>
      <th>5</th>
      <td>SP43_75_X2Y5</td>
      <td>3</td>
      <td>...</td>
      <td>2.026316</td>
      <td>128.302632</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 12 columns</p>
</div>



### Observation-level Omics Profiles
The omics-profiles of the obserations are stores in the `so.X` attribute. These are the gene / protein expression values of the observations (most likely single cells or patches). You only need to provide this information if you want to use `metrics.quadratic_entropy()`.


```python
so.X[spl].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>target</th>
      <th>H3</th>
      <th>H3K28me3</th>
      <th>...</th>
      <th>PARP</th>
      <th>DNA2</th>
    </tr>
    <tr>
      <th>cell_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>8.252318</td>
      <td>0.442500</td>
      <td>...</td>
      <td>0.674024</td>
      <td>9.002035</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12.754854</td>
      <td>1.436547</td>
      <td>...</td>
      <td>0.852686</td>
      <td>19.567230</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.946717</td>
      <td>0.470739</td>
      <td>...</td>
      <td>0.858645</td>
      <td>2.316239</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.552009</td>
      <td>0.562183</td>
      <td>...</td>
      <td>0.939330</td>
      <td>8.991525</td>
    </tr>
    <tr>
      <th>5</th>
      <td>15.079700</td>
      <td>0.492224</td>
      <td>...</td>
      <td>0.828238</td>
      <td>12.647001</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 34 columns</p>
</div>



## Compute Graphs
ATHENA can construct `knn`, `radius` and `contact` graphs. By default, ATHENA tries to build the graphs from segmentation masks. Without segmentation masks, one can only use the `kNN` and `radius` graph functionality of ATHENA. For the `contact` graph the segmentation masks are required. See the [main tutorial](https://ai4scr.github.io/ATHENA/source/tutorial.html) to see how the graph construction can be customized.

### From Segmentation Masks
Provide the `spatialOmics` instance and sample name for which the graph should be computed. Furthermore, `builder_type` defines the type of graph that is constructed and `mask_key` the segmentation masks that should be used (stored at `so.masks[spl][MASK_KEY]`)


```python
ath.graph.build_graph(so, spl, builder_type='knn', mask_key='cellmasks')
so.G[spl].keys()  # graphs are stored at so.G[spl]
```




    dict_keys(['knn'])



One can build multiple graph-representations for each sample by simply calling `build_graph` again with another `builder_type`


```python
ath.graph.build_graph(so, spl, builder_type='radius', mask_key='cellmasks')
so.G[spl].keys()  # graphs are stored at so.G[spl]
```




    dict_keys(['knn', 'radius'])



### From Coordinates

One can build the `knn` and `radius` graphs from coordinates only by setting `mask_key=None` and providing the column names of the coordinates with `coordinate_keys`.


```python
ath.graph.build_graph(so, spl, builder_type='knn', mask_key=None, coordinate_keys=('x', 'y'))
ath.graph.build_graph(so, spl, builder_type='radius', mask_key=None, coordinate_keys=('x', 'y'))
so.G[spl].keys()  # graphs are stored at so.G[spl]
```




    dict_keys(['knn', 'radius'])



## Visualise the Data
For some of the plotting functionalities ATHENA requires the x,y coordinates of each observation. They can be extracted from segmentation masks with


```python
ath.pp.extract_centroids(so, spl, mask_key='cellmasks')
```

The data can then be visualised with 


```python
ath.pl.spatial(so, spl, attr='meta_id')
```

    /Users/art/Documents/projects/athena/athena/plotting/visualization.py:217: UserWarning: Matplotlib is currently using module://matplotlib_inline.backend_inline, which is a non-GUI backend, so cannot show the figure.
      fig.show()



    
![png](quickstart_files/quickstart_20_1.png)
    


or if one wants to use custom coordinates, one can provide coordinate keys that are stored in the `so.obs[spl]` dataframe ATHENA should use 


```python
ath.pl.spatial(so, spl, attr='meta_id', coordinate_keys=['x', 'y'])
```


    
![png](quickstart_files/quickstart_22_0.png)
    


See the [main tutorial](https://ai4scr.github.io/ATHENA/source/tutorial.html) and the [docs](https://ai4scr.github.io/ATHENA/api/athena.plotting.visualization.html?highlight=spatial#athena.plotting.visualization.spatial)  to see how the plotting can be customized.

## Compute Metrics
Once the graphs are built we can use the quantifications methods from the `ath.metrics` and the `ath.neigh` module. Again, provide the `spatialOmics` instance and the sample name for which the metric should be computed. Furthermore, since ATHENA quantifies the phenotypic heterogeneity in the data, provide the column name that indicates the different phenotypes of each observation with `attr` and specify the graph topologie to use with `graph_key`.

Most of the metrics can be computed either on a sample-level (i.e. 1 value per sample) or on a observation-level (i.e. 1 value per observation). This behaviour can be controlled by setting `local={True,False}`.


```python
# compute shannon entropy for each observation, once for the radius graph and once for the knn graph
ath.metrics.shannon(so, spl, attr='meta_id',graph_key='radius')
ath.metrics.shannon(so, spl, attr='meta_id',graph_key='knn')

# compute the shannon entropy for the whole sample
ath.metrics.shannon(so, spl, attr='meta_id', local=False)
```

The results for `local=True` (default) are stores in `so.obs` as `{method}_{attr}_{graph_key}`.


```python
so.obs[spl].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>core</th>
      <th>meta_id</th>
      <th>...</th>
      <th>shannon_meta_id_radius</th>
      <th>shannon_meta_id_knn</th>
    </tr>
    <tr>
      <th>cell_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>SP43_75_X2Y5</td>
      <td>9</td>
      <td>...</td>
      <td>1.921928</td>
      <td>1.918296</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SP43_75_X2Y5</td>
      <td>3</td>
      <td>...</td>
      <td>1.918296</td>
      <td>1.918296</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SP43_75_X2Y5</td>
      <td>10</td>
      <td>...</td>
      <td>2.446439</td>
      <td>2.521641</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SP43_75_X2Y5</td>
      <td>8</td>
      <td>...</td>
      <td>2.419382</td>
      <td>2.128085</td>
    </tr>
    <tr>
      <th>5</th>
      <td>SP43_75_X2Y5</td>
      <td>3</td>
      <td>...</td>
      <td>1.870254</td>
      <td>1.459148</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 14 columns</p>
</div>



The results for `local=False` are stored in `so.spl`


```python
so.spl.loc[spl].head()
```




    pid                    75.000000
    cell_count           2771.000000
    immune_cell_count     940.000000
    shannon_meta_id         3.311329
    Name: SP43_75_X2Y5, dtype: float64



## Note On Phenotype Encodings
We advise the users to use _numeric_ phenotype encodings when computing metrics. If one has label names stored in a `pd.DataFrame` as `label_names` one can easily create numeric labels by


```python
df = pd.DataFrame({'label_names': ['A', 'B', 'B', 'C']})
df['labels_names_id'] = df.groupby('label_names').ngroup()
```

Furthermore, some functions require a strict `categorical` encoding. Thus the columns `dtype` should be set to `categorical` and include all categories across the samples. This can be achived by running the following snippet for all `categorical` columns in `so.obs[spl]`


```python
# collect all occurences in the dataset
i = set()
for spl in so.spl.index:
    i.update(so.obs[spl]['meta_id'].values)

# define categorical dtype
dtype = pd.CategoricalDtype(i)
for spl in so.spl.index:
    so.obs[spl].loc[:,'meta_id'] = so.obs[spl]['meta_id'].astype(dtype)
```

## Note On Segmentation Masks
Segmentation masks should be stored as `np.ndarray` and have `int` encodings with 0 being background. The labels do not need to be sequential but aligned with the index in `so.obs[spl]`, i.e. the label of the segmentation mask should be the same as in the index of the `so.obs` dataframe for a given observation. One can test that the labels overlap


```python
ids = np.unique(so.masks[spl]['cellmasks'])
ids = ids[ids!=0]  # remove background label
s_mask = set(ids)
s_obs = set(so.obs[spl].index.values)
assert len(s_obs - s_mask) == 0
assert len(s_mask - s_obs) == 0
```
