# resampled
A class based on Pandas and NumPy to perform bootstrap or permutation statistical hypothesis testing. In the case of bootstrapped samples the confidence interval 
is also computed. The statistics calculated are not limited, they can be user defined (e.g. a specific percentile) or exist in numpy (e.g. `np.mean`)

The class was created out of the need to analyze typically small data (a few 10s of records) that belong to different categories or discrete levels (see example 
notebook for details). The output will typically include the resampled dataframe as an educational visual of what the null hypothesis is about. 

