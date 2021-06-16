import numpy as np
import pandas as pd
from sys import exit

class Resample():
    """
        There are 2 methods in the class:
        a) run hypothesis: calculates the pvalue of a hypothesis between 2 levels of the variable under test: e.g. 
        drug dosage A and B where A, B are the levels
        b) estimate_ci (for bootstrap only): estimates the c.i. of the statistic at chosen alpha-level, of a variable at a particular level: 
        e.g. drug dosage at level A

    Example dataset:
        index    var1   var2    var3   levels
        0         4      10.2    0          A
        1         3       9.8    1          A
        2         7      11.0    1          B
        3         5      12.2    0          B
        4    ...

        """

    def __init__(self, resampling_method):
        """
        Parameters
        ----------
        resampling method : str
             Initiates the class with the hypothesis testing is targeted. Accepted values are
             'bootstrap' and 'permutation'.
        """
        if resampling_method != 'bootstrap' and resampling_method != 'permutation':
            exit("Please set resampling method as 'bootstrap' or 'permutation' ")
        self.method = resampling_method

    def run_hypothesis(self, df, target, levels, lvl1, lvl2, R, func):
        """
        Parameters
        ----------
        df : PANDAS dataframe
            Entire dataset.
        target : str
            Feature/column name that contains the data 
        levels : str
            Feature/column name describing the categories in which the 
            measurements are divided.            
        lvl1 : str/int/float
            One of the levels/categories of interest inside the condition_feature.
        lvl2 : str/in/float
            The second of the levels/categories of interest which is compared to 1.
        R : int
            number of bootstrap or permutation samples to generate.
        func : any user coded or numpy/scipy or module imported function that calculates the statistic of interest.
            The function can also be defined externally. e.g. 
            def my_func(x):
                return np.percentile(x,q=33.33)
            run_hypothesis(df,'target_feature','levels_feature',level_1,level_2,R=10000,func=my_func)
        OR:
            run_hypothesis(df,'target_feature','levels_feature',level_1,level_2,R=10000,func=np.mean)

        Returns
        -------
         resampled_: PANDAS dataframe
            DataFrame with first column being the 'levels' feature and the next R columns being the bootstrapped/permutated
            resampled samples. This is a visual of how the resampling method redistributes the target
            values among the lvl1 and lvl2.Useful only for educational purposes to demonstrate the null hypothesis.
         resampled_diff: PANDAS Series
            Difference of bootstrapped/permutated statistic between the 2 levels selected.            
        pval: numpy.float64
            p-value of the statistical test.
        """
        # reduce the dataframe to the features of interest
        df = df[[levels, target]]
        df = df[(df[levels] == lvl1) | (df[levels] == lvl2)]
        df = df.reset_index()

        x = df[df[levels] == lvl1][target]
        y = df[df[levels] == lvl2][target]
        # observed difference of statistic
        obs_diff_statistic = func(x)-func(y)

        resampled_ = df[[levels]]  # new dataframe to dump the resampled values
        # Resampling block
        if self.method == 'bootstrap':
            for i in range(0, R):
                b = df[target].sample(n=df.shape[0], replace=True)
                b = b.reset_index(drop=True)
                resampled_ = pd.concat(
                    [resampled_, b], axis=1, ignore_index=True)
        elif self.method == 'permutation':
            for i in range(0, R):
                b = df[target].sample(n=df.shape[0], replace=False)
                b = b.reset_index(drop=True)
                resampled_ = pd.concat(
                    [resampled_, b], axis=1, ignore_index=True)
        else:
            exit("Please set resampling method as 'bootstrap' or 'permutation' ")

        # calculate resampled statistic
        resampled1 = resampled_[resampled_.loc[:, 0] == lvl1]
        resampled2 = resampled_[resampled_.loc[:, 0] == lvl2]
        f_resampled1 = resampled1.loc[:, 1:R].apply(func, axis=0)
        f_resampled2 = resampled2.loc[:, 1:R].apply(func, axis=0)

        resampled_diff = f_resampled1-f_resampled2

        # calculate p-value
        pval = np.sum((resampled_diff-obs_diff_statistic) >= 0)/R

        resampled_.rename(columns={0: levels}, inplace=True)

        return resampled_, resampled_diff, pval

    def estimate_ci(self, df, target, levels, lvl, R, func, alpha_level=0.05):
        """
        Parameters
        ----------
        df : Pandas DataFrame
            Dataframe contaning the data.
        level : str/int/float
            One of the levels/categories of interest inside the levels feature.
        R : int
            Number of bootstrap samples to be generated.
        func : any user coded or numpy/scipy or module imported function that calculates the statistic of interest.
            The function can also be defined externally. e.g. 
            def my_func(x):
                return np.percentile(x,q=33.33)
            run_hypothesis(df,'target_feature','levels_feature',level_1,level_2,R=10000,func=my_func)
        OR:
            run_hypothesis(df,'target_feature','levels_feature',level_1,level_2,R=10000,func=np.mean)

        alpha_level : int
            Percentage (0,100)  for statistical confidence interval. The default is 5.

        Returns
        -------
        bootstrapped_stat: PANDAS Series
            R statisticts estimated from each bootstrapped sample. Returned explicitly for post processing,
            e.g. plot and confirm that central limit theorem kicked in.
        ci: tuple
            confidence interval at the specified alpha level, upper and lower limits.

        """
        if self.method == 'bootstrap':

            # reduce the dataframe to the features of interest at the level of interest
            df = df[[levels, target]]
            df = df[(df[levels] == lvl)]
            df = df.reset_index()

            # new dataframe to hold bootstrapped samples
            df_ = df.drop(columns=[target])

            # bootstrapping block
            for i in range(0, R):
                b = df[target].sample(n=df.shape[0], replace=True)
                b = b.reset_index(drop=True)
                df_ = pd.concat([df_, b], axis=1, ignore_index=True)
        else:
            exit('confidence interval can be estimated only for bootstrap method')

        # Series holding the per column bootstrapped statistic
        bootstrapped_stat = df_.loc[:, 1:R].apply(func, axis=0)

        # tuple holding the confidence interval
        ci = (np.percentile(bootstrapped_stat, q=(100-0.5*100*alpha_level)),
              np.percentile(bootstrapped_stat, q=0.5*100*alpha_level))

        return bootstrapped_stat, ci
