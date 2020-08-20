import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sys import exit

class permutation():
    """
    Permutation assumes:
	1. Sample has been drawn from a normal distribution but is small (n<30)
    2. assumptions for parametric approach are not met.
    3. test something other than classic approaches comparing means and medians, 
    e.g. 10th percentiles or range of weights.
    4. difficult to estimate SE for test statistic
    
	There are 2 methods in the class:
	a) calculate_pvalue: calculates the pvalue of a hypothesis between 2 levels of the variable under test: e.g. 
	drug dosage A and B where A, B are the levels
	"""
    def __init__(self,target_feature,condition_feature,condition1,condition2):
        """
        Parameters
        ----------
        condition_feature : str
            Feature/column name describing the categories in which the 
            measurements are divided.
        target_feature : str
            Feature/column name that contains the data             
        condition1 : str/int/float
            One of the levels/categories of interest inside the condition_feature.
        condition2 : str/in/float
            The second of the levels/categories of interest which is compared to 1.
        """
        self.levels=condition_feature
        self.target=target_feature
        self.lvl1=condition1
        self.lvl2=condition2
  
    def calculate_pvalue(self,df,P,func):       
        """
        Parameters
        ----------
        df : PANDAS dataframe
            Entire dataset.
        P : int
            number of permutation samples to generate.
        func : numpy/scipy function
            Any numpy or scipy stat function supported. If the the np, sp function takes many arguments,
            the function can be defined externally. e.g. 
            def my_func(x):
                return np.percentile(x,q=33.33)
            calculate_pvalue(df,P=10000,func=my_func)
        Returns
        -------
        boot_mean_diff : PANDAS Series
            Permutated sample to permutated sample mean difference for the chosen levels.

        """
        #reduce the dataframe to the features of interest
        df=df[[self.levels,self.target]]
        df=df[(df[self.levels]==self.lvl1)|(df[self.levels]==self.lvl2)]
        df=df.reset_index()
        
        x=df[df[self.levels]==self.lvl1][self.target]
        y=df[df[self.levels]==self.lvl2][self.target]
        obs_diff_statistic=func(x)-func(y)
            
        print("observed difference of statistic: ", obs_diff_statistic)

        df_=df[[self.levels]]        
        for i in range(0,P):
            b=df[self.target].sample(n=df.shape[0],replace=False)
            b=b.reset_index(drop=True)
            df_=pd.concat([df_,b],axis=1,ignore_index=True)
        
        perm1=df_[df_.loc[:,0]==self.lvl1]
        perm2=df_[df_.loc[:,0]==self.lvl2]
        
        m_perm1=perm1.loc[:,1:P].apply(func,axis=0)
        m_perm2=perm2.loc[:,1:P].apply(func,axis=0)
        
        perm_m_diff=m_perm1-m_perm2
        
        pval=np.sum((perm_m_diff-obs_diff_statistic)>=0)/P
        print("p-value",pval)
        
        
        yp=np.abs(len(perm_m_diff[perm_m_diff>np.percentile(perm_m_diff,45)])-
                  len(perm_m_diff[perm_m_diff>np.percentile(perm_m_diff,55)]))
        
        plt.figure()
        sns.distplot(perm_m_diff,kde=False)
        plt.xlabel("Difference of statistic of bootstrapped samples")
        plt.plot([obs_diff_statistic,obs_diff_statistic],[0,yp],'--r',label='Obs. Diff. of statistic')
        plt.legend()
        
        return perm_m_diff
