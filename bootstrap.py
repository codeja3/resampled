import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sys import exit

class Bootstrap():
    """
    Bootstrapping assumes:
	1. Sample has been drawn from a normal distribution but is small (n<30)
	2. Homogeneity of variance: the populations from which samples of the variable were drawn have similar Var.
	3. Data point values should be independent from each other.
	4. Interval data
	
	There are 2 methods in the class:
	a) calculate_pvalue: calculates the pvalue of a hypothesis between 2 levels of the variable under test: e.g. 
	drug dosage A and B where A, B are the levels
	b) estimate_ci: estimates the c.i. of the statistic at chosen alpha-level, of a variable at a particular level: 
	e.g. drug dosage at level A
    
    Example dataset:
        index    var1   var2    var3   level_feature
        0         4      10.2    0          A
        1         3       9.8    1          A
        2         7      11.0    1          B
        3         5      12.2    0          B
        4    ...
    
	"""
    def __init__(self,target_feature,levels_feature,level1,level2):
        """
        Parameters
        ----------
        target_feature : str
            Feature/column name that contains the data 
        levels_feature : str
            Feature/column name describing the categories in which the 
            measurements are divided.            
        level1 : str/int/float
            One of the levels/categories of interest inside the condition_feature.
        level2 : str/in/float
            The second of the levels/categories of interest which is compared to 1.
        """
        self.levels=levels_feature
        self.target=target_feature
        self.lvl1=level1
        self.lvl2=level2
  
    def calculate_pvalue(self,df,B,func):       
        """
        Parameters
        ----------
        df : PANDAS dataframe
            Entire dataset.
        B : int
            number of bootstrap samples to generate.
        func : any user coded or numpy/scipy or module imported function
            the function can also be defined externally. e.g. 
            def my_func(x):
                return np.percentile(x,q=33.33)
            calculate_pvalue(df,P=10000,func=my_func)
        OR simply call:
            calculate_pvalue(df,P=10000,func=np.mean)

        Returns
        -------
        boot_mean_diff : PANDAS Series
            Bootstrapped sample to bootstapped sample mean difference for the chosen levels.

        """
        #reduce the dataframe to the features of interest
        df=df[[self.levels,self.target]]
        df=df[(df[self.levels]==self.lvl1)|(df[self.levels]==self.lvl2)]
        df=df.reset_index()
        
        x=df[df[self.levels]==self.lvl1][self.target]
        y=df[df[self.levels]==self.lvl2][self.target]
        obs_diff_statistic=func(x)-func(y)

        print("observed difference of statistic: ",obs_diff_statistic)

        df_=df[[self.levels]]        
        for i in range(0,B):
            b=df[self.target].sample(n=df.shape[0],replace=True)
            b=b.reset_index(drop=True)
            df_=pd.concat([df_,b],axis=1,ignore_index=True)
        
        bootstrapped1=df_[df_.loc[:,0]==self.lvl1]
        bootstrapped2=df_[df_.loc[:,0]==self.lvl2]
        m_bootstrapped1=bootstrapped1.loc[:,1:B].apply(func,axis=0)
        m_bootstrapped2=bootstrapped2.loc[:,1:B].apply(func,axis=0)
        
        boot_m_diff=m_bootstrapped1-m_bootstrapped2
        
        pval=np.sum((boot_m_diff-obs_diff_statistic)>=0)/B
        print("p-value",pval)
        
        #estimate the y-point at which the cutoff will be plotted
        yp=np.abs(len(boot_m_diff[boot_m_diff>np.percentile(boot_m_diff,45)])-
                  len(boot_m_diff[boot_m_diff>np.percentile(boot_m_diff,55)]))
        
        
        plt.figure()
        sns.distplot(boot_m_diff,kde=False)
        plt.plot([obs_diff_statistic,obs_diff_statistic],[0,yp],'--r',label='Obs. Diff. of statistic')
        plt.xlabel("Difference of statistic of bootstrapped samples")
        plt.legend()
        
        return boot_m_diff

    def estimate_ci(self,df,level,B,func,alpha_level=0.05,create_dummy_var=False):
        """
        Parameters
        ----------
        df : Pandas DataFrame
            Dataframe contaning the data.
        level : str/int/float
            One of the levels/categories of interest inside the condition_feature.
        B : int
            Number of bootstrap samples to be generated.
        statistic : 'mean' or 'median'
            statistics supported.
        alpha_level : int
            Percentage (0,100)  for statistical confidence interval. The default is 5.
        create_dummy_var: Boolean
            if True it creates a variable called "dummy" in the dataset at a single level of 0.
            This is used for small datasets of only a single level to make the method usable for 
            extracting c.i. estimates. 
            e.g. dataset type
            Index  Var1   Var2  dummy
            0      10.4     3     0
            1      11.2     4     0
        Returns
        -------
        bootstrapped_means: PANDAS Series
            B means estimated from each bootstrapped sample
        ci: tuple
            confidence interval at the specified percentage, upper and lower limits.

        """
        if create_dummy_var:
            df['dummy']==0
        #reduce the dataframe to the features of interest at the level of interest
        df=df[[self.levels,self.target]]
        df=df[(df[self.levels]==level)]
        df=df.reset_index()
        
        df_=df.drop(columns=[self.target])
        
        for i in range(0,B):
            b=df[self.target].sample(n=df.shape[0],replace=True)
            b=b.reset_index(drop=True)
            df_=pd.concat([df_,b],axis=1,ignore_index=True)
        
        bootstrapped_m=df_.loc[:,1:B].apply(func,axis=0)
        
        ci= (np.percentile(bootstrapped_m,q=(100-0.5*100*alpha_level)),np.percentile(bootstrapped_m, q=0.5*100*alpha_level))
    
        return bootstrapped_m, ci
