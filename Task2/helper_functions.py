import numpy as np
import pandas as pd
import random 
from numpy.random import default_rng
from statsmodels.stats.outliers_influence import variance_inflation_factor



# We will use matplotlib to plot figures
import matplotlib.pyplot as plt


from sklearn import metrics


"""Generates uniformly sampled X for a given range with normally distributed noise"""
def generate_data(numofVar, minX, maxX, noiseMean, noiseSD, numofSamples, seed = None):
    
    if seed == None:
        rng = default_rng()
    else:
        rng = default_rng(seed)
    
    #generate noise variable 
    eps = rng.normal(noiseMean, noiseSD, numofSamples).T
    data = pd.DataFrame({ 'point': 'observed', 'eps': eps })
    
    #generate data for each covariate
    for i in range(0,numofVar):
        varname = 'x'+str(i+1)
        #x = rng.uniform(minX[i],maxX, numofSamples)
        data[varname] = rng.uniform(minX[i], maxX[i], numofSamples)    
        
    return data 

""""""
def calculate_y(data, numofVar, lineParams):
    y = data.eps
    
    for i in range(0, numofVar):
        x = 'x' + str(i+1) 
        y += data[x]*lineParams[i]

    return y 


""""""
def calculate_y_est(data, result):
    intercept, *a_est = result.params
    
    y_est = np.empty(len(data))
    y_est.fill(intercept)
    
    for i in range(0, len(a_est)):
        x = 'x' + str(i+1)
        y_est += data[x]*a_est[i]
    
    return y_est 

"""Add specific % of outlier points to original dataframe, 
specify min and max scale factor of the outlier and assign randomly 
"""
def add_outliers(data, perc_outliers, min_scale_factor, max_scale_factor) :
    OUTLIER_INDEX = round(len(data) - len(data)*(perc_outliers/100))
    data.loc[OUTLIER_INDEX: , 'point'] = 'Outlier'
    for i, row in data.iloc[OUTLIER_INDEX:].iterrows():
        scale = random.randrange(min_scale_factor, max_scale_factor*100 , 1)/100
        data.at[i,'y'] =  row['y']*scale
        data[data.point == 'Outlier'] 
    return data 

"""Scale a specific % of  data points to a randomly selected 
scale factor between the min and max scale factor
"""
def scale_datapoints_variable(data, perc_outliers, min_scale_factor, max_scale_factor):
    OUTLIER_INDEX = round(len(data) - len(data)*(perc_outliers/100))
    #print(OUTLIER_INDEX)
    data.loc[OUTLIER_INDEX: , 'point'] = 'Outlier'
    for i, row in data.iloc[OUTLIER_INDEX:].iterrows():
        scale = random.randrange(min_scale_factor, max_scale_factor, 1)#/100
        data.at[i,'y'] =  row['y']*scale
        data[data.point == 'Outlier'] 
    return data 


def scale_datapoints(data, perc_outliers, scale_factor):
    OUTLIER_INDEX = round(len(data) - len(data)*(perc_outliers/100))
    #print(OUTLIER_INDEX)
    data.loc[OUTLIER_INDEX: , 'point'] = 'Outlier'
    for i, row in data.iloc[OUTLIER_INDEX:].iterrows():
        #scale = random.randrange(min_scale_factor, max_scale_factor*100 , 1)/100
        data.at[i,'y'] =  row['y']*scale_factor
        data[data.point == 'Outlier'] 
    return data 

"""Output results from OLS model"""
def print_OLS_result(result):
    intercept, *a_est = result.params
    intercept_se, *a_est_se = result.bse 
    dp = 5
    print("\nOLS model parameters:")
    print("Intercept: {}, Intercept SE: {}".format(round(intercept, dp) , round(intercept_se, dp)))
    
    for i in range(0, len(a_est)):
        x = 'x' + str(i+1)
        print("{} coef: {}, {} coef SE: {}".format(x, round(a_est[i], dp), x, round(a_est_se[i], dp)))

        
def eval_ols_model(data):
    #MAE = metrics.mean_absolute_error(data.y, data.y_est)
    MSE = metrics.mean_squared_error(data.y, data.y_est)
    mean, sd = data['residuals'].describe()[['mean', 'std']]
    RMSE = np.sqrt(MSE)
    dp = 3
    print("\nModel Evaluation: ")
    print("Y-residual Mean: {}, Y-residual SD: {}, Root Mean Squared Error: {}".format(round(mean, dp), 
                                                                                            round(sd, dp), 
                                                                                         round(RMSE, dp)))
    return (mean, sd, RMSE)

def get_RMSE(y_actual, y_estimated):
    return  round(np.sqrt(metrics.mean_squared_error(y_actual, y_estimated)), 3)
    
"""Draws observed and estimated datapoints and y-residuals graph"""
def plot_scatter_yresiduals(data):
    #plot observations and estimates 
    fig = plt.figure(figsize=(16,5))                                  
    ax1 = fig.add_subplot(121) 
    plt.plot(data.x1, data.y, 'o', label = 'Observations')
    plt.plot(data[data.point == 'Outlier'].x1, data[data.point == 'Outlier'].y, 'ro', label='Outliers')
    plt.plot(data.x1, data.y_est, 'r+', label='Estimates')
    plt.plot(data.x1, data.y_est, 'r-', label='Regression line')
    plt.vlines(data.x1, data.y_est, data.y, linestyles='dashed', label = 'errors')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    
    ax2 = fig.add_subplot(122) 
    residuals_hist = plt.hist(data.residuals)
    plt.xlabel('Y-residual')
    plt.ylabel('Frequency')
    
    
    
#calculates VIF for each independent variable 
def calculate_vif(X):
    # Calculating VIF
   vif = pd.DataFrame()
   vif["variables"] = X.columns
   vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
   
   return(vif)