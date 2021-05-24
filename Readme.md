# About the file:
This is an ongoing competetion on [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) 

# Approach:

The require libraries are 
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as snb
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from feature_engine.missing_data_imputers import MeanMedianImputer, CategoricalVariableImputer, AddMissingIndicator
from feature_engine.categorical_encoders import OneHotCategoricalEncoder, RareLabelCategoricalEncoder,OrdinalCategoricalEncoder
from feature_engine.outlier_removers import OutlierTrimmer
from feature_engine import variable_transformers
from feature_engine.discretisers import EqualWidthDiscretiser
import seaborn as sns
sns.set_style('whitegrid')
import warnings  
warnings.filterwarnings('ignore')
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
```

## Data Analysis:
1. Skewness Co-efficient is calculated
2. Get the dataframe with all the features and SUM of the NaN values present
3. Select only those features who have atleast 1 NaN value
4. Change the SUM to PERCENTAGE 
5. Use multiple kinds of encoding as shown in the code to cahnge the data for better understanding ( For more presice direction make use of comments)


## Modeling

Craete a stacked model with Lasso, Elastinet, Gradientboost, Xgboost, LightGBM
The reported RMSE score is 0.00777



## Acheived rank: 181 at the time of submissiion