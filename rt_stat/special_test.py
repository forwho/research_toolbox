import statsmodels.formula.api as smf
import statsmodels.api as sm
import pandas as pd
import numpy as np

def linear_or_quadratic(xval,yval):
    data=pd.DataFrame({'xval':xval,'yval':yval, 'xval2':xval*xval})
    linear_model=smf.ols(formula='yval~xval',data=data).fit()
    quadratic_model=smf.ols(formula='yval~xval+xval2',data=data).fit()
    print(linear_model.aic,quadratic_model.aic)
    print(linear_model.summary())
    print(quadratic_model.summary())
