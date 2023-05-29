#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def predict_residuos(date):
    import pandas as pd
    import numpy as np
    import pmdarima as pm
    import warnings
    from  statsmodels.tsa.arima.model import ARIMA
    from datetime import datetime
    

    warnings.filterwarnings('ignore')
    path="https://raw.githubusercontent.com/mwolinsky/Residuos_Solidos_Data/main/Data%20SORFRS%20A%C3%B1o%20y%20mes.csv"

    df=pd.read_csv(path,encoding="latin1",sep=";",decimal=",")
    df["Mes"]=df.index
    df=df[0:12]
    
    df["Mes"]=range(1,13)
    df_melted=pd.melt(df, id_vars='Mes',value_vars=["2003","2004","2005","2006",
                                                "2007","2008","2009","2010",
                                                "2011","2012","2013","2014",
                                                "2015","2016","2017","2018","2019","2020"] )
    
    
    df_melted["Anio_Mes"]=df_melted.variable.astype("string")+"/"+df_melted.Mes.astype("string")
    df_melted["Anio_Mes"]=pd.to_datetime(df_melted["Anio_Mes"], format='%Y/%m')
    df=df_melted[["Anio_Mes","value"]]
    
    df.index = df['Anio_Mes']
    del df['Anio_Mes']
    
    df['Date'] = df.index
    train = df[df['Date'] < pd.to_datetime("2020-01", format='%Y-%m')]
    train['train'] = train['value']
    del train['Date']
    del train['value']
    test = df[df['Date'] >= pd.to_datetime("2020-01", format='%Y-%m')]
    del test['Date']
    test['test'] = test['value']
    del test['value']


    #ARIMA(0,1,2)
    arima_model_2=ARIMA(train,order=(0,1,2),trend="t")
    model_2=arima_model_2.fit()

    
    date=datetime.strptime(date, "%Y/%m")
    prediction=round(model_2.predict(date)[-1],4)
    
    return(prediction)

    
    


# %%





