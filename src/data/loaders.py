import numpy as np  
import pandas as pd

#  stock data
def getStooq(symbol_txt="^dji",freq_txt="w", usecache_b=True, cache_dir="data/raw"):
    
    #symbol_txt is the string of the symbol eg '^dji' for dow jones
    #freq_txt is the string for frequency eg "d" for daily data, "w" weekly, "m" is monthly, "y" = yearly
    
    # --------- import the stooq data into pandas
    import os.path 
    #fname1 = 'c:\\PythonCode\\' # define the path to check
    fname1 = cache_dir+"//"  # don't need to define path if for default dir
    fname2 = symbol_txt + "_" + freq_txt + '.csv'
    
    if (os.path.isfile(fname1+fname2) and (usecache_b==True)):
        print('loading local file')
        #stockdata = pd.read_csv('c:\\PythonCode\\^dji_w.csv')
        stockdata = pd.read_csv(fname1+fname2)
            
    else:
        print('download stooq file')
        #urlbase="https://stooq.com/q/d/l/?s=^dji&i="   # base url
        urlbase="https://stooq.com/q/d/l/?s=" #"^dji&i="   # base url
        urlfull = urlbase+symbol_txt + "&i=" + freq_txt  # append the d or w to the url for pandas
        stockdata = pd.read_csv(urlfull)  # ask pandas to create a dataframe form the stooq DJI data
        #stockdata = pd.read_csv(urlfull,parse_dates=['Date'])  # ask pandas to create a dataframe form the stooq DJI data

        if True:
            import requests
            response = requests.get(urlfull)
            #"w" is ascii mode "wb is write in binary mode
            
            with open(fname1+fname2, "w") as f:
                f.write(response.text)
    
    stockdata.Date=stockdata.Date.apply(pd.to_datetime)  # set th index to the date column

    return stockdata  