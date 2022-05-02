from math import factorial
import pandas as pd
import re
import numpy as np
from os import listdir
import random as rand




def get_mispelling(filePath) -> dict:
    """
    Return the mispelling info into a dictionary
    with the format {'well_formed_word',[..,'i-th mispelled version',..]}
    """

    with open(filePath, 'r') as h:
        sub = h.readlines()

    re_pattern = r'\$'
    regex = re.compile(re_pattern)
    # Get start times
    good_word = list(filter(regex.search, sub))
    good_word = [word.split('\n')[0].replace('$','') for word in good_word]
    
    # Get lines
    mispelled = [[]]
    for line in sub:
        if re.match(re_pattern, line): #create empty list for upcoming mispelled words
            mispelled.append([])   
        else: #mispelled word referred to the last word that followed the symbol '$' 
            mispelled[-1].append(line.split('\n')[0])  #append mispelled word to the last well formed one
                          
    mispelled = mispelled[1:]  #remove title of file
    # Merge results
    misp = {good_w:misp for good_w,misp in zip(good_word, mispelled)}
    return misp  

def loadMispellings(filePath) -> pd.DataFrame:
    """
    Load the mispelling dataset returning the number
    of recurrent mispellings for each word in the dataset. 
    """
    misp_dict=get_mispelling(filePath)

    df=pd.DataFrame.from_dict(misp_dict,orient='index')
    df=df.count(axis=1)         #.sort_values(ascending=False)    

    df.reset_index()
    df=pd.DataFrame(df,columns=['NUM_MISPELLINGS'])

    return df



def count_el(syns) -> int:
    """
    Counts the number of synonyms for the given entry of the dictionary 
    {'well_formed_word',[..,'i-th mispelled version',..]}
    """
    return 0 if syns == '-' else len(re.split(';',syns.replace('|',';')))

def loadSynonyms(filePath) -> pd.DataFrame:
    """
    Load the synonyms dataset returning the number
    of synonyms for each word in the dataset. Nouns without 
    synonyms have counter 0.
    """
    startDB=274
    df=pd.read_csv(filePath,header=0)
    df=df.drop(columns=['part_of_speech'])
    df=df.iloc[startDB:]
    
    df=df.reset_index(drop=True)

    df=df.fillna('-')
    df['synonyms']=df['synonyms'].apply(lambda x: count_el(x))
    df=df.rename({'lemma':'WORD','synonyms':'NUM_SYNONYMS'},axis=1)

    df=df.set_index(['WORD'])

    return df



def loadSubtitles(filePath) -> dict:
    """
    Return a dict {timestamp :  line} for the subtitles 
    of the given episode of friends

    args:
    filePath: episode of friend to get subtitles from
    """

    with open(filePath, 'r') as h:
        sub = h.readlines()

    re_pattern = r'[0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]{3} -->'
    regex = re.compile(re_pattern)
    # Get start times
    start_times = list(filter(regex.search, sub))
    start_times = [time.split(' ')[0] for time in start_times]
    # Get lines
    lines = [[]]
    
    for sentence in sub:
        if re.match(re_pattern, sentence):
            lines[-1].pop()
            lines.append([])
        else:
            lines[-1].append(sentence)

    lines = lines[1:]         

    # Merge results
    subs = {start_time:line for start_time,line in zip(start_times, lines)}

    return subs


#BIT ERROR RATE (BER)
#BER1..BER5 where BER1 being the lowest noise condition.

def getPoisson(k,l) -> np.float64:
    """
    Returns the Poisson distribution with parameters k and lambda
    that respectively represent the number of occurrences of an event
    we want to estimate and the prior knowledge on the average number
    of events that usually occurs. 

    In our case it is helpful estimating the number of data corruption
    due to noise that can occur along the path. 
    """
    poiss=(l**k * np.e**(-l))/factorial(k)
    return poiss


def getRouterRelation():
    """
    Returns the probability distribution of getting into
    a noise level (BER index) depending on its last value.

    The format of the result is a numpy arrya 5 X 5:
    [
                     BER1_t .. BER5_t
        BER1_{t+1} |    
        ... 
        BER5_{t+1} | 
    ]
    """

    k=np.array([[8,9,10,12,14],
                [9,7,7,8,8],
                [7,6,3,4,6],
                [5,4,3,2,2],
                [3,2,2,1,1]])
    ll=[15,10,8,6,4]

    res=np.zeros((5,5))

    #compute Poisson distribution
    for i in range(5):
        for j in range(5):
            res[i,j]=getPoisson(k[i,j],ll[i])

    #orthogonalize columns (sum = 1)
    for j in range(5):
        ort=sum(res[:,j])
        for i in range(5):
            res[i,j]=res[i,j]/ort

    return res

def getRandomEpisode():
    """
    Return the paths to a random episode
    in both english and italian version.
    """

    en_path='friends/Friends - season 1.en/'
    it_path='friends/Friends - season 1.it/'
    en_subs = sorted([en_path + f for f in listdir(en_path)])
    it_subs = sorted([it_path + f for f in listdir(it_path)])

    en=rand.choice(en_subs)
    it=rand.choice(it_subs)

    return en,it


def getBitErrorRate() :
    """
    Returns the probabilistic relations between 
    BER and the noise measures: EbNO, C/I and Phi.

                    EbNO(0)         ...        
                    C/I(0)          ...
                Phi(0) Phi(1)   ...
    BER_0 (0)
       ...
    """

    distr_eb_0=np.array([   [0.96,0.04,0,0,0],
                            [0.935,0.06,0.005,0,0],
                            [0.96,0.04,0,0,0],
                            [0.915,0.06,0.025,0,0]]).T

    distr_eb_1=np.array([   [0.375,0.03,0.015,0.135,0.445],
                            [0,0,0.025,0.355,0.62],
                            [0.96,0.035,0.005,0,0],
                            [0.725,0.225,0.05,0,0]]).T

    return np.concatenate((distr_eb_0,distr_eb_1),axis=1).tolist()

    

print(getRouterRelation())