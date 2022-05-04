import random as rand
import pandas as pd
from os import listdir
import pickle
from numpy import average, exp, sqrt, concatenate, reshape

from .utils import loadSynonyms, loadMispellings, loadSubtitles
from .score import Scorer
from sklearn.preprocessing import  MinMaxScaler
from itertools import product


class Model():
    """
    Loader of all datasets available. By means of the Score class
    the probability of occurrences of each lines are computed and
    made accessible by specific entry methods.

    In order to compute the mentioned score, the user must specify
    the weights associated with the following parameters associated
    to each word in the dataset about their probability of being
    affected by noise:
    >> len_w            : weights on the word's length;
    >> sys_w            : weights on the number of synonyms;
    >> misp_w           : weights on the number of possible mispellings;
    >> smoothing_fact   : smoothing factor that allows computing the score for words that are not in the dataset;

    The main data source are the english and italian subtitles from Friends.
    It is therefore possible to load more data in the format {'TIMESTAMP':'line'} but,
    in such a case, recall to recompute the scoring dataset.

    As the Model constructor is called, the user is able to specify whether
    he/she wants to compute scores from scratch, i.e. by loading custom datasets (computeScores=True),
    or if the pre-computed scores from the folder 'data/' will be used (computeScores=False).
    """

    def __init__(self,len_w=0.5, syn_w=1.2, misp_w=0.5, smoothing_fact=1.5,\
                                     computeScores=True, en_episode=None, it_episode=None) -> None:
        self.__synonymsDF = None
        self.__myspellingDF = None
        
        self.subs_df_en=None #timestamp, line
        self.subs_df_it=None #timestamp, line
        
        self.stat_df=None #num_syn, num_misp
        self.prob_df=None #int_line (query_map), prob.of.success
        self.switchScore_df=None #int_line1, int_line2, switchScore
        
        #prob of switch according to the previous value and noise level
        self.noisySwitchCPD_surv=None 
        self.noisySwitchCPD_hard=None
        self.noisySwitchCPD_flat=None 

        self.__len_w=len_w
        self.__syn_w=syn_w
        self.__misp_w=misp_w
        self.__smooth_f=smoothing_fact

        self.query_map={} #association 'id' <-> 'line'  (necessary for quering)

        #load datasets and scores from the /data folders
        #subtitles do not need to be loaded
        if not computeScores:
            #load mapping
            with open('data/IDlines.pkl', 'rb') as f:
                self.query_map = pickle.load(f)

            #load statistics
            self.getStatistics('data/mispellings.csv','data/synonyms.csv')
         
            #load scores            
            self.prob_df=pd.read_csv('data/StatScores.csv')            

            self.switchScore_df=pd.read_csv('data/SwitchingScores.csv')
            
            #load noisy switch CPD
            with open('data/switchProbs_surv.pkl', 'rb') as f:
                self.noisySwitchCPD_surv = pickle.load(f) #tuple (DEC_0_CPD, DEC_0+k_CPD)
            # V2
            # with open('data/switchProbs_surv_v2.pkl', 'rb') as f:
            #     self.noisySwitchCPD_surv = pickle.load(f) #tuple (DEC_0_CPD, DEC_0+k_CPD)

            with open('data/switchProbs_hard.pkl', 'rb') as f:
               self.noisySwitchCPD_hard = pickle.load(f) 
        else:
            #Compute model based on one episode only
            self.getStatistics('data/mispellings.csv','data/synonyms.csv')            
            self.loadEpisodeSubtitles(en_episode,it_episode)
            self.getProbabilities()
            self.getBatchSwitchScores(100)


    def getStatistics(self,mispellingPath,synonymsPath) -> pd.DataFrame:
        """
        Load both synonyms and mispelling dataset from the respective
        paths given as input and return a new merged dataset with the 
        following columns:
        ['WORD','NUM_SYNONYMS','NUM_MISPELLINGS']
        """
        s_df=loadSynonyms(synonymsPath)
        m_df=loadMispellings(mispellingPath)
        
        # eliminate duplicates
        s_df=s_df.loc[~s_df.index.duplicated(keep='first')] 
        m_df=m_df.loc[~m_df.index.duplicated(keep='first')]

        #beware of how the order of concat handle NaNs
        df=pd.concat([s_df,m_df],axis=1)
        df=df.fillna(float(0))

        self.stat_df=df
        self.__synonymsDF=s_df
        self.__myspellingDF=m_df


        return df

    def __loadAllSubtitles(self) -> None:
        """
        Load all the italian and english subtitles from the
        default directories (friends/).
        """
        en_path='friends/Friends - season 1.en/'
        it_path='friends/Friends - season 1.it/'
        en_subs = sorted([en_path + f for f in listdir(en_path)])
        it_subs = sorted([it_path + f for f in listdir(it_path)])

        # >> load english subtitles
        for en_file in en_subs:
            self.getSubtitles_en(en_file)

        # >> load italian subtitles
        for it_file in it_subs:
            self.getSubtitles_it(it_file)
        
        return None

    def loadEpisodeSubtitles(self,episode_en,episode_it) -> None:
        """
        Load english and italian subtitles from the
        episode with the title given as argument.
        """
        self.getSubtitles_en(episode_en)
        self.getSubtitles_it(episode_it)
        
        return None        

    def getSubtitles_en(self,dfPath) -> pd.DataFrame:
        """
        Loads into a dataset the english subtitles 
        from the given path. 
        
        It can be recursively called in order to append new
        data to the (eventually) pre-existent dataset.
        """
        df_dict=loadSubtitles(dfPath)

        df=pd.DataFrame.from_dict(df_dict,orient='index')
        df=df.fillna('').sum(1).apply(lambda x : x.replace('\n',' ')).str.lower() #shrink columns all in one and remove \n
        df=pd.DataFrame(df,columns=['LINES'])

        if self.subs_df_en is not None:        #already some data loaded
            self.subs_df_en=pd.concat([self.subs_df_en,df],ignore_index=True,axis=0,levels=None)
        else:                   #brand new dataset (firsts subs loaded)
            self.subs_df_en=df.reset_index(drop=False)       
        
        self.subs_df_en=self.subs_df_en.rename(columns={'index':'TIMESTAMPS'})      

        return self.subs_df_en

    def getSubtitles_it(self,dfPath) -> pd.DataFrame:
        """
        Loads into a dataset the italian subtitles
        from the given path. 
        
        It can be recursively called in order to append new
        data to the (eventually) pre-existent dataset.
        """
        df_dict=loadSubtitles(dfPath)

        df=pd.DataFrame.from_dict(df_dict,orient='index')
        df=df.fillna('').sum(1).apply(lambda x : x.replace('\n',' ')).str.lower() #shrink columns all in one and remove \n
        df=pd.DataFrame(df,columns=['LINES'])

        if self.subs_df_it is not None:        #already some data loaded
            self.subs_df_it=pd.concat([self.subs_df_it,df],ignore_index=True,axis=0,levels=None)
        else:                   #brand new dataset (firsts subs loaded)
            self.subs_df_it=df.reset_index(drop=False)       
        
        self.subs_df_it=self.subs_df_it.rename(columns={'index':'TIMESTAMPS'})      

        return self.subs_df_it

    #-------------SCORING FUNCTIONS----------------------
    def getProbabilities(self):
        """
        Build the probabilities DB that will be used as CPD.
        For each sentence of the 'subs' dataset compute the
        score according to the word that compone each entry.

        The result should be returned in a compatible format
        for the CPD constructor. 

        Returns a dataset with the format 
        ['LINES_ID', 'PROB. OF SUCCESS', 'PROB. OF FAILURE']      
        where the lines' id are those mapped with lines into
        the dictionary query_map.
        """

        df=self.subs_df_en
        scorer=Scorer(self.__len_w, self.__syn_w, self.__misp_w, self.__smooth_f,self.stat_df,self.subs_df_en,self.subs_df_it)

        #'line'<-->'score'
        prob={}

        #same lines will be suppressed - not a big deal since there's no reasoning upon the number
        #of occurrences of a line. Namely if the same line appears once or twelve times it does not
        #affect the score.
        for line in df['LINES']:  
            prob[line]=scorer.computeScore_line(line,2) #1 OR 2 --> score function

        #self.__query_map[line]=prob --> normalize score --> transform it into probability
        # 1. Mapping
        num_lines=len(prob)
        self.query_map=dict(zip(range(num_lines),[str.lower(s) for s in prob.keys()]))
        
        # 2. Normalization
        df_norm=pd.DataFrame.from_dict(prob,orient='index',columns=['SCORE'])   
        normalizer=MinMaxScaler(feature_range=(0.001,1)) #avoid min=0 otherwise some score would then lead to a 100% prob. of failure
        df_norm['SCORE'] = normalizer.fit_transform(df_norm[['SCORE']])
        
        # add new column (negative probability)
        neg_vals=[1-x for x in df_norm['SCORE']]
        df_norm.insert(1,'NEG_SCORE',neg_vals)

        #change line into num from map
        #3. probability of succes p= 1 - score
        df_prob=df_norm
        df_prob.rename(columns={'SCORE':'PROB. OF SUCCESS', 'NEG_SCORE':'PROB. OF FAILURE'},inplace=True)
        df_prob.insert(0,'LINES_ID',self.query_map.keys())
        df_prob.set_index(['LINES_ID'],drop=True,inplace=True)

        self.prob_df=df_prob

        return df_prob

    def getSwitchScores_lines(self) -> pd.DataFrame:
        """
        Computes the switching score, i.e., the probability
        of each line to swatch into another line. Due to noise,
        mispelling and translation (into italian) lines can indeed
        switch: 
                P(Source line X --> Y Destination line)

        The score is computed according the following parameters:
            >> number of (italian) words the lines shares 
            >> probability score of the source line (from df_stat)

        A dataframe is returned with columns {'LINE1', 'LINE2', 'SWITCH_PROB'}
        where the LINES_ elements are the identifier from the mapping dictionary.
        """

        #build df and Scorer
        df=pd.DataFrame(columns=['LINE1', 'LINE2', 'SWITCH_PROB'])
        scorer=Scorer(self.__len_w, self.__syn_w, self.__misp_w, self.__smooth_f,self.stat_df,self.subs_df_en,self.subs_df_it)

        #combinations  -- !!!! O(N x N) complexity
        comb=list(product(self.query_map.keys(),repeat=2))
        new_rows={'LINE1':[], 'LINE2':[], 'SWITCH_PROB':[]}

        for pair in comb:
            source=pair[0]
            dest=pair[1]

            line1=self.query_map[source]
            line2=self.query_map[dest]
            weight=self.prob_df.loc[source]['PROB. OF SUCCESS']

            count=scorer.computeScore_translation(line1,line2)
            weighted_score = weight * count

            new_rows['LINE1'].append(source)
            new_rows['LINE2'].append(dest)
            new_rows['SWITCH_PROB'].append(weighted_score)

        #append data to df
        app=pd.DataFrame(new_rows)
        df = df.append(app, ignore_index = True)

        #finally normalize the probabilities column - avoid 0
        normalizer=MinMaxScaler(feature_range=(1e-9,1)) #avoid min=0 otherwise some score would then lead to a 100% prob. of failure
        df['SWITCH_PROB'] = normalizer.fit_transform(df[['SWITCH_PROB']])

        self.switchScore_df=df
        return df

    def getBatchSwitchScores(self,dim_batch) -> pd.DataFrame:
        """
        Computes a survivability score based on the probability
        of each line to switch into another. The score is computed
        according the following parameters:
            >> number of (italian) words the lines shares 
            >> probability score of the source line (from df_stat)

        A dataframe is returned with columns {'ID_LINE', 'SWITCH_PROB'}
        where the ID_LINES elements are the identifier from the mapping dictionary.

        Scores are computed by a stochastic procedure that make relations
        between each lines and a its cartesian product with a set of lines
        randomly sampled according to the user-defined dim_batch parameters,
        which tells the method the dimension of the sample to take from the
        subtitles dataframe.
        """

        #build df and Scorer
        df=pd.DataFrame(columns=['LINE', 'SWITCH_PROB'])
        df_key=pd.DataFrame([self.query_map]).T #map <ID,LINE> into Pandas DF

        scorer=Scorer(self.__len_w, self.__syn_w, self.__misp_w, self.__smooth_f,self.stat_df,self.subs_df_en,self.subs_df_it)
        
        new_rows={'LINE':[], 'SWITCH_PROB':[]}
        # for each line in the dataset
        for line in self.query_map.keys():
            
            weight=self.prob_df.loc[line]['PROB. OF SUCCESS']
            line_txt=self.query_map[line]

            #get samples
            sample=df_key.sample(n=dim_batch,replace=False) #avoid replace so to get the most variability of data as possible
                                                            #(particularly useful when dim_batch is small) 
                        
            #cartesian product of current line with id_lines sampled
            curr_line=pd.DataFrame({'curr_line':[line_txt]}) #one-row dataframe - needed to exploit 'merge'
            comb=curr_line.merge(sample,how='cross') #cartesian prod. line X sample [<String> CURR_LINE, <String> SAMPLE_LINE]
            comb.columns=['S','D'] 
            
            #add new col for the switch score of the current sample
            comb['SCORE']=weight * comb.apply(lambda x: scorer.computeScore_translation(x['S'],x['D']),axis=1)
            
            new_rows['LINE'].append(line)
            new_rows['SWITCH_PROB'].append(average(comb['SCORE'].values))

            
        #append data to df
        app=pd.DataFrame(new_rows)
        df = df.append(app, ignore_index = True)
        #print(df.head())

        #finally normalize the probabilities column - avoid 0
        normalizer=MinMaxScaler(feature_range=(1e-4,1)) #avoid min=0 otherwise some score would then lead to a 100% prob. of failure
        df['SWITCH_PROB'] = normalizer.fit_transform(df[['SWITCH_PROB']])

        df.set_index('LINE')
        self.switchScore_df=df
        return df


    def getNoisySwitchScore_survivability(self) :
        """
        According to the Switching Scores datasets this function
        returns the probability to experience an error as the 
        level of noise ranges between the lowest level (BER1)
        and the highest one (BER5).

        The noise presence affect the score according to
        a exponential distribution. The higher the level of noise,
        the lower the number of hops (path between two end-points)
        needed for an error to occur
        F(1)=P(error <= 1 hop) = 1-e^{- \lambda}

        The value of lambda is affected by the level of BER and 
        by the avarage switching score 'a' of the dataset.

        It returns the CPD of both DECODER_0 and DECODER_{0+k}
        which have different cardinalities.

        Returns:
            (DEC_0_cpd, DEC_0+k_cpd) : tuple of lists
        """
        
        #distribution parameters
        _lambda=[.3,.7,1.2,1.5,1.8]  #lambda equals the BER level
        _exp_distr=[1-exp(-ll) for ll in _lambda]

        df_0=pd.DataFrame(self.switchScore_df['SWITCH_PROB'])   # base dataframe for DECODER_0
        
        #a=self.switchScore_df['SWITCH_PROB'].mean() #average switching score over the whole dataset
        max=self.switchScore_df['SWITCH_PROB'].max()
        min=self.switchScore_df['SWITCH_PROB'].min()
        avg=(max-min)/2
        
        df=pd.DataFrame({'A':[avg,avg]})                            # base dataframe for DECODER_{0+k} - card: 2

        res=([],[])                

        exp_cdf_0 = lambda x : [item for sublist in [x * y for y in _exp_distr] for item in sublist] #weight by the score x
        exp_cdf_1 = lambda x : [item for sublist in [(1-x) * y for y in _exp_distr] for item in sublist]

        # -------------  DECODER_0 --------------
        df_res_0=df_0.apply(exp_cdf_0,axis=1,result_type='expand').apply(lambda x : 1 - x) #prob. of Error if DEC_{n-1} = TRUE (0) (error occurred)
        df_res_0=df_res_0.round(decimals=3)  #round      
        df_res_1=df_res_0.apply(lambda x : 1 - x,axis=1,result_type='expand') #prob. of Error if DEC_{n-1} = FALSE (1) (error did NOT occurr)
        
        res[0].append(list(df_res_0.to_numpy().flatten(order='F')))
        res[0].append(list(df_res_1.to_numpy().flatten(order='F')))

        # -------------- DECODER_0+k --------------
        df_res=df.apply(exp_cdf_0,axis=1,result_type='expand') #prob. of Error if DEC_{n-1} = TRUE (0) (error occurred)
        df_res=df_res.merge(df.apply(exp_cdf_1,axis=1,result_type='expand'),left_index=True,right_index=True) #prob. of Error if DEC_{n-1} = FALSE (1) (error did NOT occurr)
        df_res=df_res.round(decimals=3) #round
        df_res.iloc[0]=df_res.iloc[0].apply(lambda x: 1-x)  #prob.OfSuccess=1-prob.Of.Switch (only for the row 0 - TRUE)

        #make the format suitable with the pgmpy's CPD
        for i in df_res.index:
            res[1].append(df_res.iloc[i].tolist())

        self.noisySwitchCPD_surv=res

        return res


    def getNoisySwitchScore_hardcore(self, num_lines : int = 10) :
        """
        According to the Switching Scores datasets this function
        returns the probability of each line to change into the
        others as the level of noise ranges between the lowest 
        level (BER1) and the highest one (BER5).

        Starting from the `switchScore` dataset, only the worst
        num_lines are considered.
        The scores are re-mapped such that their sum is unitary, 
        so to obtain a measure suitable with a probability distribution.

        The noise presence affect the score according to
        a exponential distribution. The higher the level of noise,
        the lower the number of hops (path between two end-points)
        needed for an error to occur
        F(1)=P(error <= 1 hop) = 1-e^{- \lambda}

        The value of lambda is affected by the level of BER and the 
        final probability is weighted by the the switch score.

        args:
            num_lines: number of the worst lines to consider
        """
        
        #distribution parameters
        _lambda=[.3,.7,1.2,1.5,1.8]  #lambda equals the BER level
        _exp_distr=[1-exp(-ll) for ll in _lambda]

        df=pd.DataFrame(self.switchScore_df['SWITCH_PROB']).sort_values(by=['SWITCH_PROB'],ascending=False).iloc[:num_lines] # take the worst scoring line
        
        # 1. Get lines corresponding to the selected id
        lines=[self.query_map[j] for j in df.index]
        # 2. Associate the latter with id from 0
        new_query_map={i:lines[i] for i in range(len(lines))}
        # 3. Overwrite query_map, i.e., the association id <-> line according to the number of selected lines
        self.query_map=new_query_map
       
        df=pd.DataFrame(df).reset_index(drop=True)
        source=df.apply(lambda x : 1/x).to_numpy()
        dest=df.to_numpy()

        df=pd.DataFrame(source.dot(dest.T)) #dot product between scores (joint score) (P(line_X --> line_Y)) 
        
        res=[]        

        exp_cdf = lambda x : [item for sublist in [x * y for y in _exp_distr] for item in sublist] #weight by the joint score x
        
        df_res=df.apply(exp_cdf,axis=1,result_type='expand') #prob. of Error if DEC_{n-1} = TRUE (0) (error occurred)
        df_res=df_res.apply(lambda x: 1-x)  #prob.OfSuccess=1-prob.Of.Switch
        df_res=df_res.div(df_res.sum(axis=0), axis=1) #orthogonalize
        df_res=df_res.round(decimals=3)   # it should help getting the sum equal to one (not to 0.99999...)
        df_res=df_res.div(df_res.sum(axis=0), axis=1) # twice to avoid small rounding errors
        
        for i in df_res.index:
            res.append(df_res.iloc[i].tolist())


        self.noisySwitchCPD_hard=res
        return res


    def getNoisySwitchScore_flat(self, sourceLine_id):
        """
        Returns the CPD of SOURCE according to the
        switching probability of the source line
        and the avarage over all the dataset.
        """
           
        #avg=self.switchScore_df['SWITCH_PROB'].mean() #average switching score over the whole dataset
        max=self.switchScore_df['SWITCH_PROB'].max()
        min=self.switchScore_df['SWITCH_PROB'].min()
        tot=self.switchScore_df['SWITCH_PROB'].sum()

        avg=(max-min)/2       
        src_w=self.switchScore_df['SWITCH_PROB'][sourceLine_id]

        flip_prob=round((src_w+avg)/tot,3)
        
        res=[[1-flip_prob],[flip_prob]]

        self.noisySwitchCPD_flat=res

        return res


    #-------------UTIL FUNCTIONS-------------------------

    def getLine(self, id) -> str:
        """
        Returns the line corresponding to the given identifier
        contained in the mapping dictionary {id:line}. 

        args:
            id: (int) identifier of the line

        Index overflow is not currently handled.
        """
        return list(self.query_map.values())[id]
    
    def getId(self, line) -> int:
        """
        Returns the indentifier corresponding to the line
        passed as input from the mapping dictionary {id:line}.

        args:
            line: (str) line
        """

        return list(self.query_map.values()).index(line)

    def getProbOfSuccess(self):
        """
        Returns as a list the column 'PROB. OF SUCCESS'
        from the probability dataset in a format compatible
        with the requirements of the pgmpy TabularCPF.
        """
        res=[]
        for el in list(self.prob_df['PROB. OF SUCCESS']):
            res.append([el])
        return res

    def getProbOfFailure(self):
        """
        Returns as a list the column 'PROB. OF FAILURE'
        from the probability dataset in a format compatible
        with the requirements of the pgmpy TabularCPF.
        """
        res=[]
        for el in list(self.prob_df['PROB. OF FAILURE']):
            res.append([el])

        return res


    def getRandomLine(self):
        """
        Returns a random line from the Friend line dataset.
        """
        id=rand.choice(list(self.query_map.keys()))
        return (id,self.query_map[id])
        


