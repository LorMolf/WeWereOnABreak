from os import sep
import numpy as np
import re

class Scorer():
    """
    Compute tha (failure) score of a sentence. The higher the score
    the more likely the sentence will be erroneusly traslated
    due to noise.py

    The total score is the summation of the score of each of its words
    according to the following parameters:
        >> length                   --> the shorter the word, the higher the score since a small change 
                                        relatively affect the number of errors over the total number of letters
        >> number of synonyms       --> the higher, the lower the score. During the final translation it is indeed
                                        more probable to get the original meaning also if the word is affected by noise
        >> number of mispellings    --> the higher, the higher the score since there are more than
                                        more possibilities to get a mispelled version of the considered word

    All the previous factors are weighted by different step sizes 
    that can be defined in the constructor.

    The dataset must be given as input with the format of the loader class. A smoothing factor 
    is passed so to address the problem of computing the score for a (currently) missing word in the dataset.
    """
    def __init__(self, len_w, syn_w, misp_w, smoothing_fact, dataSet_stats, dataSet_en, dataSet_it) -> None:
        self.__len_w=len_w
        self.__syn_w=syn_w
        self.__misp_w=misp_w
        self.__smooth_f=smoothing_fact

        #Subtitles dataset
        self.df_stats=dataSet_stats
        self.df_en=dataSet_en
        self.df_it=dataSet_it


    def __retrieveStatData(self, word) -> list[np.float64,np.float64]:
        """
        Retrieve data from the dataset for the word given as input.
        It returns a smoothing factor in case of missing values.

        It may happen that the word to compute score on is not
        present in the dataset. In such a case the following value
        would be returned :
            num_synonyms:-1
            num_mispellings:-1
        The latter are accordingly handled by the scoring function
        so to attain a consistent score.

        args:
            word: (str) word to get data for
        Return:
            (num_synonyms, num_mispellings)
        """
        
        try:
            entry=self.df_stats.loc[word]  
        except:         #KeyError(key) --> word not in dataset
            res=[-1,-1]    
        else:        
            res=[entry['NUM_SYNONYMS'],entry['NUM_MISPELLINGS']]

        return res

    def __word_score(self,word) -> np.float64:
        """
        Computes the score of a single word according
        to its length, synonym and mispelling parameters.

        If the word is not in the dataset, a score of 0.5 is assinged.
        Once it is converted into a probability, such a score will make the 
        word's error likelihood to behave like a fair coin distribution.
        """
        length = len(word)
        num_syn, num_misp = self.__retrieveStatData(word)
        
        if num_syn < 0:
            sc=0.5
        else:
            sc = self.__len_w * length + self.__syn_w * num_syn + self.__misp_w * num_misp + self.__smooth_f
        
        return sc

    def __word_score_2(self,word) -> np.float64:
        """
        Computes the score of a single word according
        to its length, synonym and mispelling parameters.

        If the word is not in the dataset, a score of 0.5 is assinged.
        Once it is converted into a probability, such a score will make the 
        word's error likelihood to behave like a fair coin distribution.

        This is an alternative version of the score. It accounts for missing
        words by means of the smoothing factors. Its handling
        is coped with by the retrieving function so it's transparent
        to this function.
        """
        length = len(word)
        num_syn, num_misp = self.__retrieveStatData(word)
        
        if num_syn < 0:
            sc=0.5
        else:
            sc = (self.__len_w * length + self.__misp_w * num_misp) / (self.__syn_w * num_syn + self.__smooth_f)
        
        return sc

    
    def computeScore_line(self,line,score_verion) -> np.float64:
        """
        Computes the score the of the english line given as input
        according to the word it is composed by.

        args:
            line:           string
            score_version:  which version of the score function to use
        """
        sc=0.0
        s_fun=self.__word_score if score_verion==1 else self.__word_score_2
        line=re.sub('[^A-Za-z0-9 ]+', ' ', line) #remove special characters (except for %20)

        for word in line.split():
            sc += s_fun(word)

        return sc


    def __compareItalianLines(self,line1,line2) -> np.int64:
        """
        Return the number of words the two lines share.py

        args:
            line1: (string) line from the dataset of italian subtitles
            line2: (string) line from the dataset of italian subtitles
        """

        #split lines into tokens
        l1=line1.split(sep=' ')
        l2=line2.split(sep=' ')

        #make the first line as a dict with each word associated with the number 0
        l1_dict={str.lower(word):0 for word in l1}
        
        #loop over the list of word of the second one and increment the corresponding value if the list (if present)
        for word in [el for el in l2 if el != ''] :
            word=str.lower(word)
            try:
                count=l1_dict[word]
            except: pass #not shared word
            else:
                l1_dict[word]=count + 1

        #sum all the dict.values()
        res=sum(l1_dict.values())

        return res


    def computeScore_translation(self,line1,line2) : #-> np.float64:
        
        """
        Computes the translation score of the given lines
        according to the number of words they share. 

        args:
            line1: (string) line from the dataset of english subtitles
            line2: (string) line from the dataset of english subtitles
        """
        d_e=self.df_en
        d_i=self.df_it

        #the case of duplicates here it is simply handled by taking the first entry (not the best solution though)        
        timestamp1=str(d_e.loc[d_e['LINES']==line1]['TIMESTAMPS'].iloc[0])
        timestamp2=str(d_e.loc[d_e['LINES']==line2]['TIMESTAMPS'].iloc[0])
        

        #handle error in case of missing timestamps - it can happen when Chandler makes joke too long that are split
        #in multiple line in the italian translation
        shared_word=0
        try:
            
            line1_it=str(d_i.loc[d_i['TIMESTAMPS']==timestamp1]['LINES'].iloc[0])
            line2_it=str(d_i.loc[d_i['TIMESTAMPS']==timestamp2]['LINES'].iloc[0])

            #remove special characters except for %20
            line1_it=re.sub('[^A-Za-z0-9 ]+', ' ', line1_it) 
            line2_it=re.sub('[^A-Za-z0-9 ]+', ' ', line2_it)        
            
            shared_word=self.__compareItalianLines(line1_it, line2_it)

        except: pass

        return shared_word
        