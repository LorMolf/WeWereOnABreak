from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination, ApproxInference
from pgmpy.sampling import GibbsSampling
from pgmpy.sampling import BayesianModelSampling
import random as rand

import os, sys

import numpy as np
from texttable import Texttable

from models.model import  Model
from models.utils import getRouterRelation, getRandomEpisode, getBitErrorRate


class TheQueryMen():
    """
    Class that implements a slightly technical version
    of the "Chinese whispers", a.k.a. The Telephone Game, by means
    of Bayesian Network.

    This implementation allows the user to specify the number
    of persons (numOfEndPoints) that play the game. 
    Each communication is affected by noise and 
    therefore can make the words change.

    The level of noise is determined by some parameters and
    make the Bit Error Rate (BER) ranging between BER1 and BER5,
    BER1 being the lowest noise condition.
    
    Several modalities of game are available, each of which
    allows to make different type of queries:
     - flat : one the input line has been chosen, its switch score
                is used to tune the BN's CPDs. It allows the player to
                query the probability of keeping unchanged the original message
                until the end of the word-of-mouth, net of the possibility to specify
                the noise condition;
     - survivability : as for flat but with a different structure
                       in the SOURCE node's CPD;
     - hardcore  : accounts for the probability of switching
                    from one line to another;

    """
	
    def __init__(self,numOfPlayers : int = 3,modality : str = 'flat', num_lines :int = -1,noiseParamsDistr : list = [.5,.5,.5]) :
        """
        Constructor of the TheQueryMen class.

        Args:
            - numOfPlayers:         (int) number of players
            - modality:             (str) game modalities (flat, survivability, hardcore)
            - num_lines :           (int) number of lines of the dataset to load. It has to be specified only
                                    if the hardcore modality is chosen.
            - noiseParamsDistr:     (list) occurrence probability of the channel noise parameters [EbN0, C/I, Phi] 
        
        """
        self.__numOfEndPoints=numOfPlayers
        self.__modality=modality
        self.__input_line=None
        self.__num_lines=0
        
        self.__statScorePath=''
        self.__switchScorePath=''

        self.statDf=None
        self.switchDf=None

        self.__noiseParamsDistr=noiseParamsDistr

        self.model=Model(computeScores=False)
        print("STATISTICS LOADED !")  

        if modality == "flat":
            self.__num_lines=len(self.model.query_map.keys())
        elif modality == "survivability":
            self.__num_lines=num_lines if num_lines > 0 else len(self.model.query_map.keys())
            print("Building the Bayesian Network...")
            self.BN=self.__buildNetwork()
            print("The BN is ready to be queried !")
        elif modality =="hardcore":
            self.__num_lines=num_lines if num_lines > 0 else 15# number of worst lines
            print("Building the Bayesian Network...")
            self.BN=self.__buildNetwork()
            print("The BN is ready to be queried !")
        else:
            print("UNKNOWN MODALITY !!")
            exit(-1)
            
            

    def generateSource(self, input_line: str = None):
        """
        Generate a random line and return its correspondent 
        identifier. It is also possible to specify a particular 
        line giving it as input.

        If the chose modality if "flat", this method make the
        bayesian model to be recomoputed since the specific 
        line's switching score affects the value of the SCORE CPD.

        Args:
        - input_line (optional): (str) line to input in the game

        Return: 
        - id_line: (int) identifier
        - str_line: (str) line
        """
        if input_line is None:
            self.__input_line, line = self.model.getRandomLine()
            print(f"\nSOURCE MESSAGE: {line}\n")
        else:
            self.__input_line=self.model.getId(input_line)
            line=input_line


        if self.__modality == 'flat':       
            print("Building the Bayesian Network...")
            self.BN=self.__buildNetwork()
            print("The BN is ready to be queried !")

        return self.__input_line, line

    
    def __buildNetwork(self) -> BayesianNetwork:
        
        # Number of lines in the considered dataset
        num_lines=self.__num_lines

        # Define the structure according to the number of paths
        ber_src_list=['BER_' + str(level) for level in range(self.__numOfEndPoints)] #each BER_i has 5 possible values BER_1 .. BER_5
        ber_dest_list=['BER_' + str(level) for level in range(1,self.__numOfEndPoints+1)]
        
        dec_src_list=['DECODER_' + str(level) for level in range(self.__numOfEndPoints)]
        dec_dst_list=['DECODER_' + str(level) for level in range(1,self.__numOfEndPoints+1)]

        bn_structure=[]
        # source of noise
        bn_structure.append(('EbNO','BER_0'))
        bn_structure.append(('C/I','BER_0'))
        bn_structure.append(('Phi','BER_0'))
        
        # 1st layer
        bn_structure.append(('SOURCE','DECODER_0'))
        bn_structure.append(('BER_0','DECODER_0'))
    
        # next layers
        for src_ber,src_dec,dst_dec in zip(ber_dest_list,dec_src_list,dec_dst_list):
            bn_structure.append((src_dec,dst_dec))
            bn_structure.append((src_ber,dst_dec))
            
        for src_ber,dst_ber  in zip(ber_src_list,ber_dest_list):
            bn_structure.append((src_ber,dst_ber))

        bn_model = BayesianNetwork(bn_structure)
        
        #---------------------------CPDs------------------------------       
        
        
        # to change with suitable distribution
        # ['EbNO','C/I','Phi'] CPDs
        eb,ci,ph=self.__noiseParamsDistr
        ebno_cpd=TabularCPD('EbNO',2,[[eb],[1.0-eb]])
        c_i_cpd=TabularCPD('C/I',2,[[ci],[1.0-ci]])
        phi_cpd=TabularCPD('Phi',2,[[ph],[1.0-ph]])
        bn_model.add_cpds(ebno_cpd,c_i_cpd,phi_cpd)

        # ['EbNO','C/I','Phi'] --> BER_0
        noise_distr=getBitErrorRate()
        noise_cpd=TabularCPD('BER_0',5,noise_distr,evidence=['EbNO','C/I','Phi'],evidence_card=[2,2,2])
        bn_model.add_cpds(noise_cpd)
        #----------------------------------------

        # BER_X for X in range(num End Points) : BER_{i-1} --> BER_i
        #standard relationship between two consecutive noise channel (modelled by a Poisson distribution)
        noise_distr=getRouterRelation() 

        for parent,son in zip(ber_src_list, ber_dest_list):  
            ber_cpd=TabularCPD(son,5,noise_distr,evidence=[parent],evidence_card=[5])
            bn_model.add_cpds(ber_cpd) 

        # -------------SOURCE----&----DECODER_X----------------
        if self.__modality == 'flat':
            """
            All DECODERs have the same CPD. Given the
            flipping score of the input line and the average
            score of the whole dataset, probability of success
            are computed accordingly to the CPF of the exponential
            distribution.
            --> TOTAL CARD: 
                2 x 2 (prev_dec) x 5 (BER)
            """

            source_distr=self.model.getNoisySwitchScore_flat(self.__input_line)
            _,dec_distr=self.model.getNoisySwitchScore_survivability() # the decoders' distributions is the same as the survivability mode

            # SOURCE
            src_cpd=TabularCPD('SOURCE',2,source_distr)            
            bn_model.add_cpds(src_cpd)

            # DECODER_0 <- (SOURCE, BER_0)
            dec_cpds=TabularCPD('DECODER_0',2,dec_distr,evidence=['SOURCE','BER_0'],evidence_card=[2,5])
            bn_model.add_cpds(dec_cpds)

            
            # DECODER_X for X in range(1, num End Points)
            dec_cpds=list([TabularCPD(dec_dst,2,dec_distr,evidence=[dec_src,ber],evidence_card=[2,5])  \
                for dec_src,dec_dst,ber in zip(dec_src_list, dec_dst_list,ber_dest_list)])
            
            for cpd in dec_cpds : bn_model.add_cpds(cpd)   
        else:

            # first element 1.0, others 0 - in theory it should not be a problem
            # being SOURCE always given.
            uniform_distr=[[.0] for _ in range(num_lines)]
            uniform_distr[0]=[1.0]
            rand.shuffle(uniform_distr)

            src_cpd=TabularCPD('SOURCE',num_lines,uniform_distr)            
            bn_model.add_cpds(src_cpd)

            
            if self.__modality == 'survivability':
                """
                DECODERs just return the probability of error
                given the 'robustness' of the given line. To do so,
                their CPDs have cardinality 2.
                
                Decoder_0 and the next ones are thus different.

                --> TOTAL CARD DECODER_0:
                2 (curr_dec) x num_lines (SOURCE) x 5 (BER)
                
                --> TOTAL CARD DECODER_{0+k}: 
                2 (curr_dec) x 2 (prev_dec) x 5 (BER)
                """
                
                #dec_0_scores,dec_k_scores=self.model.noisySwitchCPD_surv
                dec_0_scores,dec_k_scores=self.model.getNoisySwitchScore_survivability()

                dec_0_scores=list(np.around(np.array(dec_0_scores),3))
                dec_k_scores=list(np.around(np.array(dec_k_scores),3))

                # DECODER_0 <-- (SOURCE, BER_0)
                dec_cpds=TabularCPD('DECODER_0',2,dec_0_scores,evidence=['SOURCE','BER_0'],evidence_card=[num_lines,5])
                bn_model.add_cpds(dec_cpds)


                # DECODER_X for X in range(1, num End Points)
                dec_cpds=list([TabularCPD(dec_dst,2,dec_k_scores,evidence=[dec_src,ber],evidence_card=[2,5])  \
                    for dec_src,dec_dst,ber in zip(dec_src_list, dec_dst_list,ber_dest_list)])
                
                for cpd in dec_cpds : bn_model.add_cpds(cpd)

            else:       
                # self.__modality == 'hardcore':
                """
                DECODERs make some probabilistic inference about
                the proability of getting from the input line to
                a specific another one. In this regard, CPDs have
                cardinality num_lines.
                --> TOTAL CARD: 
                num_lines (curr_dec) x num_lines (prev_dec) x 5 (BER)
                """
  
                flip_distr=self.model.getNoisySwitchScore_hardcore(num_lines)

                dec_cpds=TabularCPD('DECODER_0',num_lines,flip_distr,evidence=['SOURCE','BER_0'],evidence_card=[num_lines,5])
                bn_model.add_cpds(dec_cpds)

                # DECODER_X for X in range(1, num End Points)
                dec_cpds=list([TabularCPD(dec_dst,num_lines,flip_distr,evidence=[dec_src,ber],evidence_card=[num_lines,5])  \
                    for dec_src,dec_dst,ber in zip(dec_src_list, dec_dst_list,ber_dest_list)])
                
                for cpd in dec_cpds : bn_model.add_cpds(cpd)
        
        # Check the consistency of the model
        # print(bn_model.check_model())
        self.BN=bn_model

        return bn_model        

    
    def __getMostProbableOutput(self,prob_vals):
        """
        Print the most probable output message given
        the probability distribution.
        """
        most_prob_out_line=np.argmax(prob_vals)
        line=self.model.getLine(most_prob_out_line)

        prob=round(prob_vals[most_prob_out_line],3)

        return line,prob

    def __getGibbsSamples(self,num_samples):
        """
        Returns samples generated with the Gibbs method.        
        """
        gibbs = GibbsSampling(self.BN)
        return gibbs.sample(size=num_samples)
        

    def makeApproximateQuery(self, variables : list = [], evidence : dict = {}, sampling_type='rej', num_samples=1000, printCPD=False) -> dict:
        """
        Makes an approximate query according to the game mode. If no evidence
        is given in input, this function returns the distribution of `variables` 
        using samples obtained with the specified `sampling_method`:
            - 'rej' : rejection sampling 
            - 'lw'  : likelihhod-weighted sampling
            - 'gib' : gibbs sampling

        
        Args:
            - variables:        variables name to get distribution of
            - evidence:         variables name given as evidence
            - sampling_type:    type of sampling method
            - num_samples:      number of samples to draw
            - printCPD:         (bool) print the output CPD

        Return:
            - distribution:     (dict) probability distribution of the query's outcome
        """

        model=self.BN        
        apprx_inf=ApproxInference(model)

        if evidence is None:
            inference=BayesianModelSampling(model)
            
            if sampling_type == 'rej':
                samples=inference.rejection_sample(size=num_samples)
            elif sampling_type == 'lw':
                samples=inference.likelihood_weighted_sample(size=num_samples)
            else: # 'gib'
                samples=self.__getGibbsSamples(num_samples)

            distr=apprx_inf.get_distribution(samples,variables=variables,joint=True)
        else:            
            distr=apprx_inf.query(variables=variables,evidence=evidence,n_samples=num_samples)
            
        if printCPD:
            print(distr)

        return distr.values

    def makeExactQuery(self,variables : list = [],evidences : dict = {}, printCPD=False) -> dict:
        """
        Run the query P(variables|evidence) with the Variable Elimination algorithm.
        If the `printCPD` parameter is set to True then the table with all probabilities
        is printed.

        Args:
            - variables:        (list) variables name to get distribution of
            - evidence:         (dict) variables name given as evidence
            - printCPD:         (bool) print the output CPD

        Return:
            - distribution:     (dict) probability distribution of the query's outcome
        """

        inference=VariableElimination(self.BN)      
        distr=inference.query(variables,evidence=evidences)

        if printCPD:
            print(distr)
        
        return distr.values

    
    def alwaysUnchanged(self, evidence : dict = {}, eliminationOrder : list = [], printCPD : bool = False) -> dict:
        """
        Computes the probability of keeping always unchanged the
        input line along every message exchange.

        Args:
            - evidence: (dist) dictionary of variables observed
            - printCPD: (bool) print the output CPD
            - eliminationOrder: (list) order of elimination for the Variable Elimination order 

        Return:
            - distribution: (dict) probability distribution of the query's outcome        
        """
        variables=[]
        
        if self.__modality == 'flat':
            variables.append('SOURCE')

        for i in range(self.__numOfEndPoints+1):
            variables.append(f'DECODER_{i}')

        inference=VariableElimination(self.BN)      
        
        if eliminationOrder == []:
          distr=inference.query(variables,evidence=evidence)
        else:
          distr=inference.query(variables,evidence=evidence,elimination_order=eliminationOrder)

        vals=distr.values
        
        if self.__modality == "hardcore":
            source=evidence['SOURCE']

            input_line=self.model.getLine(source)
            input_prob=round(vals[source],3)
            out_line, out_prob=self.__getMostProbableOutput(vals)
            t = Texttable()
            t.add_rows([[f'INPUT (PROB. {input_prob})', f'MOST PROBABLE OUTPUT (PROB. {out_prob})'], [input_line, out_line]])
            print(t.draw())
            
            if printCPD: print(distr)

        elif printCPD:
            print(np.array(vals).flatten()[0]) # distr.values[0] stores the probability that all the input variables (variables) has value 0, i.e., they succeeded
        
        return vals


        def traceSwitches(self,variables : list = [],evidences : dict = {}) -> dict:
            """
            This function, which can be used only in the 'hardcore' mode, compute
            the given query  P(variables|evidence) with the Variable Elimination
            algorithm.

            Exploting the structure of the Bayesian network for this modality, all the
            intermediate results are printed out. For each player, the function returns
            the probability of success of the original line and the actual most probable 
            output message.

            Args:
                - variables:        (list) variables name to get distribution of
                - evidence:         (dict) variables name given as evidence
                - printCPD:         (bool) print the output CPD

            Return:
                - distribution:     (dict) probability distribution of the query's outcome
            """

            if self.__modality != 'hardcore':
                print("This function is available for the 'hardcore' modality only !")
                exit(-1)

            
            # reset std output to avoid printing the intermediate messages
            old_stdout = sys.stdout # backup current stdout
            sys.stdout = open(os.devnull, "w")

            source=evidences['SOURCE']
            input_line=self.model.getLine(source)

            for n_pl in range(2,self.__numOfEndPoints+1):
                tmp_model = TheQueryMen(n_pl,modality='hardcore',num_lines=self.__num_lines)
                tmp_model.generateSource(input_line)

                vals=tmp_model.makeExactQuery([f'DECODER_{n_pl}'],evidences=evidences)
                        
                input_prob=round(vals[source],3)
                out_line, out_prob=self.__getMostProbableOutput(vals)
                t = Texttable()
                t.add_rows([[f'INPUT (PROB. {input_prob})', f'MOST PROBABLE OUTPUT (PROB. {out_prob})'], [input_line, out_line]])
                
                sys.stdout = old_stdout # restore original std output
                print(f"\n-----------------> PLAYER_{n_pl}")
                print(t.draw())
                sys.stdout = open(os.devnull, "w")



            sys.stdout = old_stdout # restore original std output

            return vals



