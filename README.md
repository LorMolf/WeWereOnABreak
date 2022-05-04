# WeWereOnABreak
An implementation of the "Chinese whispers" game with a Bayesian Network.

## Abstract
This project aims at exploiting the features and the strengths of Bayesian Networks for the implementation of the "Chinese whisperers" game. The presented
model allows us to determine whether the original message given as input goes through some changes due to erroneous word-of-mouth where the presence of noise
is taken into account, too. For this purpose, it has been created a custom dataset which includes a score representing the tendency of each sentence to mutate based
on the probability of each of its words being misspelt.


## 1. Introduction

The original *Chinese whispers* game, also known as the
Telephone game , requires more than 2 players to pass a message to each other, whispering it so that the other cannot
listen. 

The goal of the game, or maybe the hoped result, is to keep unchanged the
original message through the whole word of mouth process.
For the purpose of making the game more interesting and to better exploit the Bayesian
Network's potential, the following tweaks have been made:
  1. each player takes the input sentence, translates it into Italian and then translates it back to English before passing the message to the next player
  2. communications between each pair of players are affected by noise

The new feature (1) is included to add some more unpredictability.
The translation process makes it easier for the sense of a misspelt word to swap, indeed. Let's make
a simple example. If in the original message there's the word "won't" and the word "wan't" arrives instead, after the double translation the sentence's sense changes quite
drastically. The game is further made more meaningful from an engineering perspective by adding
some practical disturbance causes. Players can indeed be seen as endpoints of a telematic
communication, along whose path noise may occur due to one or more of the following
parameters:
- EbN0: bit Energy to Noise spectral density ratio
-  C/I: Carrier to Interference ratio
- Phi: Dopler phase shift

The values and the combinations of the latter make the noise to be classiffed according
`Bit Error Rate` to a (BER) scale, representing the frequency with which a swap of bit
occurs with respect to the length of the transmitted message, which ranges between the
values `BER 1` and `BER 5`, being the former the best-case scenario.

The dataset has been built from scratch starting from both the English and Italian
captions from the whole first season of the series *Friends*. The score of each of the
available words is computed according to the number of synonyms and their possible
misspellings. All of the previous statistics are computed starting from the dataset listed
in the bibliography.

The game provides three different modalities each of which implements a different struc-
ture for the Bayesian network, with consequent different types of queries the player can
make. Given an input message M, the modalities below make it possible to retrieve the
following information 
- flat mode : the score associated with M affects the CPDs' values in the BN's
structure. The player can query the probability of getting the same sentence at
the end of the path;
- survivability mode : the same as flat with some differences in the structure of the Bayesian model which shall be pointed out next;
- hardcore mode : the most complex and expensive mode that analyzes the proba- bility of getting to a specific output sentence starting from
M;

Every modality's features are net of the possibility to specify and analyze the noise presence.

## 2. Dataset and probability distributions
Starting from the synonym and misspelling statistics, the
switching probability of a line is computed according to the words composing it. Such a score is higher as higher is
the probability of getting an error. As a rule of thumb, we want the score to follow the
principles below:
- the smaller the word the higher the score - the same Bit Error Rate level affect
more word with fewer letters (fewer bits)
- the higher the number of synonyms the lower the score - in theory, the more are
the synonyms of a word, the higher the probability of a correct translation even
after a misspelling, so the lower the switching score
- the higher the number of possible misspellings, the higher the score

Each word <img src="https://render.githubusercontent.com/render/math?math={word}^{(i)}"> is thus associated with a score <img src="https://render.githubusercontent.com/render/math?math={word}^{(i)}_\text{score}">

<img src="https://render.githubusercontent.com/render/math?math={word}^{(i)}_\text{score}=\frac{\omega_\text{len} \cdot \lambda({word}^{(i)}) \!%2B\! \omega_\text{misp} \cdot \mu({word}^{(i)})}{\omega_\text{syn} \cdot \sigma({word}^{(i)}) \!%2B\! \varsigma_f}">
          
where :
- the smoothing factor <img src="https://render.githubusercontent.com/render/math?math=\varsigma_f"> is necessary in the case a particular word has neither synonyms nor misspellings available in the statistics dataset;
- <img src="https://render.githubusercontent.com/render/math?math=\lambda({word}^{(i)})"> returns the length of the word;
- <img src="https://render.githubusercontent.com/render/math?math=\lambda({word}^{(i)})"> returns the number of synonyms for the given word;
- <img src="https://render.githubusercontent.com/render/math?math=\sigma({word}^{(i)})"> returns the number of possible misspellings for the given word;

For every message <img src="https://render.githubusercontent.com/render/math?math=\mathcal{M}_i"> in the dataset, the score <img src="https://render.githubusercontent.com/render/math?math=\psi_{l_i}"> is then computed as the sum of its words' score

<img src="https://render.githubusercontent.com/render/math?math=\psi_{l_i}=\sum_{{word}_j  \in  \mathcal{M}_i.split()} {word}^{(j)}_\text{score}">

and then normalized over all the lines' scores.


The network deals with the probabilities of switching from each source message \
<img src="https://render.githubusercontent.com/render/math?math=\mathcal{M}_s"> to every destination message <img src="https://render.githubusercontent.com/render/math?math=\mathcal{M}_d">. In this regard, a <img src="https://render.githubusercontent.com/render/math?math=\textit{switch}_\text{score}"> for each couple of lines is calculated as the degree of similarity between the Italian translation of the respective messages

<img src="https://render.githubusercontent.com/render/math?math=\textit{switch}_\text{score}^{\mathcal{M}_s \rightarrow \mathcal{M}_d}= \psi_{\mathcal{M}_s} \ast \underbrace{\theta_{\mathcal{M}_s \rightarrow \mathcal{M}_d}}_{\substack{\text{translation}\\ \text{score}}}">

where the translation score <img src="https://render.githubusercontent.com/render/math?math=\theta_{\mathcal{M}_s \rightarrow \mathcal{M}_d}">
keeps track of the number of words the two sentences share. As before, the latter score is normalized over the data so as to obtain a probabilistic measure.

The final *switching score* for each available message <img src="https://render.githubusercontent.com/render/math?math=\mathcal{M}_i"> is then computed as

<img src="https://render.githubusercontent.com/render/math?math=\omega_i={avg}_{batch} \left ( \textit{switch}_\text{score}^{\mathcal{M}_i \rightarrow \mathcal{M}_d} \right )">

<img src="https://render.githubusercontent.com/render/math?math=\text{with} \mathcal{M}_d \in \textit{df.sample(n=batch\_dim, replace=False)}">

where the "*sample*" function return a random set of size *batch_dim* of different messages <img src="https://render.githubusercontent.com/render/math?math=\mathcal{M}_d"> from the dataset *df* that are used to compute a statistically consistent average of the *switching score* for the particular source message.


Regarding the noise condition along the communication path, the model is meant as a sort of transition system where the noise level of the previous endpoint, expressed in terms of BER, affects the clearness of the communication between the next pair of players. To model the likelihood of occurrence of an error, the Poisson distribution is used to express how many hops, in our case, how many message exchanges, are necessary to expect an error to verify.
![Alt text](plots/ber_dist.png?raw=true "Title")


Given such distributions, the noise level can either remain the same, improve or get worse according to the transition diagram below.
![Alt text](plots/BER_friends.png?raw=true "Title")


The way BER levels affect the switching probability of lines is finally computed as an exponential distribution.  As already mentioned, we expect to witness an error in a fewer number of steps as the noise condition worsens.  Appropriately setting the value of <img src="https://render.githubusercontent.com/render/math?math=\lambda_x"> to decrease as the <img src="https://render.githubusercontent.com/render/math?math={BER_x}"> level grows, the final probability <img src="https://render.githubusercontent.com/render/math?math=p^{(i)}_{\hat{x}}"> of getting an error for a message <img src="https://render.githubusercontent.com/render/math?math=\mathcal{M}_m"> with noise condition <img src="https://render.githubusercontent.com/render/math?math={BER}_{\hat{x}}"> is modelled as

<img src="https://render.githubusercontent.com/render/math?math=p^{(m)}_{\hat{x}} = 1 - \beta \cdot \underbrace{F_{\lambda_{\hat{x}}}(1)}_{1-e^{-\lambda_{\hat{x}}}}">

where the cumulative exponential distribution <img src="https://render.githubusercontent.com/render/math?math=F_{\lambda_{\hat{x}}}(1)">, which describes the likelihood of a bit flipping in a single message exchange, is weighted by <img src="https://render.githubusercontent.com/render/math?math=\beta">, whose value changes according to the chosen game modality:
- *hardcore* modality:
<img src="https://render.githubusercontent.com/render/math?math=\beta=\omega_{m} \ast \omega_{d}">, namely the probability of switching from $\mathcal{M}_m$ to every other message <img src="https://render.githubusercontent.com/render/math?math=\mathcal{M}_d">;

- *survivability* modality:
<img src="https://render.githubusercontent.com/render/math?math=\beta=avg(\omega_i)">,
being <img src="https://render.githubusercontent.com/render/math?math=avg(\cdot)"> the average score of all the lines in the dataset;

- *flat* modality:
<img src="https://render.githubusercontent.com/render/math?math=\beta=\frac{\omega_m + avg(\omega_i)}{\sum_{i \in df} \omega_i}">,
being <img src="https://render.githubusercontent.com/render/math?math=avg(\cdot)"> the average score of all the lines in the dataset;


The measure <img src="https://render.githubusercontent.com/render/math?math=p_k^{(m)}"> represents thus the probability of success for the message <img src="https://render.githubusercontent.com/render/math?math=\mathcal{M}_m"> under noise condition <img src="https://render.githubusercontent.com/render/math?math={BER}_k">.


To conclude, the Bit Error Rate levels' probability depending on the values of the telematic parameter above mentioned, are taken from the paper [1].




