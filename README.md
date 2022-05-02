# WeWereOnABreak
An implementation of the "Chinese whispers" game with a Bayesian Network.

## Abstract
This project aims at exploiting the features and the strengths of Bayesian Networks for the implementation of the "Chinese whisperers" game. The presented
model allows us to determine whether the original message given as input goes through some changes due to erroneous word-of-mouth where the presence of noise
is taken into account, too. For this purpose, it has been created a custom dataset which includes a score representing the tendency of each sentence to mutate based
on the probability of each of its words being misspelt.


## Introduction


## Model
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

## Dataset and probability distributions
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

Each word <img src="https://render.githubusercontent.com/render/math?math=w^{(i)}"> is thus associated with a score 
<img src="https://render.githubusercontent.com/render/math?math=w^{(i)}_\text{score}">

<img src="https://render.githubusercontent.com/render/math?math=w^{(i)}_\text{score}=\omega_\text{len} \cdot \lambda(w^{(i)}) + \omega_\text{syn} \cdot \sigma(w^{(i)}) + \omega_\text{misp} \cdot \mu(w^{(i)}) + \varsigma_f">
          
where the <img src="https://render.githubusercontent.com/render/math?math=\varsigma_f"> parameter is necessary 
in the case a particular word has neither synonyms nor misspellings available in the statistics dataset.


