# LabelRanking (LR)
Label Ranking algorithms &amp; data sets
9 LR algorithms implementations are included. One ([DTPG](./DecisionTreePointwiseGini.py)) is proposed by author, while among the other 8, 6 ([RPC](./RankPairPref.py), [LWR](./LabelWiseRanking.py), [SMP](./SMPrank.py), [KNN-PL](./KNNPlackettLuce.py), [KNN-M](./KNNMallows.py), [LogLinear](./logLinear.py)) are existing state-of-the-art label ranking algorithms, 2 are baselines (both implemented as [logRegression.py](./logRegression.py), as LogR and NAIVE).

## Data sets:
1. Facebook Post data: For each post in Facebook, sorting the predefined six emoticons by the votes from users to each of them into label ranking. The post feature is obtained using [AlchemyLanguage sentiment analysis tool](https://alchemy-language-demo.mybluemix.net/) to analyze post text content. Data format: each line as a post, with ["feature_emotion"] as post feature, and ["emoticons"] as number of votes for each emoticon. Reading data is through [ReadData](./ReadData.py).dataFacebook("datafilename"). To transform to label ranking data, using [ReadData](./ReadData.py).label2rank(). Detail can be found in [TrainTest.py](./TrainTest.py). There are four data sets, as following:
    1. [Random_Normal_User](./data) (ROU): Posts crawled from public posts of random normal users;
    2. [nytimes](./data) (NYT): Posts crawled from [New York Times Facebook Page](https://www.facebook.com/nytimes);
    3. [wapo](./data) (WaPo): Posts crawled from [The Washington Post Facebook Page](http://www.facebook.com/washington);
    4. [wsj](./data) (WSJ): Posts crawled from [The Wall Street Journal Facebook Page](https://www.facebook.com/wsj).
    
2. [Semi-synthetic data](./data/synthetic): Obtained by converting benchmark multi-class classification using Naive Bayes and regression data using feature-to-label technique from the UCI and Statlog repositories into label ranking~\cite{cheng2009decision}. These data sets are widely used as benchmark in label ranking works. As the original link in \cite{cheng2009decision} failed, we obtained the data sets from https://github.com/toppu/PLRank.
    
**Tail abstention**: For tail abstention problem, where ranking may be incomplete with some labels missing, according to the scenario in Facebook post data, they are considered abstention at tail position.

## Algorithms:
The original works codes refer to are cited at the beginning of each file. The usage is summarized in [TrainTest.py](./TrainTest.py) (not for running). Several modifications compared with original works are as follows:

1. [RPC](./RankPairPref.py): For tail abstention situation, when *Abstention* parameter is set to True, then all pairwise comparison is replicated *k* times, while pair of labels included in abstention contribute to 1 count in both orders. It is equivalent to count abstention as *1/k* times for each order. This is to make sure pairs of labels mainly appearing in tail abstention got smooth training.

2. [SMP](./SMPrank.py): The initialization is set differently from the original work. Here only distinct instances will be considered as prototypes, otherwise those same prototypes will keep the same as learning goes on, which makes no sense. As a result, the number of prototypes *K* should be set accordingly, as shown in [TrainTest.py](./TrainTest.py).

3. [DTPG](./DecisionTreePointwiseGini.py): With *alpha* parameter set as *None*, use pruning with searching hyperparameters with cross-validation in training set is used, with criteria set as *prune_criteria* parameter; as any real number larger than zero, use pruning with hyperparameter set as given value; as zero, no pruning.
