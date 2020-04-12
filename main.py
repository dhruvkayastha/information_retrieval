import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

label_map = {'no value': 0, 'potential value':1, 'certain value':2, 'high value':3}

def DCG(ranks, relevance, k=10):
	index = np.arange(k)+1
	return np.sum(relevance[ranks][:k]/np.log2(index+1))

def direct_method(filename, vocabulary):
	dframe = pd.read_json(filename, orient='index')
	print "Query:", filename.split('_')[0].upper()
	vectorizer = CountVectorizer(vocabulary=vocabulary)
	X = vectorizer.fit_transform(dframe['text'])
	tf = np.asarray(X.toarray())
	Ld = np.asarray(dframe['text'].str.split().apply(len))
	N = len(Ld)
	df = np.sum(tf>0, axis=0)
	Lavg = np.mean(Ld)
	relevance = np.zeros((N))
	label_array = np.asarray(dframe['label'])
	for lab in label_map:
		relevance[label_array == lab] = label_map[lab]

	k1 = 1.2
	b = 0.75
	k3 = 1.2
	tftq = 1.0
	BM25 = np.zeros((N))
	for t in range(len(vocabulary)):
		TF = (k1 + 1)*tf[:, t]/(k1*(1 - b + b*Ld/Lavg) + tf[:, t])
		IDF = np.log10((N - df[t] + 0.5)/(df[t] + 0.5))
		QTF = (k3 + 1.0)*tftq/(k3 + tftq) # for our queries	
		BM25 += TF*IDF*QTF

	BM25 = np.argsort(-BM25)

	TF_ISF = np.zeros((N))
	for t in range(len(vocabulary)):
		TF = np.log10(tf[:, t] + 1)
		IDF = np.log10((N + 1.0)/(df[t] + 0.5))
		QTF = np.log10(1 + tftq) # for our queries	
		TF_ISF += TF*IDF*QTF

	TF_ISF = np.argsort(-TF_ISF)

	DCG_10 = DCG(BM25, relevance)
	DCG_100 = DCG(BM25, relevance, k=100)

	ideal_ranks = np.argsort(-relevance)
	ideal_DCG_10 = DCG(ideal_ranks, relevance)
	ideal_DCG_100 = DCG(ideal_ranks, relevance, k=100)

	print "\tBM25 NDCG@10:", DCG_10/ideal_DCG_10
	print "\tBM25 NDCG@100:", DCG_100/ideal_DCG_100

	TF_DCG_10 = DCG(TF_ISF, relevance)
	TF_DCG_100 = DCG(TF_ISF, relevance, k=100)

	print "\tTF-ISF NDCG@10:", TF_DCG_10/ideal_DCG_10
	print "\tTF-ISF NDCG@100:", TF_DCG_100/ideal_DCG_100

	return np.asarray([DCG_10/ideal_DCG_10, DCG_100/ideal_DCG_100, TF_DCG_10/ideal_DCG_10, TF_DCG_100/ideal_DCG_100])


vocab_CBP = {'common':0, 'business':1, 'purpose':2}
vocab_IEV = {'independent':0, 'economic':1, 'value':2}
vocab_IP = {'identifying':0, 'particular':1}

CBP_scores = direct_method('cbp_sentence.json', vocab_CBP)
IEV_scores = direct_method('iev_sentence.json', vocab_IEV)
IP_scores = direct_method('ip_sentence.json', vocab_IP)

print "Average Scores:"

average_scores = (CBP_scores + IEV_scores + IP_scores)/3.0

print "\tBM25 NDCG@10:", average_scores[0]
print "\tBM25 NDCG@100:", average_scores[1]

print "\tTF-ISF NDCG@10:", average_scores[2]
print "\tTF-ISF NDCG@100:", average_scores[3]
