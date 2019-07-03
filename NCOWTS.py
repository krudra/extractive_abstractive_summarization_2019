import sys
from collections import Counter
import re
from gurobipy import *
import gzip
import networkx as nx
from textblob import *
import os
import time
import codecs
import math
import codecs
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.corpus import wordnet_ic, genesis
import numpy as np
import aspell
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
import pylab as pl
from itertools import cycle
from operator import itemgetter


LOWLIMIT = 1
UPLIMIT = 100
LSIM = 0.7
SIM_TH = 0.7
Tagger_Path = ''
ASPELL = aspell.Speller('lang', 'en')
WORD = re.compile(r'\w+')
cachedstopwords = stopwords.words("english")

def compute_similarity(ifname,placefile,keyterm,date,Ts):

	PLACE = {}
        fp = codecs.open(placefile,'r','utf-8')
        for l in fp:
                if PLACE.__contains__(l.strip(' #\t\n\r').lower())==False:
                	PLACE[l.strip(' #\t\n\r').lower()] = 1
        fp.close()

	fp = codecs.open(ifname,'r','utf-8')
	T = {}
	TW = {}
	M = {}
	P = []
	index = 0
	count = 0
	word = {}
	coword = {}
	window = 0
	t0 = time.time()
	
	for l in fp:
		count+=1
		wl = l.split('\t')
		temp_I = wl[3].split()
		All_I = wl[4].split()
		text = wl[2].strip(' \t\n\r')
		Length = int(wl[5])
		tid = wl[1].strip(' \t\n\r')
		temp = []
		for x in temp_I:
			x_0 = x.split('_')[0].strip(' \t\n\r')
			x_1 = x.split('_')[1].strip(' \t\n\r')
			if x_1=='PN':
                                s = x_0 + '_CN'
                                temp.append(s)
                        else:
                                temp.append(x)

		All = []
		for x in All_I:
			x_0 = x.split('_')[0].strip(' \t\n\r')
			x_1 = x.split('_')[1].strip(' \t\n\r')
			if x_1=='PN':
                                s = x_0 + '_CN'
                                All.append(s)
                        else:
                                All.append(x)

		################### Update word dictionary  ###################################
		for x in temp:
			if word.__contains__(x)==True:
				v = word[x]
				v+=1
				word[x] = v
			else:
				word[x] = 1
		
		################## Is it duplicate tweet ######################################

		k = compute_selection_criteria(All,TW)
		if k==1:
			T[index] = temp
			TW[index] = [tid,text,temp,All,Length]
			index+=1
		
	fp.close()
			
	L = len(T.keys())
	print('L: ',L)
	weight = compute_tfidf(word,count,PLACE)
	tweet_cur_window = {}
        for i in range(0,L,1):
        	temp = TW[i]
                tweet_cur_window[str(i)] = [temp[1],temp[4],set(temp[2])]   ### Text, Length, Content words ###
        
	##################### Finally apply cowts ################################
        ofname = keyterm + '_ICOWTS_' + date + '.txt'
       	optimize(tweet_cur_window,weight,ofname,Ts,0.4,0.6)
        print('Summarization done: ',ofname)
        window+=1
        t1 = time.time()
        print('Time Elapsed: ',t1-t0)

def compute_selection_criteria(current,T):
        for k,v in T.iteritems():
                new = set(current).intersection(set(v[3]))
                if len(new)==len(current):
                        return 0
        return 1

def optimize(tweet,weight,ofname,L,A1,A2):


	################################ Extract Tweets and Content Words ##############################
	word = {}
	tweet_word = {}
	tweet_index = 1
	for  k,v in tweet.iteritems():
		set_of_words = v[2]
		for x in set_of_words:
			if word.__contains__(x)==False:
				if weight.__contains__(x)==True:
					p1 = round(weight[x],4)
				else:
					p1 = 0.0
				word[x] = p1

		tweet_word[tweet_index] = [v[1],set_of_words,v[0]]  #Length of tweet, set of content words present in the tweet, tweet itself
		tweet_index+=1

	############################### Make a List of Tweets ###########################################
	sen = tweet_word.keys()
	sen.sort()
	entities = word.keys()
	print(len(sen),len(entities))

	################### Define the Model #############################################################

	m = Model("sol1")

	############ First Add tweet variables ############################################################

	sen_var = []
	for i in range(0,len(sen),1):
		sen_var.append(m.addVar(vtype=GRB.BINARY, name="x%d" % (i+1)))

	############ Add entities variables ################################################################

	con_var = []
	for i in range(0,len(entities),1):
		con_var.append(m.addVar(vtype=GRB.BINARY, name="y%d" % (i+1)))

	########### Integrate Variables ####################################################################
	m.update()


	P = LinExpr() # Contains objective function
	C1 = LinExpr()  # Summary Length constraint
	C4 = LinExpr()  # Summary Length constraint
	C2 = [] # If a tweet is selected then the content words are also selected
	counter = -1
	for i in range(0,len(sen),1):
		#P += tweet_word[i+1][0] * sen_var[i]
		P += sen_var[i]
		C1 += tweet_word[i+1][0] * sen_var[i]
		#C1 += sen_var[i]
		v = tweet_word[i+1][1] # Entities present in tweet i+1
		#print(v)
		C = LinExpr()
		flag = 0
		for j in range(0,len(entities),1):
			if entities[j] in v:
				flag+=1
				C += con_var[j]
		if flag>0:
			#print(C,flag)
			counter+=1
			m.addConstr(C, GRB.GREATER_EQUAL, flag * sen_var[i], "c%d" % (counter))

	C3 = [] # If a content word is selected then at least one tweet is selected which contains this word
	for i in range(0,len(entities),1):
		P += word[entities[i]] * con_var[i]
		#C4 = C4 + 'y' + str(i+1) + ' + '
		C = LinExpr()
		flag = 0
		for j in range(0,len(sen),1):
			v = tweet_word[j+1][1]
			if entities[i] in v:
				flag = 1
				C += sen_var[j]
		if flag==1:
			counter+=1
			m.addConstr(C,GRB.GREATER_EQUAL,con_var[i], "c%d" % (counter))

	counter+=1
	m.addConstr(C1,GRB.LESS_EQUAL,L, "c%d" % (counter))


	################ Set Objective Function #################################
	m.setObjective(P, GRB.MAXIMIZE)

	############### Set Constraints ##########################################

	fo = codecs.open(ofname,'w','utf-8')
	try:
		m.optimize()
		for v in m.getVars():
			if v.x==1:
				temp = v.varName.split('x')
				if len(temp)==2:
					fo.write(tweet_word[int(temp[1])][2])
					fo.write('\n')
	except GurobiError as e:
    		print(e)
		sys.exit(0)

	fo.close()

def compute_tfidf(word,tweet_count,PLACE):
        score = {}
        #discard = ['pray','prayer','updates','pls','please','hope','hoping','breaking','news','flash','update','tweet','pm','a/c','v/','w/o','watch','photo','video','picture','screen','pics','latest','plz','rt','mt','follow','tv','pic','-mag','cc','-please','soul','hoax','a/n','utc','some','something','ist','afr','guru','image','images']

	discard = []
        THR = 5
        N = tweet_count + 4.0 - 4.0
        for k,v in word.iteritems():
		D = k.split('_')
		D_w = D[0].strip(' \t\n\r')
		D_t = D[1].strip(' \t\n\r')
                if D_w not in discard:
                        tf = v
                        w = 1 + math.log(tf,2)
                        df = v + 4.0 - 4.0
                        try:
                                y = round(N/df,4)
                                idf = math.log10(y)
                        except Exception as e:
                                idf = 0
                        val = round(w * idf, 4)
                        if D_t=='P' and tf>=THR:
                                score[k] = val
                        elif tf>=THR and D_t=='S':
                                score[k] = val
			elif tf>=THR and len(D_w)>2:
				score[k] = val
                        else:
                                score[k] = 0
                else:
                        score[k] = 0
        return score

def main():
	try:
		_, ifname, placefile, keyterm, date, Ts = sys.argv
	except Exception as e:
		print(e)
		sys.exit(0)
	compute_similarity(ifname,placefile,keyterm,date,int(Ts))

if __name__=='__main__':
	main()
