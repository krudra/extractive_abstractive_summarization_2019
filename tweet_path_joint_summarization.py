import sys
from collections import Counter
import re
from textblob import *
from gurobipy import *
import gzip
import os
import time
import codecs
import math
import networkx as nx
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.corpus import wordnet_ic, genesis
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
import aspell
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
import pylab as pl
from itertools import cycle
from operator import itemgetter

LOWLIMIT = 0
UPPERLIMIT = 1
THR_COMP = 0

WT1 = 1
WT2 = 1
WT3 = 0

LSIM = 0.7
lmtzr = WordNetLemmatizer()
Tagger_Path = ''
ASPELL = aspell.Speller('lang', 'en')
WORD = re.compile(r'\w+')


cachedstopwords = stopwords.words("english")
AUX = ['be','can','cannot','could','am','has','had','is','are','may','might','dare','do','did','have','must','need','ought','shall','should','will','would','shud','cud','don\'t','didn\'t','shouldn\'t','couldn\'t','wouldn\'t']
NEGATE = ["aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt",
              "ain't", "aren't", "can't", "couldn't", "daren't", "didn't", "doesn't",
              "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt", "neither",
              "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
              "neednt", "needn't", "never", "none", "nope", "nor", "not", "nothing", "nowhere",
              "oughtnt", "shant", "shouldnt", "uhuh", "wasnt", "werent",
              "oughtn't", "shan't", "shouldn't", "uh-uh", "wasn't", "weren't",
              "without", "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "despite"]
PUNC = ['!','?','.',':',';']

def compute_summary(mainname,mainparsefile,ifname,parsefile,placefile,keyterm,date,Ts):

	###################################### Read Place Information ############################################################
	PLACE = {}
        fp = codecs.open(placefile,'r','utf-8')
        for l in fp:
                if PLACE.__contains__(l.strip(' \t\n\r').lower())==False:
                	PLACE[l.strip(' \t\n\r').lower()] = 1
        fp.close()

	######################################## Processing Raw File ###########################################################
	
	RPL = ['+','-',',','+91']
	index = 0
	count = 0
	dic = {}
	L = 0
	TAGREJECT = ['#','@','~','U','E','G',',']


        t0 = time.time()
	T = {}
	CT = {}
	SCT = {}
	content_count = {}
	notag_content_count = {}
	topic_count = {}
	TOPIC_SET = set([])
	LANG_SCORE = []
	IMP_SCORE = []
	SEQ2DIC = {}
	DIC2SEQ = {}
	TOTAL_COMP = 0
	TOTAL_SELECTED_COMP = 0


	fp = codecs.open(mainparsefile,'r','utf-8')
	ft = codecs.open(mainname,'r','utf-8')
	
	for l in fp:
		wl = l.split('\t')
		if len(wl)==8:
                        seq = int(wl[0])
                        main_word = wl[1].strip(' #\t\n\r').lower()
                        word = wl[1].strip(' #\t\n\r').lower()
                        tag = wl[4].strip(' \t\n\r')
                        dep = wl[6].strip(' \t\n\r')
                        if dep=='_':
                                dep = int(wl[7].strip(' \t\n\r'))
                        else:
                                dep = int(wl[6])

                        if tag=='$':
                                s = word.strip(' \t\n\r')
                                Q = s
                                for x in RPL:
                                        Q = s.replace(x,'')
                                        s = Q
                                Q = s.lstrip('0')
                                s = Q
                                try:
                                	w = str(numToWord(int(s)))
                               		if len(w.split())>1: # like 67
                                        	w = s
                            	except Exception as e:
                                	w = str(s)
                                word = w.lower()
                        elif tag=='N':
				try:
                                	w = lmtzr.lemmatize(word)
                                	word = w.lower()
				except Exception as e:
					pass
                        elif tag=='^':
				try:
                                	w = lmtzr.lemmatize(word)
                                	word = w.lower()
				except Exception as e:
					pass
				tag = 'N'
                        elif tag=='V':
                                try:
                                        w = Word(word.lower())
                                        x = w.lemmatize("v")
                                except Exception as e:
                                        x = word.lower()
                                word = x.lower()
                        else:
				pass

                        temp = [word,tag,dep,main_word]
                        dic[seq] = temp
			SEQ2DIC[seq] = word
			DIC2SEQ[word] = seq
		else:
			######################## Identify singleton component #############################################
			temp = dic.keys()
                        temp.sort()
                        G = nx.Graph()
                        for x in temp:
                                G.add_node(x)
                        for x in temp:
                                dep = dic[x][2]
                                if dep!=-1 and dep!=0 and dic[x][1] not in TAGREJECT:
                                        G.add_edge(dep,x)
                        temp = sorted(nx.connected_components(G), key = len, reverse=True)
			reject = set([])

			######################### Identify Starting Component #############################################

			START_COMP = []
			KEYS = dic.keys()
			KEYS.sort()
			flag = 0
			for ky in KEYS:
				X = dic[ky]
				if X[2]!=-1 and flag==0:
					START_COMP.append(ky)
					flag=1
				else:
					prev_key = ky - 1
					next_key = ky + 1
					if prev_key in KEYS and next_key in KEYS:
						T1 = dic[prev_key]
						T2 = dic[next_key]
						if T1[2]==-1 and T2[2]==-1 and T1[0] in PUNC and T2[0] in PUNC:
							START_COMP.append(ky)

			TL = ft.readline().split('\t')
			Tweet_Length = int(TL[5])
			COMP_SIZE = len(temp)

			TOTAL_COMP = TOTAL_COMP + COMP_SIZE

			if len(temp)>THR_COMP:
				for i in range(0,len(temp),1):
					comp = list(temp[i])
					if len(comp)<=1:
						if PLACE.__contains__(dic[comp[0]][0])==True:
							X = dic[comp[0]][0] + '_P'
						elif dic[comp[0]][1]=='N':
							X = dic[comp[0]][0] + '_CN'
						elif dic[comp[0]][1]=='^':
							X = dic[comp[0]][0] + '_CN'
						elif dic[comp[0]][1]=='$':
							X = dic[comp[0]][0] + '_S'
						else:
							X = dic[comp[0]][0] + '_' + dic[comp[0]][1]
						reject.add(X)

				
				if Tweet_Length>1:
					try:
						for i in range(0,len(START_COMP),1):
							if SEQ2DIC[START_COMP[i]+1].find(':')!=-1:
								temp = dic[START_COMP[i]]
								if PLACE.__contains__(temp[0])==True:
									X = temp[0] + '_P'
								elif temp[1]=='N':
									X = temp[0] + '_CN'
								elif temp[1]=='^':
									X = temp[0] + '_CN'
								elif temp[1]=='$':
									X = temp[0] + '_S'
								else:
									X = temp[0] + '_' + temp[1]
								reject.add(X)
					except:
						pass
			


			######################## CONTENT WORD EXTRACTION #################################################################		
			content = set([])
			content_original = set([])
			temp = TL[3].split()
			for x in temp:
				x_0 = x.split('_')[0].strip(' \t\n\r')
				x_1 = x.split('_')[1].strip(' \t\n\r')
				if x_1=='PN':
					s = x_0 + '_CN'
					content_original.add(s)
					if s not in reject:
						content.add(s)
				else:
					content_original.add(x)
					if x not in reject:
						content.add(x)
			All = set([])
			temp = TL[4].split()
			for x in temp:
				x_0 = x.split('_')[0].strip(' \t\n\r')
				x_1 = x.split('_')[1].strip(' \t\n\r')
				if x_1=='PN':
					s = x_0 + '_CN'
					if s not in reject:
						All.add(s)
				else:
					if x not in reject:
						All.add(x)
				
			L = int(TL[5])
			
			######################### SET COUNT #################################################################

			for x in content:
				x10 = x.split('_')[0].strip(' \t\n\r')
				if notag_content_count.__contains__(x10)==True:
					v = notag_content_count[x10]
					v+=1
					notag_content_count[x10] = v
				else:
					notag_content_count[x10] = 1
			
			for x in content:
				if content_count.__contains__(x)==True:
					v = content_count[x]
					v+=1
					content_count[x] = v
				else:
					content_count[x] = 1

			TSC_IMP = round(float(TL[6]),2)
			k = should_select(SCT,All)
			if k>=1:
				CT[index] = content
				SCT[index] = All
				T[index] = [TL[2].strip(' \t\n\r'),content,L,0,TSC_IMP,reject,content_original]
				TOTAL_SELECTED_COMP = TOTAL_SELECTED_COMP + COMP_SIZE
				index+=1

			dic = {}
			SEQ2DIC = {}
			DIC2SEQ = {}
			count+=1
       
	fp.close()
	ft.close()
	print('Just raw tweets: ',count,index)
	raw_tweet_index = index
	
	PP1 = TOTAL_COMP + 4.0 - 4.0
	PP2 = count + 4.0 - 4.0
	try:
		PP3 = round(PP2/PP1,4)
	except:
		PP3 = 0
	print('Component, Component/Tweet: ',TOTAL_COMP,PP3)
	
	PP1 = TOTAL_SELECTED_COMP + 4.0 - 4.0
	PP2 = index + 4.0 - 4.0
	try:
		PP3 = round(PP2/PP1,4)
	except:
		PP3 = 0
	print('Component, Component/Selected Tweet: ',TOTAL_SELECTED_COMP,PP3)
	
	CONTENT_WEIGHT = {}
	NORM_CONTENT_WEIGHT = {}
	
	CONTENT_WEIGHT = compute_tfidf_NEW(content_count,count,PLACE)
	NORM_CONTENT_WEIGHT = set_weight(CONTENT_WEIGHT,LOWLIMIT,UPPERLIMIT)
	

	fp = codecs.open(parsefile,'r','utf-8')
	ft = codecs.open(ifname,'r','utf-8')

	TOTAL_COMP = 0
	TOTAL_SELECTED_COMP = 0
	PATH_COUNT = 0
	PATH_INDEX = 0
	
	DIC2SEQ = {}
	SEQ2DIC = {}
	for l in fp:
		wl = l.split('\t')
		if len(wl)==8:
                        seq = int(wl[0])
                        main_word = wl[1].strip(' #\t\n\r').lower()
                        word = wl[1].strip(' #\t\n\r').lower()
                        tag = wl[4].strip(' \t\n\r')
                        dep = wl[6].strip(' \t\n\r')
                        if dep=='_':
                                dep = int(wl[7].strip(' \t\n\r'))
                        else:
                                dep = int(wl[6])

                        if tag=='$':
                                s = word.strip(' \t\n\r')
                                Q = s
                                for x in RPL:
                                        Q = s.replace(x,'')
                                        s = Q
                                Q = s.lstrip('0')
                                s = Q
                                try:
                                	w = str(numToWord(int(s)))
                               		if len(w.split())>1: # like 67
                                        	w = s
                            	except Exception as e:
					w = s
                                word = w.lower()
                        elif tag=='N':
				try:
                                	w = lmtzr.lemmatize(word)
                                	word = w.lower()
				except Exception as e:
					pass
                        elif tag=='^':
				try:
                                	w = lmtzr.lemmatize(word)
                                	word = w.lower()
				except Exception as e:
					pass
				tag = 'N'
                        elif tag=='V':
                                try:
                                        w = Word(word.lower())
                                        x = w.lemmatize("v")
                                except Exception as e:
                                        x = word.lower()
                                word = x.lower()
                                #count+=1
                        else:
				pass

                        temp = [word,tag,dep,main_word]
                        dic[seq] = temp
			DIC2SEQ[word] = seq
			SEQ2DIC[seq] = word
		else:
			######################## Identify singleton component #############################################
			temp = dic.keys()
                        temp.sort()
                        G = nx.Graph()
                        for x in temp:
                                G.add_node(x)
                        for x in temp:
                                dep = dic[x][2]
                                if dep!=-1 and dep!=0 and dic[x][1] not in TAGREJECT:
                                        G.add_edge(dep,x)
                        temp = sorted(nx.connected_components(G), key = len, reverse=True)
			

			############################## Identify Start Components ##########################################
			TL = ft.readline().split('\t')
			Tweet_Length = int(TL[4])
			reject = set([])
			
			START_COMP = []
			KEYS = dic.keys()
			KEYS.sort()
			flag = 0
			for ky in KEYS:
				X = dic[ky]
				if X[2]!=-1 and flag==0:
					START_COMP.append(ky)
					flag=1
				else:
					prev_key = ky - 1
					next_key = ky + 1
					if prev_key in KEYS and next_key in KEYS:
						T1 = dic[prev_key]
						T2 = dic[next_key]
						if T1[2]==-1 and T2[2]==-1 and T1[0] in PUNC and T2[0] in PUNC:
							START_COMP.append(ky)

			COMP_SIZE = len(temp)
			TOTAL_COMP = TOTAL_COMP + COMP_SIZE

			if len(temp)>THR_COMP:
				for i in range(0,len(temp),1):
					comp = list(temp[i])
					if len(comp)<=1:
						if PLACE.__contains__(dic[comp[0]][0])==True:
							X = dic[comp[0]][0] + '_P'
						elif dic[comp[0]][1]=='N':
							X = dic[comp[0]][0] + '_CN'
						elif dic[comp[0]][1]=='^':
							X = dic[comp[0]][0] + '_CN'
						elif dic[comp[0]][1]=='$':
							X = dic[comp[0]][0] + '_S'
						else:
							X = dic[comp[0]][0] + '_' + dic[comp[0]][1]
						reject.add(X)
				
				if Tweet_Length>1:
					for i in range(0,len(START_COMP),1):
						try:
							if SEQ2DIC[START_COMP[i]+1].find(':')!=-1:
								temp = dic[START_COMP[i]]
								if PLACE.__contains__(temp[0])==True:
									X = temp[0] + '_P'
								elif temp[1]=='N':
									X = temp[0] + '_CN'
								elif temp[1]=='^':
									X = temp[0] + '_CN'
								elif temp[1]=='$':
									X = temp[0] + '_S'
								else:
									X = temp[0] + '_' + temp[1]
								reject.add(X)
						except:
							pass
				

			######################## CONTENT WORD EXTRACTION #################################################################

			content = set([])
			content_original = set([])
			temp = TL[2].split()
			for x in temp:
				x_0 = x.split('_')[0].strip(' \t\n\r')
				x_1 = x.split('_')[1].strip(' \t\n\r')
				if x_1=='PN':
					s = x_0 + '_CN'
					content_original.add(s)
					if s not in reject:
						content.add(s)
				else:
					content_original.add(x)
					if x not in reject:
						content.add(x)
			All = set([])
			temp = TL[3].split()
			for x in temp:
				x_0 = x.split('_')[0].strip(' \t\n\r')
				x_1 = x.split('_')[1].strip(' \t\n\r')
				if x_1=='PN':
					s = x_0 + '_CN'
					if s not in reject:
						All.add(s)
				else:
					if x not in reject:
						All.add(x)
				
			L = int(TL[4])
			
			######################### SET COUNT #################################################################


			for x in content:
				x10 = x.split('_')[0].strip(' \t\n\r')
				if notag_content_count.__contains__(x10)==True:
					v = notag_content_count[x10]
					v+=1
					notag_content_count[x10] = v
				else:
					notag_content_count[x10] = 1
			
			
			TSC_LANG = round(float(TL[5]),2)
			TSC_IMP = round(float(TL[6]),2)
			k = should_select(SCT,All)
			if k>=1:
				CT[index] = content
				SCT[index] = All
				T[index] = [TL[1].strip(' \t\n\r'),content,L,TSC_LANG,TSC_IMP,reject,content_original]
				LANG_SCORE.append(TSC_LANG)
				IMP_SCORE.append(TSC_IMP)
				TOTAL_SELECTED_COMP = TOTAL_SELECTED_COMP + COMP_SIZE
				index+=1
				PATH_INDEX+=1

			dic = {}
			DIC2SEQ = {}
			SEQ2DIC = {}
			count+=1
			PATH_COUNT+=1
       
	fp.close()
	ft.close()
	print('After path selection: ',count,index)
	
	PP1 = TOTAL_COMP + 4.0 - 4.0
	PP2 = PATH_COUNT + 4.0 - 4.0
	try:
		PP3 = round(PP2/PP1,4)
	except:
		PP3 = 0
	print('Component, Path count, Component/Path: ',TOTAL_COMP,PATH_COUNT,PP3)
	

	PP1 = TOTAL_SELECTED_COMP + 4.0 - 4.0
	PP2 = PATH_INDEX + 4.0 - 4.0
	try:
		PP3 = round(PP2/PP1,4)
	except:
		PP3 = 0
	print('Component, Path index, Component/Selected Path: ',TOTAL_SELECTED_COMP,PATH_INDEX,PP3)

	MOD_LANG_SCORE = set_list_weight(LANG_SCORE,0,1)
	MOD_IMP_SCORE = set_list_weight(IMP_SCORE,0,1)

	XC = []
	for x in MOD_IMP_SCORE:
		if x>=0.5:
			XC.append(x)
	AVG_IMP_SCORE = round(np.median(XC),2)
	print(AVG_IMP_SCORE)

	########################################### Update Tweet Set (topic to cluster) ########################################
	
	TW = {}
	path_taken = 0
	start_index = 0
	tweet_index = 0
	for i in range(0,index,1):
		v = T[i]
		content = v[1]
		mod_content = set([])
		for x in content:
			try:
				#if NORM_CONTENT_WEIGHT[x]>=0:
				mod_content.add(x)
			except:
				pass
		if i < raw_tweet_index:
			TW[tweet_index] = [v[0],mod_content,v[2],round(v[4],2)] #tweet, content words, length, confidence score
			tweet_index+=1
		else:
			TSC = round(MOD_IMP_SCORE[start_index],2)
			if TSC>=0.8:
				TW[tweet_index] = [v[0],mod_content,v[2],round(IMP_SCORE[start_index],2)]
				tweet_index+=1
				path_taken+=1
			start_index+=1

	print('Number of paths: ',path_taken)


	########################################## Summarize Tweets #############################################################

	L = len(TW.keys())
        tweet_cur_window = {}
        for i in range(0,L,1):
                temp = TW[i]
                tweet_cur_window[i] = [temp[0].strip(' \t\n\r'),int(temp[2]),temp[1],float(temp[3])] # tweet, length, content, score

        ofname = keyterm + '_Tweet_Path_Joint_Summary_' + date + '.txt'
        optimize(tweet_cur_window,CONTENT_WEIGHT,ofname,Ts,0.4,0.6)
        t1 = time.time()
        print('Summarization done: ',ofname,' ',t1-t0)

def compute_similarity(S1,S2):
	common = set(S1).intersection(set(S2))
	X = len(common) + 4.0 - 4.0
	Y = min(len(S1),len(S2)) + 4.0 - 4.0
	if Y==0:
		return 0
	Z = round(X/Y,4)
	return Z

def should_select(T,new):
        #y = len(new) + 4.0 - 4.0
        if len(new)==0:
                return 0
        for i in range(0,len(T),1):
                temp = T[i]
                common = set(temp).intersection(set(new))
                if len(common)==len(new):
                       return 0
        return 1

def optimize(tweet,con_weight,ofname,L,A1,A2):


        ################################ Extract Tweets and Content Words ##############################
        con_word = {}
	#sub_word = {}
        tweet_word = {}
        tweet_index = 1
        for  k,v in tweet.iteritems():
                set_of_words = v[2]
                for x in set_of_words:
                	if con_word.__contains__(x)==False:
                                if con_weight.__contains__(x)==True:
                                        p1 = round(con_weight[x],4)
                                else:
                                        p1 = 0.0
                                con_word[x] = p1 * WT2
                
                tweet_word[tweet_index] = [v[1],set_of_words,v[0],v[3]]  #Length of tweet, set of content words present in the tweet, set of subevents present in the tweet, tweet itself
                tweet_index+=1

        ############################### Make a List of Tweets ###########################################
        sen = tweet_word.keys()
        sen.sort()
        entities = con_word.keys()
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
                P += tweet_word[i+1][3] * sen_var[i] * WT1
                C1 += tweet_word[i+1][0] * sen_var[i]
                v = tweet_word[i+1][1] # Entities present in tweet i+1
                C = LinExpr()
                flag = 0
                for j in range(0,len(entities),1):
                        if entities[j] in v:
                                flag+=1
                                C += con_var[j]
                if flag>0:
                        counter+=1
                        m.addConstr(C, GRB.GREATER_EQUAL, flag * sen_var[i], "c%d" % (counter))
                

        for i in range(0,len(entities),1):
                P += con_word[entities[i]] * con_var[i]
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

def compute_tfidf_NEW(word,tweet_count,PLACE):
        score = {}
        #discard = ['pray','prayer','updates','pls','please','hope','hoping','breaking','news','flash','update','tweet','pm','a/c','v/','w/o','watch','photo','video','picture','screen','pics','latest','plz','rt','mt','follow','tv','pic','-mag','cc','-please','soul','hoax','a/n','utc','some','something','ist','afr','guru','image','images']

        discard = []
        #THR = int(round(math.log10(tweet_count),0))
        THR = 5
        N = tweet_count + 4.0 - 4.0
        for k,v in word.iteritems():
                D = k.split('_')
                D_w = D[0].strip(' \t\n\r')
                D_t = D[1].strip(' \t\n\r')
                if D_w not in discard:
                        tf = v
                        w = 1 + math.log(tf,2)
                        #w = tf
                        df = v + 4.0 - 4.0
                        #N = tweet_count + 4.0 - 4.0
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

def numToWord(number):
        word = []
        if number < 0 or number > 999999:
                return number
                # raise ValueError("You must type a number between 0 and 999999")
        ones = ["","one","two","three","four","five","six","seven","eight","nine","ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen","eighteen","nineteen"]
        if number == 0: return "zero"
        if number > 9 and number < 20:
                return ones[number]
        tens = ["","ten","twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety"]
        word.append(ones[int(str(number)[-1])])
        if number >= 10:
                word.append(tens[int(str(number)[-2])])
        if number >= 100:
                word.append("hundred")
                word.append(ones[int(str(number)[-3])])
        if number >= 1000 and number < 1000000:
                word.append("thousand")
                word.append(numToWord(int(str(number)[:-3])))
        for i,value in enumerate(word):
                if value == '':
                        word.pop(i)
        return ' '.join(word[::-1])


def main():
	try:
		_, mainname, mainparsefile, ifname, parsefile, placefile, keyterm, date, Ts = sys.argv
	except Exception as e:
		print(e)
		sys.exit(0)
	compute_summary(mainname,mainparsefile,ifname,parsefile,placefile,keyterm,date,int(Ts))
	print('Koustav Done')

if __name__=='__main__':
	main()
