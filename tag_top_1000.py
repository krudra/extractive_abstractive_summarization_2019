import sys
import os

Tagger_Path = ''

command = Tagger_Path + './runTagger.sh --output-format conll ' + sys.argv[1] + ' > tag.txt'
os.system(command)

fp = open('tag.txt','r')
fo = open(sys.argv[2],'w')
s = ''
for l in fp:
	wl = l.split('\t')
	if len(wl)>1:
		word = wl[0].strip(' \t\n\r').lower()
		tag = wl[1].strip(' \t\n\r')
		p = word + '/' + tag
		s = s + p + ' '
	else:
		fo.write(s.strip(' \t\n\r'))
		fo.write('\n')
		s = ''
fp.close()
fo.close()
