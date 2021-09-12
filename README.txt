This repo contains codes to jointly summarize tweets and paths using an ILP based unsupervised approach. It also prunes noisy contents. 

###########################################################################################################################################

Step 1: Extract concepts from raw AIDR classified tweets
python tweet_concept_extraction.py /demo/infrastructure_20150425.txt place/nepal_place.txt infrastructure_concept_20150425.txt

Step 2: Generate 1000 word summary using COWTS [Rudra et al. CIKM 2015]
python NCOWTS.py /demo/infrastructure_concept_20150425.txt place/nepal_place.txt infrastructure 20150425 1000

Step 3: Tag above generated 1000 word summary
python tag_top_1000.py demo/infrastructure_ICOWTS_20150425.txt demo/infrastructure_icowts_tagged_20150425.txt

Step 4: Generate Path from the tagged data
	Check the code at https://github.com/krudra/Bigram_Path_Generation
	
Step 5: Extract concepts from paths
python path_concept_extraction.py demo/infrastructure_icowts_tagged_20150425.txt_paths place/nepal_place.txt infrastructure_path_concept_20150425.txt

Step 6: Combine tweets and paths [having confidence >= 0.80], remove noisy components to generate final summary
python tweet_path_joint_summarization.py demo/infrastructure_concept_20150425.txt demo/infrastructure_20150425.txt.predict demo/infrastructure_path_concept_20150425.txt demo/infrastructure_path_20150425.txt.predict place/nepal_place.txt infrastructure 20150425 200

################################################################################################################################################

NOTE:
	A. Set the paths accordingly
	B. In all the codes set the Twitter Tagger path accordingly
	C. infrastructure_20150425.txt.predict is parse file. You have to generate it separately using Twitter parser [Kong et al., EMNLP 2014]
	D. Demo directory contains some sample files just to demonstrate format of files passing from one module to the next one. To generate summaries kindly apply the methods over original dataset.
	E. For the path generation module [TwitterSumm], kindly drop a mail.

Dataset: http://www.cnergres.iitkgp.ac.in/subeventsummarizer/dataset.html
Queries: koustav.rudra@cse.iitkgp.ernet.in  [Path generation module was originally developed by Siddhartha Banerjee in our prior work in Hypertext'16. If you have any queries regarding path generation step kindly forward queries to sbanerjee@ist.psu.edu]
