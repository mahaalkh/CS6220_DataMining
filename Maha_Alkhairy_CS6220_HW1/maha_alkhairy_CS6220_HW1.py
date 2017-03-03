import numpy as np
import matplotlib.pyplot as plt
import re 
from collections import Counter

### for parsing of data
PAPER_ID = "#index"  
PAPER_TITLE = "#*"
PAPER_TITLE_re = "#\*"

## seperated by semicolons 
AUTHORS = "#@" 

YEAR = "#t" 
VENUE = "#c" 
REF = "#%" 
ABSTRACT = "#!" 
reg_match = '^{}(.*)'

## contains all the information the data file gives in the format: 
## {id_paper: {p_title: String, authors: [String, ...], year: number,
##				 venue: String , refs = [id_paper, ...], ...}
PUBLICATION = {}

### related papers to the author
## {"author_name": [p_id, ...], ...}
AUTHS_P = {}

### number of related papers to the author
## {"author_name": int, ...}
AUTHS_P_num = {}

### related papers to the venue
## {"venue_name": [p_id, ...], ...}
VEN_P = {}

### number of related papers for each venue
## {"venue_name": int, ...}
VEN_P_num = {}


### the impact factor of each venue 
## {"venue_name": int, ...}
VENUE_IMPACT = {}

### the impact factor of each venue with 10 or more publications
## {"venue_name": int, ...}
VENUE_IMPACT_10 = {}


### related papers to the year
## {year: [p_id, ...], ...}
YEAR_P = {}

### average number of references per year
## {year: int}
YEAR_P_refs_num = {}

### average number of citations per year
## {year: int, ...}
YEAR_P_cite_num = {}

### number of citations per publication 
## {publication: #citations, ...}
CITE_P_num = {}

### references per publication
## {publication: [ref, ...] , ...}
REF_P = {}

### number of references per publication
## {publication: #refs, ...}
REF_P_num = {}


### the cumulative list of references for all publications
REFERENCES = []


def parse_data(filename = "arnetminer/AP_train.txt"): 
	"""
	reads the lines from the file and saves the data in the required format
	:filename: String 
	:return: None 
	:effect: saves the data in the global variable (Map) PUBLICATION 
	"""
	global PUBLICATION, AUTHS_P, VEN_P, YEAR_P, REFERENCES
	fd = open(filename, 'r', encoding = "UTF-8")
	current_p_id = None 
	for line in fd:
		line.strip()
		if line.startswith(PAPER_ID): 
			# print l
			p_id_t = re.search(reg_match.format(PAPER_ID), line).group(1).strip()
			p_id = int(p_id_t)
			current_p_id = p_id
			current_p_id_t = p_id_t
			if p_id in PUBLICATION.keys(): 
				print("NOT SUPPOSED TO BE HERE")
			else: 
				PUBLICATION[p_id] = {"p_title": None, 
									 "authors": [], 
									 "year": None, 
									 "venue": None, 
									 "refs": []}
		## the title  
		if line.startswith(PAPER_TITLE): 
			title = re.search(reg_match.format(PAPER_TITLE_re), line).group(1).strip()
			PUBLICATION[current_p_id]["p_title"] = title.strip()
		## the authors 
		if line.startswith(AUTHORS):
			authors = re.search(reg_match.format(AUTHORS), line).group(1).strip()
			auths = list(map(lambda ath: ath.strip(), authors.split(';')))
			for a in auths:
				if a != "":
					if a in AUTHS_P.keys():
						AUTHS_P[a].append(current_p_id)
					else:
						AUTHS_P[a] = [current_p_id]
			PUBLICATION[current_p_id]["authors"] = [at for at in auths if at != ""]

		## the yop 
		if line.startswith(YEAR):
			y = re.search(reg_match.format(YEAR), line).group(1).strip()
			if y != '':
				PUBLICATION[current_p_id]["year"] = int(y)
				if int(y) in YEAR_P.keys(): 
					YEAR_P[int(y)].append(current_p_id)
				else: 
					YEAR_P[int(y)] = [current_p_id]
		## the venue 
		if line.startswith(VENUE): 
			venue =  re.search(reg_match.format(VENUE), line).group(1).strip()
			if (venue != "" and current_p_id_t != ""): 
				PUBLICATION[current_p_id]["venue"] = venue
				if venue in VEN_P.keys(): 
					VEN_P[venue].append(current_p_id)
				else: 
					VEN_P[venue] = [current_p_id]
		## the refs 
		if line.startswith(REF): 
			ref = re.search(reg_match.format(REF), line).group(1).strip()
			PUBLICATION[current_p_id]["refs"].append(int(ref))
			REFERENCES.append(int(ref))

####################################################
################# QUESTION 1 #######################
### 1.a. : 
def count_authors(): 
	"""
	counts the number of distinct authors
	:return: int  
	"""
	return len(AUTHS_P.keys())

def count_venues(): 
	"""
	counts the number of distinct 
	:return: int
	"""
	return len(VEN_P.keys())

def count_publications(): 
	"""
	counts the number of publications
	:return: int
	"""
	return len(PUBLICATION.keys())




####################################################
################# QUESTION 2 -- for mean and std ... calculations #######################
def number_pub_venue(): 
	"""
	counts the number of publications per venue 
	:effect: saves the count in the global dictinory VEN_P_num
	"""
	global VEN_P_num
	VEN_P_num = dict(map(lambda p: (p[0], len(p[1])), VEN_P.items()))

def number_pub_auth(): 
	"""
	counts the number of publications per venue 
	:effect: saves the count in the global dictinory AUTH_P_num
	"""
	global AUTHS_P_num
	AUTHS_P_num = dict(map(lambda p: (p[0], len(p[1])), AUTHS_P.items()))

def get_mean_of_pub_auths(): 
	"""
	gets the mean of the number of publications per author
	:return: int
	"""
	return np.mean(np.array(list(AUTHS_P_num.values())))

def get_std_of_pub_auths(): 
	"""
	gets the mean of the number of publications per author
	:return: int
	"""
	return np.std(np.array(list(AUTHS_P_num.values())))

def get_quantile_of_pub_auths(): 
	"""
	gets the Q1, Q2, Q3, Q4 of the number of publications per author
	:return: [float, float, float, float]
	"""
	l = np.array(sorted(list(AUTHS_P_num.values())))
	return np.percentile(l, np.arange(0, 100, 25))

def get_mean_of_pub_vens(): 
	"""
	gets the mean of the number of publications per author
	:return: int
	"""
	return np.mean(np.array(list(VEN_P_num.values())))

def get_std_of_pub_vens(): 
	"""
	gets the mean of the number of publications per author
	:return: int
	"""
	return np.std(np.array(list(VEN_P_num.values())))

def get_median_and_Q1_Q3_of_pub_vens(): 
	"""
	gets the mean of the number of publications per author
	:return: [float, float, float, float]
	"""
	l = np.array(sorted(list(VEN_P_num.values())))
	## second quantile is the mean
	return np.percentile(l, np.arange(0, 100, 25))

def get_pub_largest_num_ven(): 
	"""
	gets the name of the venue with the largest number of publications
	:return: String
	"""
	return max(VEN_P_num, key=VEN_P_num.get)

###################################################
################# QUESTION 3 -- getting the references and citations numbers #######################
def number_ref():
	"""
	computes the number of references per publication 
	:effect: fills REF_P_num and REF_P
	"""
	global REF_P_num, REF_P
	REF_P_num = dict(map(lambda p: (p[0], len(p[1]['refs'])), PUBLICATION.items()))
	REF_P = dict(map(lambda p: (p[0], p[1]['refs']), PUBLICATION.items()))


def number_cite(): 
	"""
	computes the number of citations per publication 
	:effect: fills CITE_P_num
	"""
	global CITE_P_num 
	CITE_P_num =  Counter(REFERENCES)


def get_quantile_of_number_of_citations_venue(cites_ven): 
	"""
	gets the mean of the number of citations for the publications in the  venue
	:return: [float, float, float, float]
	"""
	l = np.array(sorted(cites_ven))
	return np.percentile(l, np.arange(0, 100, 25))

def get_mean_of_number_of_citations_venue(cites_ven): 
	"""
	gets the mean of the number of citations for the publications in the  venue
	:return: int
	"""
	return np.mean(np.array(cites_ven))

	
def get_pub_largest_num_refs(): 
	"""
	returns the publication index and title with the most references
	:return: int
	"""
	index_max_pub = max(REF_P_num, key=REF_P_num.get)
	return (index_max_pub,  PUBLICATION[index_max_pub]["p_title"])

def get_pub_largest_num_citations(): 
	"""
	returns the publication index and title with the most citations
	:return: int
	"""
	index_max_pub = max(CITE_P_num, key=CITE_P_num.get)
	return (int(index_max_pub),  PUBLICATION[int(index_max_pub)]["p_title"])


def calculate_venue_impact_factor(): 
	"""
	gets the impact factor for each venue
	:effect: fills in VENUE_IMPACT
	"""
	global VENUE_IMPACT

	for v, pubs in VEN_P.items(): 
		cites_num_pubs = []
		for p in pubs:
			if p in CITE_P_num: 
				 cites_num_pubs.append(CITE_P_num[p])
		VENUE_IMPACT[v] = sum(cites_num_pubs) / len(pubs)


def calculate_venue_impact_factor_10(): 
	"""
	gets the impact factor for each venue 
	:effect: fills in VENUE_IMPACT_10
	"""
	global VENUE_IMPACT_10
	
	for v, pubs in VEN_P.items(): 
		cites_num_pubs = []
		if len(pubs) >= 10:
			for p in pubs:
				if p in CITE_P_num: 
					cites_num_pubs.append(CITE_P_num[p])
			VENUE_IMPACT_10[v] = sum(cites_num_pubs) / len(pubs)


def get_venue_with_largest_impact_factor(ten_more = False): 
	"""
	gets the venue name with the largest impact factor
	:return: string
	"""
	m_ven = None
	if ten_more:
		m_ven = max(VENUE_IMPACT_10, key=VENUE_IMPACT_10.get)
	else: 
		m_ven = max(VENUE_IMPACT, key=VENUE_IMPACT.get)
	return m_ven

##########################
def get_citation_counts_of_venue(venue_name):
	"""
	gets the citation counts for each publication in the venue
	:return: [int, ...]
	"""
	pubs = VEN_P[venue_name]
	cites_num_pubs = []
	for p in pubs:
		if p in CITE_P_num:
			cites_num_pubs.append(CITE_P_num[p])
		else: 
			cites_num_pubs.append(0)

	return cites_num_pubs

#########################

def calculate_year_ref_avg(): 
	"""
	calculates the average number of references per year
	:effect: fills in YEAR_P_refs_num
	"""
	global YEAR_P_refs_num
	for y, pubs in YEAR_P.items():
		refs_num_pubs = [] 
		for p in pubs:
			if p in REF_P_num:
				refs_num_pubs.append(REF_P_num[p])
		YEAR_P_refs_num[y] = sum(refs_num_pubs)/len(pubs)

def calculate_year_cite_avg(): 
	"""
	calculates the average number of citations per year
	:effect: fills in YEAR_P_cite_num
	"""
	global YEAR_P_cite_num
	for y, pubs in YEAR_P.items(): 
		cite_num_pubs = [] 
		for p in pubs:
			if p in CITE_P_num:
				cite_num_pubs.append(CITE_P_num[p])
		YEAR_P_cite_num[y] = sum(cite_num_pubs)/len(pubs)

############################### PLOTTING ########################

def histogram_graph(x, number_bins, x_label, y_label, title, log_scale=False):
	"""
	graphs a histogram 
	:x: [int, ...] 
	:number_bins: int
	:x_label: string
	:y_label: string
	:title: string 
	:log_scale: Boolean 
	"""
	n, bins, patches = plt.hist(x, number_bins, log=log_scale)
	plt.xlabel(x_label) 
	plt.ylabel(y_label)
	plt.title(title)
	plt.show()   


def regular_graph(x, y, x_label, y_label, title):
	"""
	plots a line graph
	:x: [int, ...] 
	:y: [int, ...] 
	:x_label: string
	:y_label: string
	:title: string 
	"""
	plt.plot(x, y, linewidth = 2)
	plt.xlabel(x_label) 
	plt.ylabel(y_label)
	plt.title(title)
	plt.show()

###########################

## small helper: 
def get_x_list(d): 
	"""
	gets the x values for the histogram
	:return: [int, ...]
	"""
	x = [i for v, i in d.items()]
	return x

############################ TO ANSWER ######################

def question1_a(): 
	"""
	calls the methods to answer part a in question 1
	"""
	print("number of distinct authors: ", count_authors())
	print("number of distinct venues: " , count_venues())
	print("number of distinct publications: ", count_publications())


def question2_stats():
	"""
	gets the statistics for the second question
	""" 
	number_pub_auth()
	number_pub_venue()
	

	print("Venue, maximum number of publications: ", max(VEN_P_num.values()))	
	print("Author, maximum number of publications: ", max(AUTHS_P_num.values()))


	print("mean pubs auths", get_mean_of_pub_auths())
	print("std pubs auths", get_std_of_pub_auths())
	print("quantile pubs auths", get_quantile_of_pub_auths())


	print("mean pubs vens", get_mean_of_pub_vens())
	print("std pubs vens", get_std_of_pub_vens())
	print("quantile pubs vens", get_median_and_Q1_Q3_of_pub_vens())

	print("venue with largest number of publications", get_pub_largest_num_ven())


def question_2_histograms():
	"""
	plots the histograms for the second question 
	1) histogram for #authors to #publications
	2) histogram for #venues to #publications 
	"""

	x_author = get_x_list(AUTHS_P_num)
	histogram_graph(x_author,
					 50,
					 'Number of Publications', 
					 'Number of Authors (log scale)', 
					 "Authors and Publications", 
					 log_scale=True)

	x_venue = get_x_list(VEN_P_num)
	histogram_graph(x_venue,
					 50, 
					 'Number of Publications', 
					 'Number of Venues', 
					 "Venues and Publications", 
					 log_scale=True)


def question_3_stats(): 
	"""
	gets the statistics for the third question
	"""
	number_ref() 
	number_cite()

	print("number of publications with citations: ", len(CITE_P_num.values()))

	print("publication with largest number of refs: ", get_pub_largest_num_refs())
	print("publication with largest number of citations: ", get_pub_largest_num_citations())

	calculate_venue_impact_factor()
	calculate_venue_impact_factor_10()

	venue_largest_impact = get_venue_with_largest_impact_factor()
	venue_largest_impact_10 =  get_venue_with_largest_impact_factor(ten_more = True)

	ven_cites_largest_10 = get_citation_counts_of_venue(venue_largest_impact_10)

	print("venue with largest apparent impact, impact factor: ", venue_largest_impact, ", ", VENUE_IMPACT[venue_largest_impact]) 
	print("venue with largest apparent impact and with 10 or more publications, impact factor: ", venue_largest_impact_10, ", ", VENUE_IMPACT_10[venue_largest_impact_10]) 


	print("the citation counts of all the publications from the venue (with atleast 10 publications) with the largest impact factor: ", 
    get_citation_counts_of_venue(venue_largest_impact_10)) 

	print("mean number of citations for the venue with the largest impact and 10 or more publications: ",
	 get_mean_of_number_of_citations_venue(ven_cites_largest_10))
	print("median number of citations for the venue with the largest impact and 10 or more publications: ",
	 get_quantile_of_number_of_citations_venue(ven_cites_largest_10))
	

	print("the citation counts of all the publications from the venue with the largest impact factor: ", 
	    get_citation_counts_of_venue(venue_largest_impact))


	calculate_year_ref_avg()
	calculate_year_cite_avg()

def question_3_plots(): 
	"""
	draws the histogram plots for
	1) ref to pubs
	2) cites to pubs
	3) impact factor to number of venues with that impact factor 
	4) impact factor to number of venues with that impact factor with 10 or more pubs


	and the line plots for 
	1) year to average number of citations 
	2) year to average number of references

	"""

	##1) 
	x_refs = get_x_list(REF_P_num)
	histogram_graph(x_refs,
					 50, 
					 'Number of References', 
					 'Number of Publications', 
					 "Publications and References (log scale)", 
					 log_scale=True)

	##2) 
	x_cites = get_x_list(CITE_P_num)
	histogram_graph(x_cites,
					 50, 
					 'Number of Citations', 
					 'Number of Publications', 
					 "Publications and Citations (log scale)", 
					 log_scale=True)

	##3) 
	x_venue_impact = get_x_list(VENUE_IMPACT)
	histogram_graph(x_venue_impact,
					 50, 
					 'Venue Impact', 
					 'Number of Venues', 
					 "Venue impact and Venue (log scale)", 
					 log_scale=True)

	##4) 
	x_venue_impact_10 = get_x_list(VENUE_IMPACT_10)
	histogram_graph(x_venue_impact_10,
					 50, 
					 'Venue impact',  
					 'Number of Venues (with 10 or more publications)', 
					 "Venue impact and Venue (limited to the ones with 10 publications) (log scale)", 
					 log_scale=True)
	#1) 

	# print(YEAR_P_cite_num)
	regular_graph(list(YEAR_P_cite_num.keys()), 
				  list(YEAR_P_cite_num.values()), 
				  "Year", 
				  "Average Number of Citations", 
				  "Trend of number of citations per year")

	#2) 
	# print(YEAR_P_refs_num)
	regular_graph(list(YEAR_P_refs_num.keys()), 
				  list(YEAR_P_refs_num.values()), 
				  "Year", 
				  "Average Number of References", 
				  "Trend of number of references per year")



def main(filename="arnetminer/AP_train.txt"):
	"""
	calls the functions in the correct order 
	"""

	### getting the data 
	print('parsing the data')
	parse_data(filename=filename)
	print("Done parsing the data")

	print("1.a \t \n")
	question1_a()

	print("1.b:\nthe number of publications is likely to be correct because they are indexed uniquely.\nthe number of distinct venues could be off because the venue could have more that one way of being referred to. \nthe number of distinct author's could be off because the author could have be referred to by more than one different way in the publications. \n1.c: the authors could have be referred to by more than one different way in the publications.  ")

	question2_stats()

	print ("2.b: ", "\n", "median = 1, mean = 3.2917856029, third quartile = 1 \n", 
		"so 75% of the data has the same value. ", 
	  "since the mean is higher than the median, that means that the data is skewed and that the difference between the max and min values is high. ", 
	  "the standard deviation is 8 which means the data is spread out. ")

	question_2_histograms()

	question_3_stats()

	question_3_plots()

	print("3.d: \n",  
	"The impact factor (mean number of citation counts = 8.32) and the median = 1 ", 
	"the mean is greater than the median which means that the max of the number of citations is much greater than the minimum number of citations") 

	print("	3.e: \n In the trend for the number of references per year, ",
	 "it is low in the older years and peaks in year 2010. \n", 
	 "In the trend for the number of citations,  it is higher in the older", 
	 " years and is decreasing in the later years and peaks in year 1945.")  

main()
 




