# -*- coding: utf-8-*-
import sys    
reload(sys)    
sys.setdefaultencoding('utf8')
import os
import sys
import pymongo
import argparse
from random import choice
import stem
import os
import time 
from timeout import timeout
import json

from sklearn import datasets
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import re
from langdetect import detect
from langdetect import detect_langs
import sklearn.neighbors
import shutil

MONGO_HOST = "10.108.98.178"
ONE_PAGE_DOMAIN = "./onePage/"
ONE_SERVER_PAGE = set(["it works", "Apache2 Debian Default Page: It works", 
	"Welcome to nginx!", 
	"Apache2 Ubuntu Default Page: It works",
	"Welcome to nginx on Debian!"])

WANT_CLASS_SET = set(['Whistleblowing', 'Hacking', 'Drugs', 'Email and Messaging', 
'Financial_Services', 'Hosting', 'Blog', 'Books', 'Erotica', 'Introduction Points', 
'Forums', 'Commercial Services'])


new_map = {
	"Erotica" : set([]),
}



##过滤HTML中的标签
#将HTML中标签等信息去掉
#@param htmlstr HTML字符串.
def filter_tag(htmlstr):
    #先过滤CDATA
    re_cdata=re.compile('//<!\[CDATA\[[^>]*//\]\]>',re.I) #匹配CDATA
    re_script=re.compile('<\s*script[^>]*>[^<]*<\s*/\s*script\s*>',re.I)#Script
    re_style=re.compile('<\s*style[^>]*>[^<]*<\s*/\s*style\s*>',re.I)#style
    re_br=re.compile('<br\s*?/?>')#处理换行
    re_h=re.compile('</?\w+[^>]*>')#HTML标签
    re_comment=re.compile('<!--[^>]*-->')#HTML注释
    s=re_cdata.sub('',htmlstr)#去掉CDATA
    s=re_script.sub('',s) #去掉SCRIPT
    s=re_style.sub('',s)#去掉style
    s=re_br.sub('\n',s)#将br转换为换行
    s=re_h.sub('',s) #去掉HTML 标签
    s=re_comment.sub('',s)#去掉HTML注释
    #去掉多余的空行
    blank_line=re.compile('\n+')
    s=blank_line.sub('\n',s)
    s=replaceCharEntity(s)#替换实体
    return s

##替换常用HTML字符实体.
#使用正常的字符替换HTML中特殊的字符实体.
#你可以添加新的实体字符到CHAR_ENTITIES中,处理更多HTML字符实体.
#@param htmlstr HTML字符串.
def replaceCharEntity(htmlstr):
    CHAR_ENTITIES={'nbsp':' ','160':' ',
                'lt':'<','60':'<',
                'gt':'>','62':'>',
                'amp':'&','38':'&',
                'quot':'"','34':'"',}

    re_charEntity=re.compile(r'&#?(?P<name>\w+);')
    sz=re_charEntity.search(htmlstr)
    while sz:
        entity=sz.group()#entity全称，如&gt;
        key=sz.group('name')#去除&;后entity,如&gt;为gt
        try:
            htmlstr=re_charEntity.sub(CHAR_ENTITIES[key],htmlstr,1)
            sz=re_charEntity.search(htmlstr)
        except KeyError:
            #以空串代替
            htmlstr=re_charEntity.sub('',htmlstr,1)
            sz=re_charEntity.search(htmlstr)
    return htmlstr

def onion1stat():
    f = open('./onionclassify_final_code/onion1.log', 'r')
    wiki_onionset = set()
    for line in f:
        line_arr = line[:-1].split('\t')
        type_in = line_arr[1].strip()
        onion_addr = line_arr[0].strip()
        wiki_onionset.add(onion_addr)
    return wiki_onionset




def add_practise_data():


	type_domain_set = {
		"Commercial Services" : set(["counterfn2x4wqga", "applei7nkshrsnih", "3dbr5t4nfgdsidus",
									"cardezg5f3arbt53", "bitstorenctdwhmo", "cityzenp4d2eytjh",
									"2kka4f23atzonzhq", "armoryohajjhou5m"]),

		"Financial_Services" : set(["btctrdocivn3rsh5.onion", "5f7dxxhqch4mzluw", "2222222nvcgjoga7",
									"coincloud2u2lxwf", "bitcloak43blmhmn", "cashouta6stry43f", "2222222l6oxgijgz"]),

		"Drugs" : set(["bakeryn4t2zyxmrx", "cannabi4ewmalq3g", "22dnhlkgj6b57xo4", "amazipq53xyj333a",
						"bitphar76n5t3qag", "24lw3mq3cpmvvy4p"]),
		"Email and Messaging" : set(["altaddresswcxlld", "grrmailb3fxpjbwm", "eludemaillhqfkh5", "bitmailendavkbec", 
			           "mail2tor6fez4hmd"]),
		"Hosting" : set(["3nuwa555bojyptrb", "alphaimgekptibai", "3kqpypputjn2dhpp",
					     "gadmbvhi5i44wlif", "teenxxxbtl7wsllp", "dhosting4okcs22v"]),
		"Hacking" : set(["2ogmrlfzdthnwkez", "m3pwzf3vsichzhvh", "renthackyzogj4b4",
						 "renthackyzogj4b4"]),
		"Erotica" : set(["porngwjr2flqjgfq", "deeppornx5rnxn5j", "wedopornbqnifksn", 
						"younggjg2mmeljbs", "cfwl3urfcsml22hb", "destroplh4zwowdk"]),
		"Whistleblowing" : set(["secrdrop5wyphb5x"])
	}
	path = './oniondataset'
	for type_info in type_domain_set:
		domain_set = type_domain_set[type_info]
		for domain in domain_set:
			old_path = os.path.join('./to_class_onion', domain + '.onion')
			print old_path
			new_file_path = os.path.join(path, type_info, domain + '.onion')
			if not os.path.exists(new_file_path):
				# continue
				print new_file_path
				shutil.copyfile(old_path, new_file_path)

		# if os.path.exists()










class MongoIns():
	def __init__(self):
		need_mongo = 1
		if need_mongo == 1:
			self.client = pymongo.MongoClient(MONGO_HOST,27117)
			self.oniondata = self.client['oniondata']
			self.document = self.oniondata['document']
			self.link = self.oniondata['link']
			self.pagerank = self.oniondata['pagerank']

	"""
	author: yinshuai
	function: 
	"""


	def init_pagerank_col(self):
		link_set = set()
		for info in self.link.find():
			source = info['source']
			target = info['target']
			link_set.add(source)
			link_set.add(target)
		for link in link_set:
			self.pagerank.insert({"link" : link, "score" : 1.0})

	def remove_target(self):
		url = "http://dmzwvie2gmtwszof.onion"
		self.pagerank.find()

	def cal_pagerank(self, iterations = 20):
		#init_pagerank_col()
		for i in range(iterations):
			for link_url in self.pagerank.find():
				pr = 0.15
				url = link_url['link']
				cursor = self.link.find({"target" : url}).distinct("source")
				for distinct_source_url in cursor:
					#get the source pagerank value
					source_score_info = self.pagerank.find_one({"link" : distinct_source_url}, 
										{"score" : 1, "_id" : 0})
					source_score = source_score_info['score']
					#get the sum link count via the source 
					source_link_to_count = self.link.find({"source" : distinct_source_url}).count()
					pr += 0.85 * (source_score / source_link_to_count)
				self.pagerank.update({"link" : url}, {"$set" : {"score" : pr}})





	def write_doc(self):
		f = open("./onedomainpage.log", 'r')
		path = './to_class_onion'
		if not os.path.exists(path):
			os.makedirs(path)
		one_page_domain = set()
		for line in f:
			domain = line[:-1]
			one_page_domain.add(domain)
		distinct_domain = self.document.distinct("domain")
		for domain in distinct_domain:
			if domain in one_page_domain:
				pass
			else:
				doc_domain_list = self.document.find({"domain" : domain}).limit(40)#only fetch 40 pages
				doc_all = ""
				for content in doc_domain_list:
					title =  content['doc'][0]['title'].strip()
					doc_all += title
					doc_all += "\n"
					doc = filter_tag(content['doc'][0]['content'].strip())
					doc_all += doc
				try:
					language = detect(doc_all)
					if language != 'en':
						continue
					file = path + '/' + domain
					f = open(file, 'w')
					f.write(doc_all)
					f.close()
					print f 
					

				except Exception,e :
					print e 
					print 'except found'
					continue



	def write_single_domain_to_log(self):
		if not os.path.exists(ONE_PAGE_DOMAIN):
			os.makedirs(ONE_PAGE_DOMAIN)
		f = open("./onedomainpage.log", 'w')
		distinct_domain = self.document.distinct("domain")
		for domain in distinct_domain:
			cnt = self.document.find({"domain" : domain}).count()
			if cnt == 1:
				# f = open(ONE_PAGE_DOMAIN + '/' + domain, 'w')
				doc = self.document.find({"domain" : domain}, {"doc" : 1, "_id" : 0}).limit(1)
				for content in doc:
					title =  content['doc'][0]['title'].strip()
					print title#via title to judge welcome page 
					if title in ONE_SERVER_PAGE:
						f.write(domain + '\n')
					else:
						print "heheh"
					




def test_classifier_LAJI(X, y, clf, test_size=0.4, y_names=None, confusion=False):
    # train-test split
    print 'test size is: %2.0f%%' % (test_size * 100)
    X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X, y, test_size=test_size)

    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)

    if not confusion:
        print colored('Classification report:', 'magenta', attrs=['bold'])
        print sklearn.metrics.classification_report(y_test, y_predicted, target_names=y_names)
    else:
        print colored('Confusion Matrix:', 'magenta', attrs=['bold'])
        print sklearn.metrics.confusion_matrix(y_test, y_predicted)



def judge_length_of_content():
	ignore_domain_path = ret_ignore_domain()
	folder = './to_class_onion'
	for file in os.listdir(folder):
		if file in ignore_domain_path:
			continue
		file_path = os.path.join(folder, file)
		content = open(file_path, 'r').read()
		temp = open('./temp.log', 'w')
		if len(content) > 60 and len(content) < 70: 
			print file
			print content
			temp.write(file + '\n')
			temp.write(content + '\n')
			temp.write('---------' + '\n\n\n\n\n\n\n')



def if_ignore_non_content_domain():
	folder = './to_class_onion'


def ret_ignore_domain():
	ignore_domain_path = './ignoredomain'
	ignore_domain_set = set()
	for file in os.listdir(ignore_domain_path):
		file_path = os.path.join(ignore_domain_path, file)
		for line in open(file_path, 'r').readlines():
			ignore_domain_set.add(line[:-1])
	print "the length of ignore domain is %s " % (len(ignore_domain_set))
	return ignore_domain_set


def classify_doc():
	#yinshuai there, i do not want to classify the domain like "it works and bad hosting "
	ignore_domain_set = ret_ignore_domain()
	oniondata_learn = datasets.load_files("oniondataset")
	# twenty_test = datasets.load_files("data/20news-bydate/20news-bydate-test")
	print len(oniondata_learn.target_names),len(oniondata_learn.data)
	count_vect = CountVectorizer(stop_words="english",decode_error='ignore')
	X_train_counts = count_vect.fit_transform(oniondata_learn.data)
	print X_train_counts.shape

	path = './to_class_onion'
	docs_new = []
	onion_new = []

	for file in os.listdir(path):
		if file in ignore_domain_set:
			continue
		file_path = os.path.join(path, file)
		content = open(file_path, 'r').read().strip()
		if len(content) < 60:
			continue 
		docs_new.append(content)
		onion_new.append(file)
		if len(docs_new) == 2000:
			break


	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
	print X_train_tfidf.shape

	# clf = MultinomialNB().fit(X_train_tfidf, oniondata_learn.target)
	clf = LinearSVC().fit(X_train_tfidf, oniondata_learn.target)
	weights = 'distance'
	n_neighbors = 4
	# clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
	# docs_new = ['God is love','OpenGL on the GPU is fast', 'fake visa do you want to buy', 'drugs bitcoin buy']
	# docs_new = []
	# docs_new.append("Matrix TrilogyNEW: 30 DAYS storage time! image upload Enter the matrix Bitcoin Donations Welcome! Please help us to keep this service alive. 18p1JK7GGpHadf8StM1icfRvrufsh2efyT")
	# docs_new.append("Virtual y PBX  para tu Negocio.")
	X_new_counts = count_vect.transform(docs_new)
	X_new_tfidf = tfidf_transformer.transform(X_new_counts)
	predicted = clf.predict(X_new_tfidf)

	index = 0
	for doc, category in zip(docs_new, predicted):
		# print   onion_new[index], oniondata_learn.target_names[category]
		if oniondata_learn.target_names[category] == "Hacking":
			print onion_new[index]
		index += 1
	    # print("%r => %s") %(doc, oniondata_learn.target_names[category])






# classify_doc()


mongo_ins = MongoIns()
# mo
# mongo_ins.write_doc()
# mongo_ins.cal_pagerank()


# add_practise_data()


classify_doc()	


# judge_length_of_content()


sys.exit(0)



# write_single_domain_to_log









