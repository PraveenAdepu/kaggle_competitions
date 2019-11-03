# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 17:55:46 2016

@author: SriPrav
"""

# full leak can be extracted with 10 GB memory
# but you can extract a subset of leak if you have less memory
# pypy leak.py takes 30 mins

import csv
import os

data_directory = 'C:\\Users\SriPrav\Documents\R\\13Outbrain'
os.chdir(data_directory)
print os.getcwd(); # Prints the working directory


memory = 120 # stands for 10GB, write your memory here
limit = 114434838 / 10 * memory * 10

promoted_content = data_directory+'\input\promoted_content.csv'
filename = data_directory+'\input\page_views.csv'

leak = {}
for c,row in enumerate(csv.DictReader(open(promoted_content))):
    if row['document_id'] != '':
        leak[row['document_id']] = 1 
print(len(leak))
count = 0

#filename = '../input/page_views_sample.csv' # comment this out locally
for c,row in enumerate(csv.DictReader(open(filename))):
#    if count>limit:
#	    break
    if c%1000000 == 0:
        print (c,count)
    if row['document_id'] not in leak:
	    continue
    if leak[row['document_id']]==1:
	    leak[row['document_id']] = set()
    lu = len(leak[row['document_id']])
    leak[row['document_id']].add(row['uuid'])
    if lu!=len(leak[row['document_id']]):
	    count+=1
fo = open('leak.csv','w')
fo.write('document_id,uuid\n')
for i in leak:
    if leak[i]!=1:
	    tmp = list(leak[i])
	    fo.write('%s,%s\n'%(i,' '.join(tmp)))
	    del tmp
fo.close()