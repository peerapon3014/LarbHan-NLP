# -*- coding: utf-8 -*-
import numpy as np
#Creating an index for each word in our vocab.
index_dict = {} #Dictionary to store index for each word
i = 0

#Create a count dictionary
def count_dict(sentences):
    word_count = {}
    for word in word_set:
        word_count[word] = 0
        for sent in sentences:
            if word in sent:
                word_count[word] += 1
    return word_count
 
#Term Frequency
def termfreq(document, word):
    N = len(document)
    occurance = len([token for token in document if token == word])
    return occurance/N


#Inverse Document Frequency
 
def inverse_doc_freq(word):
    try:
        word_occurance = word_count[word] + 1
    except:
        word_occurance = 1
    return np.log(total_documents/word_occurance)


def tf_idf(sentence):
    tf_idf_vec = np.zeros((len(word_set),))
    for word in sentence:
        tf = termfreq(sentence,word)
        idf = inverse_doc_freq(word)
         
        value = tf*idf
        tf_idf_vec[index_dict[word]] = value
    return tf_idf_vec


import json
from pythainlp.tokenize import word_tokenize


# dataset = load_dataset("pythainlp/han-instruct-dataset-v1.0")
# qa_data = load_dataset("pythainlp/han-instruct-dataset-v1.0")

qa_data = {}

with open("dict_chitchat_th.json") as json_file:
  qa_data = json.load(json_file)
  qa_data["ใครสร้างคุณ"] = 'พีรพล โพธิ์คำ 643021113-0' 
  

#print five questions
# list(qa_data.keys())[0:5]

with open('clean_larp_han.csv', encoding="utf8") as f:
  lines1 = f.readlines()

  data1 = lines1[1:]

qa_dict1 = {}
qa_dict1["พระเจ้าตาก"] = 'ยุทธศาสตร์ยิ่งใหญ่ความตั้งใจเด็ดเดี่ยว' 

for item in data1:
  # x = item.replace("\n","")
  x = item.split(",")
  # print(x)

  words = word_tokenize(x[1].replace("?"," ").strip(), engine="newmm")
  seg_w =" ".join(words)
  qa_dict1[seg_w] = x[2].strip()





#เขียนโค้เพิ่มเพื่อ ให้คำนวน TF-IDF ของข้อมูลคำถาม ใช้โค้ดตัวอย่างจากส่วนที่ 1
questions = list(qa_data.keys()) # คำถามเก็บอยู่ตัวแปร question แล้ว
questions1 = list(qa_dict1.keys()) # คำถามเก็บอยู่ตัวแปร question แล้ว


#Preprocessing the text data
sentences = []
word_set = []
 
 
for sent in questions1:
    x = word_tokenize(sent)
    sentences.append(x)
    for word in x:
        if word not in word_set:
            word_set.append(word)
#Set of vocab
word_set = set(word_set)
#Total documents in our corpus
total_documents = len(sentences)


 
#Creating an index for each word in our vocab.
index_dict = {} #Dictionary to store index for each word
i = 0
for word in word_set:
    index_dict[word] = i
    i += 1

word_count = count_dict(sentences)

#คำนวน TF-IDF
vectors = []
for sent in sentences:
    vec = tf_idf(sent)
    vectors.append(vec)


#เช็คขนาด vector ก่อน
vectors[0].shape


import numpy as np
from numpy.linalg import norm


#Similarity = (A.B) / (||A||.||B||)
#ref https://www.geeksforgeeks.org/how-to-calculate-cosine-similarity-in-python/


def cosine_sim(A,B):
  cosine = np.dot(A,B)/(norm(A)*norm(B))
  return cosine
 


#ประโยคที่ 100
# print(sentences[100])


s100_vec = vectors[100]


# new_sen1 =tf_idf( word_tokenize("จะเบื่อฉัน", engine="newmm"))
# new_sen2 =tf_idf( word_tokenize("ยังกินเบื่อ", engine="newmm"))


#ควรจะมากๆ เพราะประโยคคล้ายๆ กัน
# print(cosine_sim(s100_vec,new_sen1))


#ควรจะน้อยกว่า
# print(cosine_sim(s100_vec,new_sen2))


def ask(q):
  try:
    q = word_tokenize(q, engine="newmm")
    t1 = tf_idf(q)
  except:
    return "ฉันบ่รู้"
   
  maxCosine = 0
  q = ""
 
  for Key in qa_dict1.keys():
    k = word_tokenize(Key, engine="newmm")
    t2 = tf_idf(k)
    c = cosine_sim(t1,t2)
 
    if c > maxCosine:
      q = Key
      maxCosine = c
   
  if maxCosine > 0.5:
    return qa_dict1[q]
  else:
    return "???"


print(ask("ขุ่ยคือใคร"))

#print five questions
# print(list(qa_data1.values())[110:115])




