# -*- coding: utf-8 -*-
import numpy as np
import csv
#Creating an index for each word in our vocab.
index_dict = {} #Dictionary to store index for each word
i = 0

#Create a count dictionary
#นับคำที่อยู่ใน doc
def count_dict(sentences):
    word_count = {}
    for word in word_set:
        word_count[word] = 0
        for sent in sentences:
            if word in sent:
                word_count[word] += 1
    return word_count
 
#Term Frequency #TF
#ความถี่ในแต่ละคำใน doc
def termfreq(document, word):
    N = len(document)
    occurance = len([token for token in document if token == word])
    return occurance/N


#Inverse Document Frequency
#คำนั้นปรากฏในเอกสารทั้งหมดมากแค่ไหน
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


from pythainlp.tokenize import word_tokenize

# dataset = load_dataset("pythainlp/han-instruct-dataset-v1.0")
qa_dict1 = {}
qa_dict1["พระเจ้าตาก"] = 'ยุทธศาสตร์ยิ่งใหญ่ความตั้งใจเด็ดเดี่ยว' 
qa_dict1["ci/cd"] = 'CI/CD เป็นวิธีการที่ช่วยให้เราสามารถสร้าง Application ซึ่งเป็นแนวคิดที่ช่วยลดปัญหาปัญหาระหว่างทีม Development และทีม Operation ก่อนที่ Deploy ไปยัง Production' 


with open('clean_larp_han.csv', encoding="utf8") as f:
  lines1 = f.readlines()
  data1 = lines1[1:]
  for item in data1:
    # x = item.replace("\n","")
    x = item.split(",")
    # print(x)
    words = word_tokenize(x[1].replace("?"," ").strip(), engine="newmm")
    words1 = word_tokenize(x[2].replace("?"," ").strip(), engine="newmm")
    seg_w =" ".join(words)
    seg_w1 =" ".join(words1)
    qa_dict1[seg_w] = x[2].strip()
    qa_dict1[seg_w1] = x[2].strip()


with open('clean_food.csv', newline='', encoding='utf-8') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)  # skip header
    for row in csvreader:
        name = row[1]  #ques
        text = row[2]   #ans
        text = text.replace('\n', ' ')  # เชื่อมข้อความในบรรทัดเดียวกันด้วยช่องว่าง
        # ใช้ word_tokenize ตัดคำ
        words = word_tokenize(x[1].replace("?","").strip(), engine="newmm")
        words1 = word_tokenize(x[2].replace("-","").replace("#","").strip(), engine="newmm")
        seg_w =" ".join(words)
        seg_w1 =" ".join(words1)
        qa_dict1[seg_w] = x[2].strip()
        qa_dict1[seg_w1] = x[2].strip()

with open("clean_wiki.csv", encoding="utf8") as f_wiki:
   wiki_reader = csv.reader(f_wiki)
   for row in wiki_reader:
      name = row[1]
      text = row[2]
      text = text.replace("\n", " ")
      words = word_tokenize(name.replace("?", " ").strip(), engine="newmm")
      seg_word = " ".join(words)
    #   qa_dict1[seg_word] = text


questions1 = list(qa_dict1.keys()) # คำถามเก็บอยู่ตัวแปร question 

#Preprocessing the text data
sentences = [] #[['ฉัน'],['ไป'],['เที่ยว'],['ที่'],['ภูเก็ต']]
word_set = []  #เก็บคำไม่ซ้ำ
 
 
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
    return "ห่านก็มั่ยรู้วว :("
   
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
    return "อิหยังน้ออ"

def search_food(q):
    try:
        query_tokens = word_tokenize(q, engine="newmm")
        query_vec = tf_idf(query_tokens)
    except:
        return "ไม่สามารถแปลคำค้นหาได้ :("
    
    results = []
    for key, value in qa_dict1.items():
        key_tokens = word_tokenize(key, engine="newmm")
        key_vec = tf_idf(key_tokens)
        similarity = cosine_sim(query_vec, key_vec)
        if similarity > 0.5:  # เลือกค่าความคล้ายที่มากกว่า 0.5
            results.append(value)
    
    if results:
        return results
    else:
        return "ไม่พบข้อมูลที่ตรงกับคำค้นหา"



print(ask("ขุ่ยคือใคร"))
# print(ask("ข้าวเม่าทอด"))
# print(ask("ข้าวกะเพรา"))

#print five questions
# print(list(qa_data1.values())[110:115])




