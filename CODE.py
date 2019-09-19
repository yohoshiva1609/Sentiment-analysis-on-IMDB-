#Text Classification using logistic classifier
#import the required packages 
import re 
import nltk
from nltk import corpus
import pickle
from sklearn.datasets import load_files
from nltk.corpus import stopwords

#loading data set
data_text=load_files("G:/GIT/git-02/txt_sentoken/") #check for txt_sentoken file go inside the file and copy the file patha and past here
x,y=data_text.data,data_text.target

#pickling x and y for easy processing 
with open("x.pickle",'wb') as f:
    pickle.dump(x,f)
with open("y.pickle",'wb') as f:
    pickle.dump(y,f)

#unpickling x and y
with open("x.pickle",'rb') as f:
    x=pickle.load(f)
with open("y.pickle",'rb') as f:
    y=pickle.load(f)

corpus=[]
#pre-processing text
for i in range(0,len(x)):
    #removing non charactera[1,,.:;`!....ect]
    data=re.sub(r'\W',' ',str(x[i]))
    #convert lower into upper case
    data=data.lower()
    #removing single characters in between the text 
    data=re.sub(r'\s+[a-z]\s+',' ',data)
    #removing single characters at starting sentence
    data=re.sub(r'^[a-z]\s+',' ',data)
    #removing extra spaces
    data=re.sub(r'\s+',' ',data)
    corpus.append(data)
    
    
#building TFIDF model
from sklearn.feature_extraction.text import TfidfVectorizer
vector=TfidfVectorizer(max_features=2000,min_df=3,max_df=0.6,stop_words=stopwords.words('english'))    
x=vector.fit_transform(corpus).toarray()

#pickling the TFIDF model 
with open("TFIDF.pickle",'wb') as f:
    pickle.dump(vector,f)

#unpickling the TFIDF model 
with open("TFIDF.pickle",'rb') as f:
    ve=pickle.load(f)

    
#spliting data into train and test sets 
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3)

#building logistic classifier 
from sklearn.linear_model import LogisticRegression
reg=LogisticRegression()

#pickling the reg object
with open("reg.pickle",'wb')as f:
    pickle.dump(reg,f)


#unpickling the reg 
with open("reg.pickle",'rb') as f:
    re=pickle.load(f)


#fitting model to train set
reg.fit(xtrain,ytrain)
#predicting the x-test set
y_p=reg.predict(xtest)

#predicting single reviwe 

#loding the sample review 
sample=['I love Ironman'] #put you are text here 

sample=ve.transform(sample).toarray()

print(reg.predict(sample))





