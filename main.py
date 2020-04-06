import pickle
import pandas as pd
import numpy as np
import re, string
print('Please wait...')
subm = pd.read_csv('/home/nishu/nlp project/Toxic comment classifier/sample_submission.csv')
test = pd.read_csv('/home/nishu/nlp project/Toxic comment classifier/test.csv')
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
preds = np.zeros((len(test), len(label_cols)))
with open('M.pickle', 'rb') as fm:
    M = pickle.load(fm)
print('Processing.......')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()
with open('vec.pickle', 'rb') as fm:
    vec = pickle.load(fm)
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
COMMENT = 'comment_text'
print()
# test_term_doc = vec.transform(test[COMMENT])
# test_x = test_term_doc
# for i , j in enumerate(label_cols):
#     m,r = M[i]
#     preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]
# submid = pd.DataFrame({'id': subm["id"]})
# submission = pd.concat([submid, pd.DataFrame(preds, columns = label_cols)], axis=1)
# submission.to_csv('fin.csv', index=False)


print("Please choose One of the option Below : ")
print("1. press 1 to determine toxicity in a sentence .")
print("2. press 2 to determine toxicity in more than 1 sentence .")
print("3. press 3 to determine toxicity of a list of items in a excel file with label 'comment_text' .")
print("4. press 4 to exit")
choice=int(input("Enter Your choice : "))
if(choice==1):
    print()
    sen=input("Enter the sentence : ")
    print()
    sen=[sen]
    test_term_doc = vec.transform(sen)
    test_x = test_term_doc
    for i , j in enumerate(label_cols):
        m,r = M[i]
        preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]
    print('Toxic         : ',preds[0][0]*100,'%')
    print('Severe_toxic  : ',preds[0][1]*100,'%')
    print('Obscene       : ',preds[0][2]*100,'%')
    print('Threat        : ',preds[0][3]*100,'%')
    print('Insult        : ',preds[0][4]*100,'%')
    print('Identity_hate : ',preds[0][5]*100,'%')
elif(choice==2):
    print()
    lis=[]
    num=int(input("Enter number of sentences"))
    print()
    for i in range(num):
        print("Enter sentence ",i+1,":")
        sen=input()
        print()
        sen=[sen]
        test_term_doc = vec.transform(sen)
        test_x = test_term_doc
        for i , j in enumerate(label_cols):
            m,r = M[i]
            preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]
#         lis[i]=preds[0]
#         print(preds[0])
#         lis.append(preds[-1])######
#         print(lis[0])
#         print(lis[1])
        print('Toxic : ',preds[0][0]*100,'%')
        print('Severe_toxic : ',preds[0][1]*100,'%')
        print('Obscene : ',preds[0][2]*100,'%')
        print('Threat : ',preds[0][3]*100,'%')
        print('Insult : ',preds[0][4]*100,'%')
        print('Identity_hate : ',preds[0][5]*100,'%')
#     test_term_doc = vec.transform(sen)
#     test_x = test_term_doc
#     for i , j in enumerate(label_cols):
#         m,r = M[i]
#         preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]
#     for i in range(num):
#         print("Toxic Comment Classification of sentence : ")
#         print()
#         print('Toxic         : ',lis[i][0]*100,'%')
#         print('Severe_toxic  : ',lis[i][1]*100,'%')
#         print('Obscene       : ',lis[i][2]*100,'%')
#         print('Threat        : ',lis[i][3]*100,'%')
#         print('Insult        : ',lis[i][4]*100,'%')
#         print('Identity_hate : ',lis[i][5]*100,'%')
#         print()
    
    
elif(choice==3):
    inp=input("Enter input csv name : ")
    out=input("Enter output csv name : ")
    test = pd.read_csv('/home/nishu/nlp project/Toxic comment classifier/'+inp+'.csv')
    test_term_doc = vec.transform(test[COMMENT])
    test_x = test_term_doc
    for i , j in enumerate(label_cols):
        m,r = M[i]
        preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]
    submid = pd.DataFrame({'id': subm["id"]})
    submission = pd.concat([submid, pd.DataFrame(preds, columns = label_cols)], axis=1)
    submission.to_csv(out+'.csv', index=False)  