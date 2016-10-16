# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 15:19:25 2015
Homework 2: Implement perceptron training using data given in 
/u/cs448/data/pos. What is your accuracy on the test
file when training on the train file? Plot a graph of accuracy vs iteration

@author: Md Iftekhar Tanveer (itanveer@cs.rochester.edu)
"""
import numpy as np
import random 
import matplotlib.pyplot as plt

# Reads the list of tags from file
def readalltags(tagsetfile):
    with open(tagsetfile) as f:
        tags = [item.strip() for item in f]
    return tags

# Applies dynamic programming to find the best tag sequence
def viterbi(line,E,T,tags):
    wrdlist = line.split(' ')
    x = np.ones((len(tags),len(wrdlist)))*-1*np.inf
    b = np.zeros((len(tags),len(wrdlist)))
    for i,aword in enumerate(wrdlist):
        # As I didn't see any start or end tag in the tagset, I am assuming
        # all the weights for transition from the start tag to any other tag
        # is zero (which is not true in reality).
        # So for the first word, I don't consider the transition prob
        if i==0:
            for tagid,atag in enumerate(tags):
                x[tagid,i] = E.get((atag,aword.lower()),-1*np.inf)
                b[tagid,i] = -1 # Means this is the first word
            continue

        # if not the first word, consider both transition and emission prob
        for atagid,atag in enumerate(tags):
            # theoretically, the weights should be -ve inf if a specific
            # pair is not found in the corpus. However, something didn't
            # appear in the corpus doesn't mean that its probability is
            # totally zero. So, I am assigning a small value instead of
            # -ve inf.
            emmval = E.get((atag,aword.lower()),-1*1e8) #emission prob
            for atagid_prev,atag_prev in enumerate(tags):
                trval = T.get((atag_prev,atag),-1*1e8)  #transition prob
                total = x[atagid_prev,i-1]+emmval+trval 
                # Debug
#                print 'currtag',atag+'('+str(atagid)+')','prevtag',atag_prev+\
#                '('+str(atagid_prev)+')','i',str(i),'word',aword,\
#                'emm',emmval,'trans',trval,'tot',total                
                if total>x[atagid,i]:
                    x[atagid,i] = total  # Take the maximum logprob
                    b[atagid,i] = atagid_prev # keep a backward pointer
    idx = np.argmax(x[:,-1])
    annot=[]
    # Trace back the sequence using the back pointer
    for idx_ in xrange(np.size(b,axis=1),0,-1):
        annot.append(tags[int(idx)])
        idx = b[idx,idx_-1]
    annot.reverse()
    return wrdlist,annot

# Calculate the accuracy of viterbi over a given test file
def calcaccuracy(file,E,T,tags):
    with open(file) as f:
        totalWords=0.
        countCorrect=0.
        for aline in f:
            data = [item.strip() for index, item in \
            enumerate(aline.strip().split(' ')) if not index==0]
            testline = ' '.join(data[0::2])
            annotGT = data[1::2]
            wrdlst,annt=viterbi(testline,E,T,tags)
            countCorrect=countCorrect+sum([a1==a2 for a1,a2 in zip(annotGT,annt)])
            totalWords=totalWords+len(annotGT)
    return float(countCorrect)/totalWords,countCorrect,totalWords

# Learn weights of POS tagger using perceptron
def perceptron(aline,E,T,tags,countCorrect,totalWords):
    # reading words and tags from training file
    data = [item.strip() for index, item in \
    enumerate(aline.strip().split(' ')) if not index==0]
    words = data[0::2]
    annotGT = data[1::2]
    # Applying viterbi decoding
    testline = ' '.join(words)
    wrdlst,annt=viterbi(testline,E,T,tags)
    # Using perceptron to modify weights
    if not annotGT==annt:
        for i,(tag_pred,tag_GT) in enumerate(zip(annt,annotGT)):
            # Modify the emission weights:
            #Add true value
            if not (tag_GT,words[i].lower()) in E:
                E[tag_GT,words[i].lower()] = 1
            else:
                E[tag_GT,words[i].lower()] += 1
            # Subtract predicted value
            if not (tag_pred,words[i].lower()) in E:
                E[tag_pred,words[i].lower()] = -1
            else:
                E[tag_pred,words[i].lower()] -= 1                    
            # Modify the transition weights if it is not the first tag
            if i>0:
                # add true value
                if not (annotGT[i-1],tag_GT) in T:
                    T[annotGT[i-1],tag_GT] = 1
                else:
                    T[annotGT[i-1],tag_GT] += 1
                # subtract predicted value
                if not (annt[i-1],tag_pred) in T:
                    T[(annt[i-1],tag_pred)] = -1
                else:
                    T[(annt[i-1],tag_pred)] -= 1
    # Counting total words, correct words and accuracy
    countCorrect=countCorrect+sum([a1==a2 \
    for a1,a2 in zip(annotGT,annt)])
    totalWords=totalWords+len(annotGT)  
    acc = float(countCorrect)/totalWords
    # print accuracy
    #print 'acc=',acc,'correct=',countCorrect,\
    #'total=',totalWords
    return acc,countCorrect,totalWords,E,T

def saveweights(filename,E,T):
    with open(filename,'w') as f:
        for item in E.items():
            print>>f,'E_'+item[0][0]+'_'+item[0][1]+' '+str(item[1])
        for item in T.items():
            print>>f,'T_'+item[0][0]+'_'+item[0][1]+' '+str(item[1])

# Main method
def main():
    # Assuming 5 pass over the data
    iterations = 5
    # Initialize the emission and transition weights as blank hash maps    
    E={}
    T={}
    tags = readalltags('./alltags') # read all the tags
    accperit =[]    # list to record accuracy in each iteration
    accperit_dev = [] # list of accuracy for dev set
    plt.figure(1)
    # The perceptron will run for a constant number of iterations
    for it_ in xrange(iterations):
        totalWords=0.
        countCorrect=0.
        
        # Read training files and shuffle
        f=open('./train')
        alldata = f.readlines();
        random.shuffle(alldata)
        
        # run perceptron
        print 'training perceptron. Please wait ...'
        count=0.        
        # Read each line and run perceptron command
        for aline in alldata:
            count+=1
            acc,countCorrect,totalWords,E,T = perceptron(aline,E,T,tags,\
                countCorrect,totalWords)
            if((round(count/len(alldata)*100.0)%1)==0):
                print int(count/len(alldata)*100.0),'%'

        # Calculate accuracy on dev set
        devacc = calcaccuracy('./dev',E,T,tags)[0]
        accperit.append(acc)
        accperit_dev.append(devacc)
        # Print status
        print 'Iteration:',it_,'acc_train:',acc,'acc_dev:',devacc
        
        # plot accuracy vs iteration    
        plt.plot(np.array(accperit)*100,'r-')
        plt.hold(True)
        plt.plot(np.array(accperit_dev)*100,'b-')
        plt.hold(False)
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy, %')
        plt.legend(['acc_train','acc_dev'])
        plt.draw()
        plt.pause(1)
        plt.savefig('output_plot.png',dpi=300)
        
    # Save the weights and display results on test data
    saveweights('weights',E,T)    
    testacc,correctword,totalword = calcaccuracy('./test',E,T,tags)
    print 'Result on test data:'
    print 'Total Words =',totalword
    print 'Correctly tagged =',correctword
    print 'accuracy=',testacc*100,'%'
        
    


    
            
if __name__=='__main__':
    main()
