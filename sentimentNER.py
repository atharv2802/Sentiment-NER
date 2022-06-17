# link for the play text : https://www.gutenberg.org/files/1286/1286-h/1286-h.htm#chap02 
# Sentiment analysis of the play "A Midsummer Night's Dream" using VADER (Valence Aware Dictionary and sEntiment Reasoner)

import subprocess

p = subprocess.Popen('pip install -r requirements.txt', stdout=subprocess.PIPE, shell=True)
(output, err) = p.communicate()
p_status = p.wait()
print("Command output : ", output)
print("Command exit status/return code : ", p_status)
print("Installed required libraries")


import importlib.util
import sys

def check_package(name):
    if name in sys.modules:
        print(f"{name!r} already in sys.modules")
    elif (spec := importlib.util.find_spec(name)) is not None:
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        print(f"{name!r} has been imported")
    else:
        print(f"Can't find the {name!r} module")
        print(f"Please install {name!r} module using pip")
        return 'Error'
    
    
def split_data_tokenize(data):
    lines = data.split("\n")
    non_empty_lines = [line for line in lines if line.strip() != ""]
    filtered = ""
    for line in non_empty_lines:
        filtered += line + "\n"
    token_text = sent_tokenize(filtered)
    return token_text
    
def sentiment_scores(sentence):

    sentiment_obj = SentimentIntensityAnalyzer()
 
    sentiment_dict = sentiment_obj.polarity_scores(sentence)
     
    print("Overall sentiment dictionary is : ", sentiment_dict)
    print("sentence was rated as ", sentiment_dict['neg']*100, "% Negative")
    print("sentence was rated as ", sentiment_dict['neu']*100, "% Neutral")
    print("sentence was rated as ", sentiment_dict['pos']*100, "% Positive")
 
    print("Sentence Overall Rated As", end = " ")
 
    # decide sentiment as positive, negative and neutral
    if sentiment_dict['compound'] >= 0.15 :
        print("Positive\n")
        return "Positive", sentiment_dict['compound']
 
    elif sentiment_dict['compound'] <= - 0.15 :
        print("Negative\n")
        return "Negative", sentiment_dict['compound']
 
    else :
        print("Neutral\n")
        return "Neutral", sentiment_dict['compound']
        



if __name__ == "__main__" :
    
    vader = check_package('vaderSentiment')
    if vader == 'Error' :
        raise Exception ('vaderSentiment package not installed!\nPlease install "vaderSentiment" package using pip')
        
    nltk_pack = check_package('nltk')
    if nltk_pack == 'Error' :
        raise Exception ('nltk package not installed!\nPlease install "nltk" package using pip')
        
    stanza_pack = check_package('stanza')
    if nltk_pack == 'Error' :
        raise Exception ('stanza package not installed!\nPlease install "stanza" package using pip')
        
    torch_pack = check_package('torch')
    if nltk_pack == 'Error' :
        raise Exception ('torch package not installed!\nPlease install "torch" package using pip')
        
    import nltk
    nltk.download('punkt')
    from nltk.tokenize import sent_tokenize
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    
    text_file = open("play.txt", "r",encoding='utf-8')
    text_data = text_file.read()
    text_file.close()
    
    text_list = split_data_tokenize(text_data)
    
    count_pos = 0
    count_neg = 0
    count_neu = 0
    compound_score = 0
        
    with open("Result/lineAnalysisResult.txt","w+") as result_file:
        print("\n--------------------Analysis of each line---------------------\n",file=result_file)
        for i in range(len(text_list)):
            line_res,score = sentiment_scores(text_list[i])
            
            compound_score+=score
            
            if line_res == "Positive":
                count_pos+=1
            elif line_res == "Negative":
                count_neg+=1
            else:
                count_neu+=1
            
            print("Line "+ str(i+1) + " : " + line_res + "\nCompound Score : " + str(score) , file=result_file)
        print("\n**********************************************************************",file=result_file)   
        
    result_file.close()
    
    mean_compound_score = compound_score/len(text_list)
    
    with open("Result/Result.txt","w+") as result_file:
        print("\n\n--------------------Overall Sentiment statistics---------------------\n",file=result_file)
        print("\nCount of sentences in the data file : " +  str(len(text_list)), file=result_file)
        print("\nCount of positive sentences : " +  str(count_pos), file=result_file)
        print("\nCount of negative sentences : " +  str(count_neg), file=result_file)
        print("\nCount of neutral sentences : " +  str(count_neu), file=result_file)
        print("\nMean/Average compund score of all sentences : " +  str(mean_compound_score), file=result_file)
        print("\n**********************************************************************",file=result_file)
    
    result_file.close()
           
     
    #Ner
    import numpy as np
    import stanza
    stanza.download('en')
    model = stanza.Pipeline('en')
    
    def unique(l):
        x = np.array(l)
        return np.unique(x)
    
    with open("Result/NERAnalysisResult.txt","w") as result_file:
        for s in text_list:
            a = model(s)
            for ent in a.entities:
                print(f'{ent.text}\t{ent.type}',file=result_file)
    
    result_file.close()
    
    with open("Result/NERAnalysisResult.txt","r") as result_file:
        t = result_file.read()
        ner_list = t.splitlines()
        print(ner_list)
        
        person_list = []
        loc_list = []
        org_list = []
                
        for i in ner_list:
          x = i.split("\t")
          if x[1] == "PERSON":
            person_list.append(x[0])
          elif x[1] == "ORG":
            org_list.append(x[0])
          elif x[1] == "GPE" or x[1] == "LOC":
             loc_list.append(x[0])
        
        u_person_list = []
        u_loc_list = []
        u_org_list = []
        
        u_person_list = unique(person_list)
        u_loc_list = unique(loc_list)
        u_org_list = unique(org_list)
        
    result_file.close()
    
    with open("Result/Result.txt","a") as result_file:
        print("\n\n--------------------Overall NER statistics---------------------\n",file=result_file)
        print("\nTotal Persons : " +  str(len(person_list)),file=result_file)
        print("\nTotal locations : " +  str(len(loc_list)),file=result_file)
        print("\nTotal organizations : " +  str(len(org_list)),file=result_file)
        
        print("\nTotal Distinct Persons : " +  str(len(u_person_list)),file=result_file)
        print("\nTotal Distinct locations : " +  str(len(u_loc_list)),file=result_file)
        print("\nTotal Distinct organizations : " +  str(len(u_org_list)),file=result_file)
        
        print("\nDistinct Person list : " +  str(u_person_list),file=result_file)
        print("\nDistinct location list : " +  str(u_loc_list),file=result_file)
        print("\nDistinct organization list : " +  str(u_org_list),file=result_file)
        print("\n**********************************************************************",file=result_file)
    
    result_file.close()
    
    
    
    print("\n\n--------------------Overall Sentiment statistics---------------------\n")
    print("\nCount of sentences in the data file : " +  str(len(text_list)))
    print("\nCount of positive sentences : " +  str(count_pos))
    print("\nCount of negative sentences : " +  str(count_neg))
    print("\nCount of neutral sentences : " +  str(count_neu))
    print("\nMean/Average compund score of all sentences : " +  str(mean_compound_score))
    print("\n**********************************************************************")
    
    
    print("\n\n--------------------Overall NER Statistics---------------------\n")
    print("\nTotal Persons : " +  str(len(person_list)))
    print("\nTotal locations : " +  str(len(loc_list)))
    print("\nTotal organizations : " +  str(len(org_list)))
    
    print("\nTotal Distinct Persons : " +  str(len(u_person_list)))
    print("\nTotal Distinct locations : " +  str(len(u_loc_list)))
    print("\nTotal Distinct organizations : " +  str(len(u_org_list)))
    
    print("\nDistinct Person list : " +  str(u_person_list))
    print("\nDistinct location list : " +  str(u_loc_list))
    print("\nDistinct organization list : " +  str(u_org_list))
    print("\n**********************************************************************")
