import sys
import pandas as  pd 

def usage():
    return "Usage: python normalize_dataset.py dataset.csv\nWe assume label column name being \"sentiment\" "

#Verifying number of arguments
if(len(sys.argv)!=2):
    sys.exit(usage())

#Normalizing x between -1 and 1, given the maximum and the minimum value of our sentiment labels
def normalize(x,xmin,xmax):
    moy = float(xmin+xmax)/2
    r        = float(xmax - xmin) / 2
    normalized = (x - moy) / r
    return normalized

#Loading csv data - Add the following parameter for "stanford-sentiment-treebank.train.csv" ==> encoding="ISO-8859-1"
dataset = pd.read_csv(sys.argv[1],sep =";")

"""
print("Before normalization: ")
print(dataset)
"""

#Getting maximum/min value on the sentiment column
minval = dataset.sentiment.min()
maxval = dataset.sentiment.max()

#Normalizing the dataset between -1 and 1
dataset.sentiment = normalize(dataset.sentiment,minval, maxval);

"""
print("\nAfter normalization:")
print(dataset)
"""

newfile  = "normalized_"+sys.argv[1]
dataset.to_csv(newfile, sep=';', encoding='utf-8')

print("Saved as: "+newfile)
