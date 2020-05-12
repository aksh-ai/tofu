
def entropy(c,n):
    return -(c/n)*math.log(c/n,2)
    #Mathematical formula to calculate the entropy value upon each split

def cal_entropy(c1,c2):
    if c1==0 or c2==0: #Returns 0 if one of the groups have items of the same class
        return 0
    return entropy(c1,c1+c2)+entropy(c2,c1+c2) #else call the entropy function to calculate it until c1 or c2 reaches 0

#Considering a single cell...
    
def single_entropy(division):
    s=0
    n=len(division) #Total no. of elements in in the selected cell 
    classes=set(division) 

    for c in classes: #for each class of element get the entropy
        n_c=sum(division==c)
        e=n_c/n*entropy(sum(division==c),sum(division!=c)) #Weighted Average
        s=s+e
        
    return s,n
#Considering both the cells..
def get_entropy(y_predict,y_real):
    #y_predict is the split decision
    if len(y_predict) != len(y_real):
        return None
    n=len(y_predict)
    s1,n1=single_entropy(y_predict[y_real]) #Calculating the entropy for the left side
    s2,n2=single_entropy(y_predict[~y_real]) #Calculating the entropy for the right side
    s=n1/n*s1+n2/n*s2 #Overall entropy condidering the total elements in the tree
    return s

class DecisionTree(object):
    def __init__(self,maxdepth):
        self.depth=0  #Assignment of initial and maxdepth
        self.maxdepth=maxdepth

    def find_best_split(self,col,y):
        min_entropy=10
        n=len(y)

        for value in set(col):
            y_predict=col<value
            my_entropy=get_entropy(y_predict,y)

            if(my_entropy<min_entropy): #if the entropy of the selected split is lower than 10 , then min_entropy=my_entropy
                min_entropy=my_entropy
                cutoff=value
#Result of this will provide the best splits for a selected feature
        return min_entropy,cutoff
    def find_best_split_all(self,x,y):
        col=None
        min_entropy=1
        cutoff=None
        for i,c in enumerate(x.T):# Iterate each feature 
            entropy,curr_cutoff=self.find_best_split(c,y) #Find the best split of the selected feature
            if entropy==0:
                #The perfect cutoff , which means we have found a perfect cell with a singular class
                return i,curr_cutoff,entropy
            elif entropy<=min_entropy:
                entropy=min_entropy
                col=i
                cutoff=curr_cutoff #further searching feature by feature , perfect cutoff is found out
            return col,cutoff,min_entropy

    def fit(self,x,y,par_node={},depth=0):
        # Here x is the feature set and y is the target
        if par_node is None:
            return None #If no further divisions exist , retain the previously stopped values of the trees
        elif len(y)==0:
            return None #No target data given
        elif self.allsame(y): # All the features of the same class has been grouped together
            return {'val':y[0]}

        elif depth>=maxdepth:
            return None #Depth of the tree reaches or exceeds the maxdepth assigned
        else:
            col,cutoff,entropy=self.find_best_split_all(x,y) #Find the best split
            y_left=y[x[:,col]<cutoff] # Considering left side data
            y_right=y[x[:,col]>=cutoff] #Considering right side data 

            #Generating tree for left data
            par_node={'left'}=self.fit(x[x[:,col]<cutoff],y_left,{},depth+1)
            #Generating tree for right data
            par_node={'right'}=self.fit(x[x[:,col]>=cutoff],y_right,depth+1)
            self.depth+=1 #Increasing the depth as we move further down through the subgroups
            self.trees=par_node
            return par_node
    
    def allsame(self,items):
        return all(x==items[0] for x in items)





            

    


               




        





        

