# -*- coding: utf-8 -*-
'''
Created on 2017年10月23日
1st edited on March 29, 2018: create a new parser for sklearn tree classifier
@author: steven.cy.chuang
'''
import math
import copy
import pandas as pd
import numpy as np
from operator import itemgetter 

class MyParser():
    """
    This Parser parses the dot_data into several collections as attributes:
        0. material - the split lines:
            ['0 [label="SIZEX <= 1.4005\\ngini = 0.0088\\nsamples = 327450\\nvalue = [138, 358, 1, 949, 2, 326002]"] ',
             '1 [label="POSX <= 179.4\\ngini = 0.0039\\nsamples = 295944\\nvalue = [138, 303, 1, 137, 2, 295363]"] ',
             ...
             '6 [label="gini = 0.0136\\nsamples = 23895\\nvalue = [0, 8, 0, 156, 0, 23731]"] ',
             '4 -> 6 ']
        1. nodes    - the non-leaf nodes, the example is as following:
            {'0': ('SIZEX', '<= 1.4005'),
             '1': ('POSX', '<= 179.4'),
             '4': ('POSY', '<= 32.65')}
        2. leaf     - the data frame for the leaf , the example is as following:
            id    gini                          count                                             factor  
            0  2  0.0031  [138, 171, 1, 137, 2, 292663]   {'POSX': '<= 179.4', 'SIZEX': '<= 1.4005', 'PO... 
            1  3  0.0889        [0, 132, 0, 0, 0, 2700]   {'POSX': '> 179.4', 'SIZEX': '> 1.4005', 'POSY... 
            2  5  0.1687       [0, 47, 0, 656, 0, 6908]   {'POSX': nan, 'SIZEX': '<= 1.4005', 'POSY': '<...
            3  6  0.0136       [0, 8, 0, 156, 0, 23731]   {'POSX': nan, 'SIZEX': '> 1.4005', 'POSY': '> ... 
        3. parents  - the dictionary of parents which presents id relationship for child : parent
            {'1': '0', '2': '1', '3': '1', '4': '0', '5': '4', '6': '4'}
    """
    
    def __init__(self, dot_data):
        """
        dot_data: str
            original dot_data to parse
        """
        self.dot_data = dot_data
        self.material = dot_data.split(';\n')[1:-1]
        idCont = self.material[0].split(' ', 1)  # id,content
        arrCont = idCont[1].split('"')[1].split("\\n")  # ['SIZEX <= 1.4005',..]
        [_key, _val] = arrCont[0].split(' ', 1)  # SIZEX, <= 1.4005
        self.nodes = {'0': (_key, _val)}
        self.leaf = pd.DataFrame(columns=['id', 'gini', 'count', 'factor'])
        self.parents = {}
    
    def getParents(self):
        # Get the dictionary of parents
        for i in range(2, len(self.material), 2):
            [id1, id2] = itemgetter(*[0, 2])(self.material[i].split(' '))    
            self.parents[id2] = id1
            
    def getNodes(self):
        # Separate leaf and non-leaf nodes by determing if the content start with gini(because no criterion)
        # 1.Get the leaf table and fill it with id, gini, amount
        # 2.Get the non-leaf nodes dictionary which contains the criteria
        for i in range(1, len(self.material), 2):
            idCont = self.material[i].split(' ', 1)
            arrCont = idCont[1].split('"')[1].split("\\n")  # ['SIZEX <= 1.4005', 'gini = 0.0136',...]    
            [_key, _val] = arrCont[0].split(' ', 1)  # SIZEX, <= 1.4005
            if _key == 'gini':
                self.leaf = self.leaf.append({'id':idCont[0],
                                    'gini':_val.replace('= ', ''),
                                    'count':arrCont[2].split('= ')[1]
                                    }, ignore_index=True)
            else:
                self.nodes[idCont[0]] = (_key, _val)
    
    def parse(self):
        """
        Parse the dot_data of tree and get the collections parents and nodes
        """
        self.getParents()
        self.getNodes()
        return(self.nodes, self.parents)
    
    def getLeaf(self):
        """
        Fill the leaf table with the criteria and get the table 
        """
        factor = pd.DataFrame()
        numLeaf = len(self.leaf.index)
        for i in range(numLeaf):
            # Get the criteria from its parent
            _id = self.leaf.get_value(index=i, col='id')
            par = self.parents.get(_id)
            
            # Put the criteria into the dict "factors" until all parent nodes has been searched
            lenBin = '{0:0'+str(numLeaf.bit_length()-1)+'b}'
            right = lenBin.format(i) # convert the index into binary like '01' means right、left bottom-up
            while (par != None):
                (_key, _val) = self.nodes.get(par)
                # 1 means right '>'
                if right[-1]=='1': 
                    _val = _val.replace('<=', '>') 
                factor.set_value(index = i, col = _key, value = _val)
                _id = par
                par = self.parents.get(_id)
                right = right[:-1]

        for i in self.leaf.index:
            self.leaf.set_value(index = i, col = 'factor', value = factor.loc[i].to_dict())
        return self.leaf
    
class ClfParser():
    """
    The parser for the decision tree of sklearn, use the properties of classifier directly.
    Attributes:
        clf(obj): the trained classifier of the decision tree of sklearn
        nameFtrs (list): the list containing the names of features
    """
    def __init__(self, clf, nameFtrs):
        self.clf = clf
        self.nameFtrs = nameFtrs
    
    @staticmethod    
    def deterBound(bound, value, lte):
        """
        Determine a new boundary of left/right child for one feature of decision tree.
        The new interval will be shrinking while min will be replaced by the value or max will be replaced by the value 
        Args:
            bound (length 2 array):　the original boundary for one feature, [min, max]
            value (float): the criterion to split child nodes
            lte (boolean): True for left child node(<=), False for right child node(>)
        Returns:
            bound (length 2 array): new boundary for the child node
        Raises:
            check for value should be between min and max, and check min < max
        """
        # For the case of "less than equal", check value < max. Then, replace max by the value
        if lte:
            if bound[1] is not np.nan and value > bound[1] :
                raise ValueError("value > max")
            bound[1] = value
            
        # For the case of "great than", check value > min. Then, replace min by the value
        else:
            if bound[0] is not np.nan and value < bound[0] :
                raise ValueError("value < min")
            bound[0] = value
            
        # Final check for max > min, otherwise throw exception
        if  bound[0] > bound[1]:
            raise ValueError("min > max")
        return bound
    
    def recurRulePath(self, idNode, bounds, factors):
        """
        Get the rule paths by recursive way(depth-first search).
        Args:
            idNode (int): the ID of the current node
            bounds (dict): the dictionary containing of the rule path for the current node. 
                Each pair means the correspond boundary for one feature, e.g., {'POSY': [23.75, 32.65], 'SIZEX': [1.4, nan]}
            factors (dict): the dictionaries of bounds for all (leaf) nodes, e.g., {3: {'POSX': [nan, 17], 'SIZEX': [nan, 0.19]},4: {'POSX': [nan, 179],...
        """
        # the inner object of the classifier, it contains the properties of the tree such as children, threshold, etc.
        clftree = self.clf.tree_
        
        # If leaf node, pass the final path directly
        if clftree.children_left[idNode] == -1 and clftree.children_right[idNode] == -1:
            factors[idNode] = bounds
        
        # If non-leaf node, parse the rule path recursively
        else:
            nameFtr = self.nameFtrs[clftree.feature[idNode]]
            threshold = clftree.threshold[idNode]
            
            # Check if the boundary of the feature exists
            if nameFtr not in bounds.keys():
                bounds[nameFtr] = [np.nan, np.nan]
            
            # Check if the left child exists    
            if clftree.children_left[idNode] != -1:
                boundsLeft = copy.deepcopy(bounds)
                boundsLeft[nameFtr] = self.deterBound(boundsLeft[nameFtr], threshold, lte = True)
                self.recurRulePath(clftree.children_left[idNode], boundsLeft, factors)
            
            # Check if the right child exists  
            if clftree.children_right[idNode] != -1:
                boundsRight = copy.deepcopy(bounds)
                boundsRight[nameFtr] = self.deterBound(boundsRight[nameFtr], threshold, lte = False)
                self.recurRulePath(clftree.children_right[idNode], boundsRight, factors)
        
        return factors
    
    def getLeaf(self):
        """
        Fill the leaf table with the criteria and get the table 
        """
        clftree = self.clf.tree_
        leaf = pd.DataFrame(columns=["id", "gini", "count", "factor"])
        factors = self.recurRulePath(0, {}, {})
        
        for idNode in factors.keys():
            leaf = leaf.append({"id" : idNode,
                         "gini" : clftree.impurity[idNode],
                         "count" : str([int(c) for c in clftree.value[idNode][0]]),
                         "factor" : self.bound2str(factors.get(idNode))
                         }, ignore_index=True)
        return leaf
    
    @staticmethod 
    def bound2str(bounds):
        factor = {}
        for ftr in bounds.keys():
            bound = bounds.get(ftr)
            boundStr = ""
            if bound[0] is not np.nan:
                boundStr += "> " + str(bound[0])
                if bound[1] is not np.nan:
                    boundStr += ", <= " + str(bound[1])
            elif bound[1] is not np.nan:
                boundStr += "<= " + str(bound[1])
            factor[ftr] = boundStr
        
        return (factor)
    
if __name__ == "__main__":
    dot_data = open("../file/tree.txt", 'r').read()
    parser = MyParser(dot_data)
    parser.parse()
    leaf = parser.getLeaf()
    print(leaf.sort_values('gini'))
    print(leaf.iloc[0]["factor"])
    print(type(leaf.iloc[0]["factor"]))