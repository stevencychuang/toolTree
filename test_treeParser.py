# -*- coding: utf-8 -*-
'''
Created on 2018年3月28日

@author: STEVEN.CY.CHUANG
'''
import sys
import unittest
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
sys.path.append('../')
from util.treeParser import ClfParser

class Test_ClfParser(unittest.TestCase):
    dfTree = pd.read_pickle("../file/dfTree")
    
    def test_checkLTE(self):
        bound = [np.nan, np.nan]
        bound = ClfParser.deterBound(bound, 3.0, lte=True)
        self.assertEqual(bound, [np.nan, 3.0])
        
        bound = [np.nan, 3.0]
        bound = ClfParser.deterBound(bound, 2.0, lte=True)
        self.assertEqual(bound, [np.nan, 2.0])
        
        bound = [3.0, np.nan]
        try:
            ClfParser.deterBound(bound, 4.0, lte=True)
        except ValueError as msg:
            self.assertEqual(str(msg), "value > max")
            
            
        bound = [np.nan, np.nan]
        bound = ClfParser.deterBound(bound, 3.0, lte=False)
        self.assertEqual(bound, [3.0, np.nan])
        
        bound = [1.2, 3.0]
        bound = ClfParser.deterBound(bound, 2.0, lte=False)
        self.assertEqual(bound, [2.0, 3.0])
        
        bound = [2.0, 3.0]
        try:
            ClfParser.deterBound(bound, 1, lte=False)
        except ValueError as msg:
            self.assertEqual(str(msg), "value < min")
            
        bound = [5.0, 3.0]
        try:
            ClfParser.deterBound(bound, 6, lte=False)
        except ValueError as msg:
            self.assertEqual(str(msg), "min > max")
        
    def test_recurRulePath(self):
        # Case 1, depth = 2 and 5 features
        dfTree = self.dfTree
        clf = DecisionTreeClassifier(max_depth=2, random_state=0)
        nameFtrs = ['PINNUM', 'POSX', 'POSY', 'SIZEX', 'SIZEY']
        clf.fit(dfTree[nameFtrs], dfTree['CODE'])
        clfParser = ClfParser(clf, nameFtrs)
        
        ans = { 2: {'POSX': [np.nan, 179.39999389648438], 'SIZEX': [np.nan, 1.4005000591278076]},
                3: {'POSX': [179.39999389648438, np.nan], 'SIZEX': [np.nan, 1.4005000591278076]},
                5: {'POSY': [np.nan, 32.650001525878906], 'SIZEX': [1.4005000591278076, np.nan]},
                6: {'POSY': [32.650001525878906, np.nan], 'SIZEX': [1.4005000591278076, np.nan]}}
        
        idNode = 0
        factors = clfParser.recurRulePath(idNode, {}, {})
        self.assertEqual(factors, ans)
        
        # Case 2, depth = 3 and 5 features
        dfTree = self.dfTree
        clf = DecisionTreeClassifier(max_depth=3, random_state=0)
        nameFtrs = ['PINNUM', 'POSX', 'POSY', 'SIZEX', 'SIZEY']
        clf.fit(dfTree[nameFtrs], dfTree['CODE'])
        clfParser = ClfParser(clf, nameFtrs)
        
        ans = { 3: {'POSX': [np.nan, 179.39999389648438], 'SIZEX': [np.nan, 0.19349999725818634]},
                4: {'POSX': [np.nan, 179.39999389648438], 'SIZEX': [0.19349999725818634, 1.4005000591278076]},
                6: {'POSX': [179.39999389648438, np.nan], 'POSY': [np.nan, 112.14999389648438], 'SIZEX': [np.nan, 1.4005000591278076]},
                7: {'POSX': [179.39999389648438, np.nan], 'POSY': [112.14999389648438, np.nan], 'SIZEX': [np.nan, 1.4005000591278076]},
                10: {'POSY': [np.nan, 23.75], 'SIZEX': [1.4005000591278076, np.nan]},
                11: {'POSY': [23.75, 32.650001525878906], 'SIZEX': [1.4005000591278076, np.nan]},
                13: {'POSY': [32.650001525878906, 38.150001525878906], 'SIZEX': [1.4005000591278076, np.nan]},
                14: {'POSY': [38.150001525878906, np.nan], 'SIZEX': [1.4005000591278076, np.nan]}}
        
        idNode = 0
        factors = clfParser.recurRulePath(idNode, {}, {})
        self.assertEqual(factors, ans)
        
        # Case 3, depth = 3 and 1 feature
        clf = DecisionTreeClassifier(max_depth=3, random_state=0)
        nameFtrs = ['SIZEX']
        clf.fit(dfTree[nameFtrs], dfTree['CODE'])
        clfParser = ClfParser(clf, nameFtrs)
        
        ans = { 2: {'SIZEX': [np.nan, 0.19349999725818634]},
                4: {'SIZEX': [0.19349999725818634, 0.76050001382827759]},
                5: {'SIZEX': [0.76050001382827759, 1.4005000591278076]},
                8: {'SIZEX': [1.4005000591278076, 1.4014999866485596]},
                9: {'SIZEX': [1.4014999866485596, 1.5759999752044678]},
                11: {'SIZEX': [1.5759999752044678, 2.5510001182556152]},
                12: {'SIZEX': [2.5510001182556152, np.nan]}}
        idNode = 0
        factors = clfParser.recurRulePath(idNode, {}, {})
        self.assertEqual(factors, ans)
    
    def test_getLeaf(self):
        # Case 1, depth = 2 and 5 features
        dfTree = self.dfTree
        clf = DecisionTreeClassifier(max_depth=2, random_state=0)
        nameFtrs = ['PINNUM', 'POSX', 'POSY', 'SIZEX', 'SIZEY']
        clf.fit(dfTree[nameFtrs], dfTree['CODE'])
        clfParser = ClfParser(clf, nameFtrs)
        leaf = clfParser.getLeaf()
        print(leaf)
        
    def test_bound2str(self):
        factor = ClfParser.bound2str({"POSX": [179.39999389648438, np.nan], "POSY": [112.14999389648438, np.nan], "SIZEX": [np.nan, 1.4005000591278076]})     
        self.assertEqual(factor, {"POSX": "> 179.39999389648438", "POSY": "> 112.14999389648438", "SIZEX": "<= 1.4005000591278076"})
        
        factor = ClfParser.bound2str({'POSY': [23.75, 32.650001525878906], 'SIZEX': [1.4005000591278076, np.nan]})     
        self.assertEqual(factor, {"POSY": "> 23.75, <= 32.650001525878906", "SIZEX": "> 1.4005000591278076"})

        
if __name__ == '__main__':
#     Test_ClfParser().test_bound2str()
    Test_ClfParser().test_getLeaf()
#     Test_ClfParser().test_recurRulePath()
#     unittest.main()
