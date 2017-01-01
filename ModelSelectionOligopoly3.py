# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 17:29:44 2016

@author: Hajime
"""
import warnings
import matplotlib.pyplot as plt
import numpy as np
import collections, numpy
import copy
import os
import csv
import sys
import scipy.optimize

import math
import itertools

from sklearn import model_selection
from sklearn import preprocessing


os.chdir(os.path.dirname( os.path.abspath( __file__ ) ))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from ModelSelection import ModelSelection

import EstimateOligopoly3 as estimation
import SimulateOligopoly3 as simulation

class ModelSelectionOligopoly_Simulate:
    def __init__(self, setting_sim\
    #savedata=0,savegraph=0,showfit=0\
    ):        
        #setting for simulation
        self.setting_sim = setting_sim
        
    def SimulateData(self):       
        flag_OptFailed = 1
        flag_NegativeMC = 1
        flag_DeltaSingular = 1        
        while flag_NegativeMC==1 or flag_DeltaSingular==1 or flag_OptFailed==1:
            sim = simulation.SimulateBresnahan(**self.setting_sim)
            print('---Starting Simulation---')
            sim.Simulate()
            flag_NegativeMC = sim.flag_NegativeMC
            flag_DeltaSingular = sim.flag_DeltaSingular
            flag_OptFailed = sim.flag_OptFailed
            if flag_NegativeMC==1 or flag_DeltaSingular==1 or flag_OptFailed==1:
                print('---Simulation Failed NegativeMC=%i, DeltaSingular=%i, OptFailed=%i---' %(flag_NegativeMC, flag_DeltaSingular, flag_OptFailed))
        print('*****Data Simulated*****')
        self.Data_simulated = sim.Data
        
    def MergeData(self,data_dic):
        n = list(data_dic.values())[0].shape[0]
        data_col={}
        i=0
        data_comb = np.ones(n)
        for item in data_dic.keys():
            data_comb = np.c_[data_comb, data_dic[item]]
            if data_dic[item].ndim==1:
                data_col[item] = np.arange(i, i+1)
                i=i+1
            if data_dic[item].ndim==2:
                data_col[item] = np.arange(i, i+data_dic[item].shape[1])
                i=i+data_dic[item].shape[1]
        data_comb=np.delete(data_comb,0,axis=1)
        return data_comb, data_col
        
    def RecoverData(self,data_comb, data_col):
        data_recovered = {}
        for item in data_col.keys():
            data_recovered[item] = data_comb[:,data_col[item]]
            if data_recovered[item].shape[1]==1:
                data_recovered[item]=data_recovered[item].flatten()
        return data_recovered

    def clusters(self, l, K):
    #Taken from http://stackoverflow.com/questions/18353280/iterator-over-all-partitions-into-k-groups
        if l.size>0:
            prev = None
            for t in self.clusters(l[1:], K):
                tup = sorted(t)
                if tup != prev:
                    prev = tup
                    for i in range(K):
                        yield tup[:i] + [[l[0]] + tup[i],] + tup[i+1:]
        else:
            yield [[] for _ in range(K)]           
    def neclusters(self, l, K):
        for c in self.clusters(l, K):
            if all(x for x in c): yield c   


if __name__=='__main__':
    def clusters(l, K):
    #Taken from http://stackoverflow.com/questions/18353280/iterator-over-all-partitions-into-k-groups
        if l.size>0:
            prev = None
            for t in clusters(l[1:], K):
                tup = sorted(t)
                if tup != prev:
                    prev = tup
                    for i in range(K):
                        yield tup[:i] + [[l[0]] + tup[i],] + tup[i+1:]
        else:
            yield [[] for _ in range(K)]           
    def neclusters(l, K):
        for c in clusters(l, K):
            if all(x for x in c): yield c   
    
    
    #setting for simulation
    setting_sim={}
    setting_sim['nmkt'] = 50               
    setting_sim['nprod'] = 3
    setting_sim['col_group'] = np.array([0,1,2]) #True model
    setting_sim['Dpara'] = np.array([-4.,10.,2.,2.]) #price, const, char1, char2,... #nchar
    setting_sim['Spara'] = np.array([15.,2.,2.,     2.,2.,2.]) #nchar+nchar_cost-1. -1 for constant
    setting_sim['var_xi'] = .3
    setting_sim['var_lambda'] = 1.       
    setting_sim['var_x'] = 1.
    setting_sim['var_x_cost'] = 1.
    setting_sim['cov_x'] = .5
    setting_sim['cov_x_cost'] = .5
    setting_sim['mean_x'] = 0.
    setting_sim['mean_x_cost'] = 0.       
    setting_sim['flag_char_dyn'] = 0
    setting_sim['flag_char_cost_dyn'] = 0
       
    mso_s = ModelSelectionOligopoly_Simulate(setting_sim=setting_sim)    
    mso_s.SimulateData()
    Data_All,data_col = mso_s.MergeData(mso_s.Data_simulated)

    #setting for estimation    
    setting_est = {}
    setting_est['ivtype'] = 0
    setting_est['weighting']='invA'
    setting_est['Display'] = False   
    setting_est['data_col'] = data_col
    setting_est['data_col_test'] = data_col
    setting_est['init_guess'] = np.append(setting_sim['Dpara'],setting_sim['Spara']  )
    setting_est['para_true'] = np.append(setting_sim['Dpara'],setting_sim['Spara']  )
    #setting for cv, gmm
    setting_cv = {}
    hypara_cv = {'groups':mso_s.Data_simulated['mktid']}
    #Models
    models = []
    prods = np.arange(setting_sim['nprod'])
    for ngroup in range(1,setting_sim['nprod']+1):            
        for g in neclusters(prods, ngroup):
            col = np.zeros(setting_sim['nprod'])
            for i in range(ngroup):
                col[g[i]] = i
            models.append({'col_group':col})

    
    ms = ModelSelection(EstClass=estimation.EstimateBresnahan, Data_All=Data_All, models=models, setting=setting_est, cvtype='GroupKFold',cvsetting=setting_cv,cvhypara=hypara_cv)
    ms.fit()
    #def __init__(self, EstClass, Data_All, models, setting, cvtype='KFold',cvsetting=None)
    #def __init__(self, data_col=None, data_col_test=None,\
    #             ivtype=0, weighting='invA', col_group=None, colmat=None,Display=False)