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
import time
import datetime

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
        data_dic = {}
        for item in data_col.keys():
            data_dic[item] = data_comb[:,data_col[item]]
            if data_dic[item].shape[1]==1:
                data_dic[item]=data_dic[item].flatten()
        return data_dic

    def clusters(self, l, K):
    #Taken from http//stackoverflow.com/questions/18353280/iterator-over-all-partitions-into-k-groups
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
    #Taken from http//stackoverflow.com/questions/18353280/iterator-over-all-partitions-into-k-groups
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
    setting_sim_temp={}
    #setting_sim_temp['nmkt'] = [25,50,100]
    setting_sim_temp['nmkt'] = [100]               
    #setting_sim_temp['nprod'] = [3]
    setting_sim_temp['nprod'] = [3]
    #setting_sim_temp['Dpara'] = [np.array([-4.,10.,2.,2.])] #price, const, char1, char2,... #nchar
    #setting_sim_temp['Dpara'] = [np.array([-2.,  2.,    2.,2.])] #price, const, char1, char2,... #nchar
    setting_sim_temp['Dpara'] = [np.array([-2.,  2.,    5.,5.])] #price, const, char1, char2,... #nchar
    #setting_sim_temp['Spara'] = [np.array([10.,2.,2.,     2.,2.,2.])] #nchar+nchar_cost-1. -1 for constant
    setting_sim_temp['Spara'] = [np.array([7.,   0.,0.,     .3,.3,.3])] #nchar+nchar_cost-1. -1 for constant
    #setting_sim_temp['x_dist'] = ['Exponential']
    setting_sim_temp['x_dist'] = ['Normal']
    setting_sim_temp['x_cost_dist'] = ['Normal']
    
    #setting_sim_temp['var_xi'] = [.3]
    setting_sim_temp['var_xi'] = [1.]
    #setting_sim_temp['var_lambda'] = [1.]       
    setting_sim_temp['var_lambda'] = [1.]       
    #setting_sim_temp['var_x'] = [1.]
    setting_sim_temp['var_x'] = [1.]
    #setting_sim_temp['var_x_cost'] = [1.]
    setting_sim_temp['var_x_cost'] = [1.]
    #setting_sim_temp['cov_x'] = [.5]
    setting_sim_temp['cov_x'] = [.0]
    setting_sim_temp['cov_x_bwprod'] = [.5]
    #setting_sim_temp['cov_x_cost'] = [.5]
    setting_sim_temp['cov_x_cost'] = [.0]
    setting_sim_temp['mean_x'] = [0.]
    setting_sim_temp['mean_x_cost'] = [0.]       
    setting_sim_temp['flag_char_dyn'] = [1]
    setting_sim_temp['flag_char_cost_dyn'] = [1]
    settings_sim = model_selection.ParameterGrid(setting_sim_temp)
    n_settings_sim = len(list(settings_sim))

               
    #setting for estimation    
    setting_est = {}
    setting_est['ivtype'] = 1
    setting_est['weighting']='invA'
    setting_est['Display'] = False   
    #setting for cv, gmm
    setting_cv = {'n_splits':2}


    #Repeat
    rep = 10
    cv_choice_p_all = []
    gmm_choice_p_all = []

    start=time.time()    
    for setting_i in range(n_settings_sim):
        setting_sim = list(settings_sim)[setting_i]
        setting_est['init_guess'] = np.append(setting_sim['Dpara'],setting_sim['Spara']  )
        setting_est['para_true'] = np.append(setting_sim['Dpara'],setting_sim['Spara']  )
        #Models
        models = []
        prods = np.arange(setting_sim['nprod'])
        for ngroup in range(1,setting_sim['nprod']+1):            
            for g in neclusters(prods, ngroup):
                col = np.zeros(setting_sim['nprod'])
                for i in range(ngroup):
                    col[g[i]] = i
                models.append({'col_group':col})
        N_models = len(list(models))   

        cv_choice_p = np.zeros([N_models, N_models])
        gmm_choice_p = np.zeros([N_models, N_models])
        for model_i in range(N_models):
            
            ###For test###
            model_i = 1
            
            true_model = list(models)[model_i]
            
            setting_sim['col_group'] = true_model['col_group'] #True model
            cv_score = np.zeros([N_models, rep])
            gmm_score = np.zeros([N_models, rep])
            cv_choice = np.zeros(rep)
            gmm_choice = np.zeros(rep)
            for r in range(rep):
                start_rep = time.time()
                mso_s = ModelSelectionOligopoly_Simulate(setting_sim=setting_sim)    
                mso_s.SimulateData()
                #sys.exit()
                Data_All,data_col = mso_s.MergeData(mso_s.Data_simulated)
                #print('share',mso_s.Data_simulated['share'])
                setting_est['data_col'] = data_col
                setting_est['data_col_test'] = data_col

                print('++++++++++++++CV++++++++++++++')
                hypara_cv = {'groups':mso_s.Data_simulated['mktid']}
                ms_cv = ModelSelection(EstClass=estimation.EstimateBresnahan, Data_All=Data_All, models=models, setting=setting_est, cvtype='GroupKFold',cvsetting=setting_cv,cvhypara=hypara_cv)
                ms_cv.fit()
                cv_score[:,r] = ms_cv.score_models
                        
                print('++++++++++++++GMM++++++++++++++')
        
                ms_gmm = ModelSelection(EstClass=estimation.EstimateBresnahan, Data_All=Data_All, models=models, setting=setting_est, cvtype='InSample',cvsetting=setting_cv,cvhypara=hypara_cv)
                ms_gmm.fit()
                gmm_score[:,r] = ms_gmm.score_models
                         
                cv_choice[r] = np.where(np.min( cv_score[:,r] )==cv_score[:,r])[0][0]
                gmm_choice[r] = np.where(np.min( gmm_score[:,r] )==gmm_score[:,r])[0][0]
                
                end_rep = time.time()
                time_rep = end_rep-start_rep
                print('###########rep %i, time for one rep:%f######################'%(r, time_rep))
                
            sys.exit()

            
            cv_cp = np.bincount(cv_choice.astype(int))
            while len(cv_cp)<N_models:
                cv_cp = np.append(cv_cp, 0.0)
            gmm_cp = np.bincount(gmm_choice.astype(int))
            while len(gmm_cp)<N_models:
                gmm_cp = np.append(gmm_cp, 0.0)
            cv_choice_p[model_i,:] = cv_cp/rep
            gmm_choice_p[model_i,:] = gmm_cp/rep
            
        cv_choice_p_all.append(cv_choice_p)
        gmm_choice_p_all.append(gmm_choice_p)
                   
        
    end = time.time()
    time_calc = end-start
    print('Total Time:'+str(time_calc))
    DateCalc=datetime.date.today().strftime('%b-%d-%Y')
    np.save('cv_choice_p_all_'+DateCalc+'.npy',cv_choice_p_all)
    np.save('gmm_choice_p_all_'+DateCalc+'.npy',gmm_choice_p_all)
            
    
