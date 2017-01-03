# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 16:58:15 2016

@author: Hajime
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import collections, numpy
import pandas as pd
import copy
import os
import csv
import sys
import scipy.optimize

os.chdir(os.path.dirname( os.path.abspath( __file__ ) ))
datadir = 'data/'

class SimulateBresnahan:
    
    def __init__(self,nmkt = 2, nprod = 3, colmat=None, col_group=None,\
    Dpara = np.array([-5.,1.,-1.]),var_xi = .3, var_lambda = 1.,\
    Spara = np.array([10.,1.,2.]), var_x=1. , cov_x=.5, var_x_cost=1., cov_x_cost=.5,\
    mean_x = 0., mean_x_cost = 0.,\
    flag_char_dyn = 0, flag_char_cost_dyn=0):
        self.nmkt = nmkt                
        #self.nchar = nchar_input
        #self.nchar_cost = nchar_cost_input        
        self.nprod = nprod
        self.nobs = self.nmkt*self.nprod

        if colmat == None:
            self.colmat = self.GenColMat(col_group)
        if colmat != None:
            self.colmat = colmat #collusion matrix
            
        print('colmat:')
        print(str(self.colmat))

        self.Dpara = Dpara
        self.Spara = Spara
        
        self.nchar = self.Dpara.size-1 #include constant, not price
        self.nchar_cost = self.Spara.size - self.nchar +1 #include constant
        
        self.prodid = np.tile( np.arange(self.nprod), self.nmkt)
        self.mktid = np.repeat( np.arange(self.nmkt), self.nprod )
        
        self.mkt_dum = self.CreateDummy(self.mktid)
        self.prod_dum = self.CreateDummy(self.prodid)
        self.col_dum = np.tile(self.colmat,[self.nmkt,1])
        
        self.var_xi = var_xi
        self.var_lambda = var_lambda
        
        self.var_x = var_x
        self.var_x_cost = var_x_cost
        self.cov_x = cov_x
        self.cov_x_cost = cov_x_cost
        self.mean_x = mean_x
        self.mean_x_cost = mean_x_cost
        
        self.flag_char_dyn = flag_char_dyn
        self.flag_char_cost_dyn = flag_char_cost_dyn
        
        
        #Initialize the error flags
        self.flag_OptFailed = 0
        self.flag_NegativeMC = 0
        self.flag_DeltaSingular = 0
    
    def CreateChar(self):
        m = self.mean_x * np.ones(self.nchar-1)
        v = self.cov_x * np.ones([self.nchar-1,self.nchar-1])
        np.fill_diagonal(v, self.var_x )
        if self.flag_char_dyn==0:
            chars = np.random.multivariate_normal(m,v,self.nprod) #Assuming static char
            chars = np.c_[np.ones([self.nprod,1]), chars]
            self.char = chars[self.prodid]
        if self.flag_char_dyn==1:
            chars = np.random.multivariate_normal(m,v,self.nobs) #Assuming dynamic char
            chars = np.c_[np.ones([self.nobs,1]), chars]
            self.char = chars
            
    def CreateCharCost(self): 
        m = self.mean_x_cost * np.ones(self.nchar_cost-1)
        v = self.cov_x_cost * np.ones([self.nchar_cost-1,self.nchar_cost-1])
        np.fill_diagonal(v, self.var_x_cost )
        if self.flag_char_cost_dyn == 0:
            chars_cost = np.random.multivariate_normal(m,v,self.nprod) #Assuming static char
            chars_cost = np.c_[np.ones([self.nprod,1]), chars_cost]
            self.char_cost = chars_cost[self.prodid]
        if self.flag_char_cost_dyn == 1:
            chars_cost = np.random.multivariate_normal(m,v,self.nobs) #Assuming dynamic char
            chars_cost = np.c_[np.ones([self.nobs,1]), chars_cost]
            self.char_cost = chars_cost
        
    def CreateErrors(self):
        m = np.zeros(2)
        v = np.array([[self.var_xi,0.],[0,self.var_lambda]])
        error = np.random.multivariate_normal(m,v,self.nobs)
        self.xi = error[:,0]
        self.lam = error[:,1]
        
    def CreateCost(self):
        para = self.Spara
        char = np.c_[self.char, self.char_cost[:,1:]]
        self.char_all = char
        self.MC = np.dot(char,para) + self.lam
        if np.any(self.MC<0):
            #sys.exit('Negative Marginal Cost')
            self.flag_NegativeMC=1
            return
        
    def Demand(self,PriceVec):
        p_char = np.c_[PriceVec,self.char]
        delta = np.dot(p_char,self.Dpara) + self.xi
        expdelta = np.exp(delta) 
        expdelta_sum = self.SumByGroupDummy(self.mkt_dum, expdelta) +1
        share = expdelta/expdelta_sum
        return share
        
    def Profit_mkt(self, PriceVec_mkt, m_id):
        mc = self.MC[self.mktid==m_id]
        prof = self.Share_mkt(PriceVec_mkt, m_id)*(PriceVec_mkt.T - mc)
        return prof
    
    def Share_mkt(self,Pricevec_mkt, m_id):
        Pricevec_mkt = Pricevec_mkt.flatten()        
        p_char = np.c_[Pricevec_mkt,self.char[self.mktid==m_id]]
        delta = np.dot(p_char,self.Dpara) + self.xi[self.mktid==m_id]
        expdelta = np.exp(delta)
        expdelta_sum = np.sum(expdelta) +1.
        share = expdelta/expdelta_sum
        return share
    
    def DeltaInv(self,Pricevec_mkt, m_id):
        Pricevec_mkt = Pricevec_mkt.flatten()
        Delta = np.empty([self.nprod,self.nprod])
        
        OwnD = self.Dpara[0]*self.Share_mkt(Pricevec_mkt, m_id)*(1. - self.Share_mkt(Pricevec_mkt, m_id)) 
        a = np.tile(self.Share_mkt(Pricevec_mkt, m_id),self.nprod).reshape([self.nprod,self.nprod])
        b = np.repeat(self.Share_mkt(Pricevec_mkt, m_id),self.nprod).reshape([self.nprod,self.nprod])
        c = - self.Dpara[0] * a*b ###Dpara[0] for price not constant
        Delta = c *self.colmat        
        np.fill_diagonal( Delta, OwnD)
        
        #self.Delta = Delta
        try:
            f = -np.linalg.inv(Delta) 
        except np.linalg.linalg.LinAlgError:
            self.flag_DeltaSingular = 1
            print('Delta singular')
            return np.ones_like(Delta).astype(int) #just to let the code run.
        return f
    
    def FOC_vec(self,Pricevec_mkt, MCvec_mkt, m_id):
        a = Pricevec_mkt.flatten()
        b = MCvec_mkt.flatten()
        deltainv = self.DeltaInv(Pricevec_mkt, m_id)
        c = np.dot(deltainv , self.Share_mkt(Pricevec_mkt, m_id) )
        return a - b - c
    
    def make_FOC_obj(self):
        def FOC_obj(Pricevec_mkt, MCvec_mkt, m_id):
            a = self.FOC_vec(Pricevec_mkt, MCvec_mkt, m_id) **2
            return np.sum(a)
        return FOC_obj
    
    def Simulate(self):
        self.CreateErrors()
        self.CreateChar()
        self.CreateCharCost()
        self.CreateCost()
        if self.flag_NegativeMC==1:
            return
        
        self.PriceEquil = np.zeros([self.nobs,1])
        self.ShareEquil = np.zeros([self.nobs,1])
        
        for i in range(self.nmkt):
            print('----------market '+str(i)+'----------')
            MCvec_mkt = self.MC[self.mktid==i]
            guess = MCvec_mkt*1.1
            print('MCvec_mkt:'+str(MCvec_mkt))
            #temp = scipy.optimize.fmin(self.make_FOC_obj(), x0=guess,\
            #ftol=1e-4, xtol=1e-4, maxiter=50000, maxfun=50000, args=(MCvec_mkt,i,),full_output=True)
            #p = temp[0]
            ###################
            bnd = np.tile( np.array([0,None]), self.nprod).reshape([self.nprod,2])
            temp = scipy.optimize.minimize(self.make_FOC_obj(), x0=guess,args=(MCvec_mkt,i,),method='SLSQP', bounds=bnd,\
            options={'ftol':1e-10, 'maxiter':5000, 'disp':'False'} \
            )
            if self.flag_DeltaSingular==1 or self.flag_OptFailed==1:
                return
            
            self.temp = temp
            #print(temp)
            p = temp.x
            ###################
            print('Pricevec_mkt:'+str(p))
            print('Markup:'+str(p-MCvec_mkt))
            if temp.fun>0.01:
                self.flag_OptFailed = 1
                print('Optimization Failed')
                return
            self.PriceEquil[self.mktid==i] = np.array([p]).T
            self.ShareEquil[self.mktid==i] = np.array([ self.Share_mkt(p, i) ]).T

        self.outsh = ( 1- self.SumByGroup(self.mktid , self.ShareEquil) ).T
        if np.any(self.outsh<.1):
            print('----Outside share close to zero----')
        if np.any(self.ShareEquil==0):
            print('----Share zero----')
            
        self.Data = {'x_demand':self.char,'x_cost_only':self.char_cost,'x_cost':self.char_all,\
        'share':self.ShareEquil,'price':self.PriceEquil,'mktid':self.mktid,'prodid':self.prodid}

        self.SaveData()
            
    def SaveData(self):
        os.chdir(os.path.dirname( os.path.abspath( __file__ ) )+'\data')
        #Data        
        np.savetxt('share_Bresnahan.csv',self.ShareEquil,delimiter=',')
        np.savetxt('x_demand_Bresnahan.csv',self.char,delimiter=',')
        np.savetxt('x_cost_Bresnahan.csv',self.char_cost,delimiter=',')
        np.savetxt('price_Bresnahan.csv',self.PriceEquil,delimiter=',')
        #Index
        np.savetxt('mktid_Bresnahan.csv',self.mktid,delimiter=',')
        np.savetxt('prodid_Bresnahan.csv',self.prodid,delimiter=',')
        #Parameter
        para_true = np.append(self.Dpara,self.Spara)
        np.savetxt('para_true_Bresnahan.csv',para_true,delimiter=',')

        os.chdir(os.path.dirname( os.path.abspath( __file__ ) ))

    
    #---Functions------
    def CreateDummy(self,groupid):
        nobs = groupid.size
        id_list = np.unique(groupid)
        id_num = id_list.size    
        groupid_dum = np.zeros([nobs,id_num])    
        for i in range(id_num):
            a = (groupid==id_list[i]).repeat(id_num).reshape([nobs,id_num])
            b = (id_list==id_list[i]).repeat(nobs).reshape([id_num,nobs]).T
            c = a*b
            groupid_dum[c] = 1        
        return groupid_dum

    def SumByGroupDummy(groupid_dummy,x):
        a = np.dot(x.T,groupid_dummy)
        b = np.dot(groupid_dummy,a.T)
        return b
        
    def SumByGroup(self,groupid,x,shrink=0):
        nobs = groupid.size
        id_list = np.unique(groupid)
        id_num = id_list.size
        if x.ndim==1:
            x = np.array([x]).T
        nchar = x.shape[1]
        if shrink==0:    
            sums = np.zeros([nobs,nchar])
            for i in range(id_num):
                a = np.sum(x[groupid==id_list[i],:],axis=0)
                sums[groupid==id_list[i],:] =a
            return sums
        if shrink==1:
            sums = np.zeros([id_num,nchar])
            for i in range(id_num):
                a = np.sum(x[groupid==id_list[i],:],axis=0)
                sums[i] = a
            return sums

    def GenColMat(self,col_group):
        nfirm = col_group.shape[0]
        groups = np.unique(col_group)
        ngroup = groups.shape[0]
        groupid_dum = self.CreateDummy(col_group)
        colmat = np.zeros([nfirm,nfirm])
        for i in range(ngroup):
            a = groupid_dum[:,i]
            colmat = colmat + np.dot( a.reshape([nfirm,1]), a.reshape([1,nfirm]) )
        return colmat


    #---------------
        
        
if __name__ == "__main__":        
    nprod = 4
    col_group = np.array([0,1,2,3])
    Dpara = np.array([-4., 10., 2., 2., 1.])
    Spara = np.array([5., .2, .2, .2, .2, .2, .2])
    
    os.chdir(os.path.dirname( os.path.abspath( __file__ ) )+'\data')
    
    os.chdir(os.path.dirname( os.path.abspath( __file__ ) ))
    
    sim1 = SimulateBresnahan(nmkt_input = 50, nprod_input = nprod,\
    col_group_input = col_group, Dpara_input = Dpara,var_xi_input = 1., var_lambda_input = 1.,\
    Spara_input = Spara,\
    var_x_input=3. , cov_x_input=.0, var_x_cost_input=1., cov_x_cost_input=.5,\
    mean_x_input = 5., mean_x_cost_input=5.,\
    flag_char_dyn_input = 1, flag_char_cost_dyn_input=1)
    sim1.Simulate()
    print('flag_DeltaSingular:'+str(sim1.flag_DeltaSingular))
    print('flag_NegativeMC:'+str(sim1.flag_NegativeMC))
    print('flag_OptFailed:'+str(sim1.flag_OptFailed))
    
    #def __init__(self,nmkt_input = 2, nchar_input = 2, nchar_cost_input = 2, nprod_input = 3, colmat_input = np.identity(3),\
    #Dpara_input = np.array([-5.,1.,-1.]),var_xi_input = .3, var_lambda_input = 1.,\
    #Spara_input = np.array([10.,1.,2.]), var_x_input=1. , cov_x_input=.5, var_x_cost_input=1., cov_x_cost_input=.5,\
    #flag_char_dyn_input = 0, flag_char_cost_dyn_input=0):
        
        
        