
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


class EstimateBresnahan:    
    def __init__(self, data_col=None, data_col_test=None,\
                 ivtype=0, weighting='invA', col_group=None, colmat=None, init_guess=None, para_true=None,Display=False):
        self.data_col = data_col
        self.data_col_test = data_col_test
        self.ivtype = ivtype
        self.weighting = weighting
        self.col_group=col_group
        self.colmat=colmat
         
        self.flag_Display = Display
        
        if init_guess is not None:
            self.init_guess = init_guess
        self.para_true = para_true
                
    def CalcXi(self, Dpara):
        p_char = np.c_[self.price, self.x_demand] #Dpara[0] for price, not for constant
        return self.delta - np.dot(p_char, Dpara)
            
    def CalcLam(self, Dpara, Spara):
        MC = self.SolveMC(Dpara)
        return MC - np.dot(self.x_cost,Spara)

    def Share_mkt(self,Pricevec_mkt, m_id, Dpara):
        p_char = np.c_[Pricevec_mkt,self.char[self.mktid==m_id]]
        delta = np.dot(p_char,Dpara) + self.xi[self.mktid==m_id]
        expdelta = np.exp(delta)
        expdelta_sum = np.sum(expdelta) +1
        share = expdelta/expdelta_sum
        return share

    def DeltaInv(self,Pricevec_mkt, m_id, Dpara):
        Pricevec_mkt = Pricevec_mkt.flatten()
        Delta = np.empty([self.nprod,self.nprod])
        
        OwnD = Dpara[0]*self.share[self.mktid==m_id]*(1. - self.share[self.mktid==m_id] ) 
        a = np.tile(self.share[self.mktid==m_id] ,self.nprod).reshape([self.nprod,self.nprod])
        b = np.repeat(self.share[self.mktid==m_id] ,self.nprod).reshape([self.nprod,self.nprod])
        c = - Dpara[0] * a * b ###Dpara[0] for price not constant
        Delta = c * self.colmat        
        np.fill_diagonal( Delta, OwnD)        
        return -np.linalg.inv(Delta)

    def SolveMC(self,Dpara):
        #Calc marginal cost mc
        MC = np.zeros(self.nobs)
        for i in range(self.nmkt):
            m_id = i
            Pricevec_mkt = self.price[self.mktid==i]
            c = np.dot( self.DeltaInv(Pricevec_mkt, m_id, Dpara), self.share[self.mktid==i] ) 
            MC[self.mktid==i] = Pricevec_mkt - c            
        return MC
         
        
    def get_params(self,deep=True):
        out = {}
        out['data_col'] = self.data_col
        out['data_col_test'] = self.data_col_test
        out['ivtype'] = self.ivtype
        out['weighting'] = self.weighting
        out['col_group']=self.col_group
        out['colmat']=self.colmat
        return out
        
    def set_params(self,data_col=None, data_col_test=None,\
                 ivtype=0, weighting='invA', col_group=None, colmat=None):
        self.data_col = data_col
        self.data_col_test = data_col_test
        self.ivtype = ivtype
        self.weighting = weighting
        self.col_group=col_group
        self.colmat=colmat

        
    def fit(self, Data):
        #Model
        if self.colmat is None:
            self.colmat = self.GenColMat(self.col_group)  

        #Read Data
        print('self.data_col:'+str(self.data_col))        
        print('Data:'+str(Data))
        self.x_demand = Data[:,self.data_col['x_demand']]
        self.x_cost_only = Data[:,self.data_col['x_cost_only']]
        self.x_cost = Data[:,self.data_col['x_cost']]
        self.share = Data[:,self.data_col['share']].flatten()
        self.price = Data[:,self.data_col['price']].flatten()      
        self.mktid = Data[:,self.data_col['mktid']].flatten()
        self.prodid = Data[:,self.data_col['prodid']].flatten()
        
        self.mktid = self.ShapeID(self.mktid)
        
        #print('self.x_cost:'+str(self.x_cost))
        #print('self.x_cost.shape:'+str(self.x_cost.shape))
        print('self.mktid:'+str(self.mktid))
        print('self.mktid.shape:'+str(self.mktid.shape))
        print('self.mktid==0:'+str(self.mktid==0))

        #Get Parameters
        self.nobs = self.share.shape[0]
        self.nmkt = np.unique(self.mktid).shape[0]
        self.nprod = np.unique(self.prodid).shape[0]
        
        self.nchar_demand = self.x_demand.shape[1]
        self.nchar_cost = self.x_cost.shape[1]
        self.npara = 1+ self.nchar_demand + self.nchar_cost #+1 for price

        #Create IV
        if self.ivtype==0:
            self.IV_temp = np.zeros([self.nobs, (self.nchar_cost-1)*(self.nprod )]) 
            for i in range(self.nmkt):
                c = np.ones(self.nprod)
                for j in range(1,self.nchar_cost): #ommiting constant
                    a = self.x_cost[self.mktid==i,j]
                    b = a
                    for k in range(1,self.nprod):
                        b = np.c_[b, np.roll(a,-k)]
                    c = np.c_[c, b]
                c = np.delete(c,0,1) #drop constant
                self.IV_temp[self.mktid==i,:] = c
        if self.ivtype==1:
            self.IV_temp = self.x_cost        

        self.IV = np.tile(self.IV_temp, (2,1))
        self.niv = self.IV.shape[1] 
        
        if np.linalg.matrix_rank(self.IV)<self.niv:
            print('IV rank: '+str(np.linalg.matrix_rank(self.IV)))
            print('niv: '+str(self.niv))
            np.save('IV_error.npy',self.IV)
            np.save('x_cost_error.npy',self.x_cost)
            
            sys.exit('IV not full rank.')
        if self.weighting=='invA':
            self.invA = np.linalg.inv(np.dot(self.IV.T, self.IV))
        if self.weighting=='I':
            self.invA = np.identity(self.niv)
        #self.invA = np.linalg.solve(np.dot(self.IV.T, self.IV),np.identity(self.niv)) #(Z'*Z)^(-1) 
        try:
            np.linalg.cholesky(np.dot(self.IV.T, self.IV))
        except np.linalg.LinAlgError:
            sys.exit('np.dot(self.IV.T, self.IV) is not positive definite.')
        
        try:
            np.linalg.cholesky(self.invA)
        except np.linalg.LinAlgError:
            sys.exit('the weighting matrix is not positive definite.')
        
        
        #Calc delta
        self.outsh = ( 1- self.SumByGroup(self.mktid , self.share) ).T
        self.delta = np.log( self.share ) + np.log(self.outsh)
        #Estimation
        #init_guess = np.ones(self.npara)
        #init_guess = np.array([ -2.0, 5.5, 1.9, 2.0, 0.9, 5.3, 0.18, 0.19, 0.22, 0.19 ,0.145, 0.2])
        try:
            init_guess = self.init_guess
        except AttributeError:
            init_guess = np.ones(self.npara)
            
        #######
        self.para_sol = scipy.optimize.fmin(self.make_gmmobj(), x0=init_guess,ftol=1e-6,xtol=1e-6,maxiter=5000000 )
        #######
        ###################
        #bnd = np.tile( np.array([0,None]), self.nprod).reshape([self.nprod,2])
        #temp = scipy.optimize.minimize(self.make_gmmobj(), x0=init_guess,method='SLSQP',\
        #options={'ftol':1e-10, 'maxiter':5000,'eps':.001} \
        #)
        #self.temp = temp
        #print(temp)
        #self.para_sol = temp.x
        ###################

        self.Dpara_sol = self.para_sol[0:self.nchar_demand+1] #plus 1 for price
        self.Spara_sol = self.para_sol[self.nchar_demand+1:]
        
        self.Dpara_true = self.para_true[0:self.nchar_demand+1] #plus 1 for price
        self.Spara_true = self.para_true[self.nchar_demand+1:]
        print('Dpara_sol:'+str(self.Dpara_sol))
        print('Spara_sol:'+str(self.Spara_sol))

        xi = self.CalcXi(self.Dpara_sol)
        lam = self.CalcLam(self.Dpara_sol,self.Spara_sol)
        gmmresid = np.append(xi,lam)
        
        temp5 = np.dot(gmmresid.T, self.IV)
        f = np.dot(np.dot(temp5, self.invA),temp5.T) 
        self.GMMfit = f
        
        
    def make_gmmobj(self):
        def gmmobj(para):
            Dpara = para[0:self.nchar_demand+1] #plus 1 for price
            Spara = para[self.nchar_demand+1:]
            xi = self.CalcXi(Dpara)
            lam = self.CalcLam(Dpara,Spara)
            gmmresid = np.append(xi,lam)
            
            temp5 = np.dot(gmmresid.T, self.IV)
            f = np.dot(np.dot(temp5, self.invA),temp5.T)
            
            if self.flag_Display==1:
                print('current para: '+str(para))
                print('current GMM objective: '+str(f))
            
            if f<0:
                self.xi = xi
                self.lam = lam
                self.para_temp = para
                self.f = f
                np.save('xi_error.npy',xi)
                np.save('lam_error.npy',lam)
                np.save('para_error.npy',para)
                np.save('f_error.npy',f)
                np.save('invA_error.npy',self.invA)
                sys.exit('gmm objective negative')
                        
            return f
        return gmmobj


                                       
    #---------Functions---------        
    def SumByGroup(self, groupid, x, shrink=0):
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

    def CreateDummy(self, groupid):
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

    def GenColMat(self, col_group):
        nfirm = col_group.shape[0]
        groups = np.unique(col_group)
        ngroup = groups.shape[0]
        groupid_dum = self.CreateDummy(col_group)
        colmat = np.zeros([nfirm,nfirm])
        for i in range(ngroup):
            a = groupid_dum[:,i]
            colmat = colmat + np.dot( a.reshape([nfirm,1]), a.reshape([1,nfirm]) )
        return colmat

    def ShapeID(self, groupid):
        #Make ID continuous, start from zero
        nobs = groupid.size
        id_list = np.unique(groupid)
        id_num = id_list.size    
        newid = np.zeros(nobs)
        newid_list = np.arange(id_num)
        for i in range(id_num):
            newid[groupid==id_list[i]] = i
        return newid.astype(int)

    def RecoverData(self,data_comb, data_col):
        data_recovered = {}
        for item in data_col.keys():
            data_recovered[item] = data_comb[:,data_col[item]]
            if data_recovered[item].shape[1]==1:
                data_recovered[item]=data_recovered[item].flatten()
        return data_recovered
    
    
    #-----Calc Score----
    def score(self,Data):
        #---Init----
        #Read Data
        self.x_demand_test = Data[:,self.data_col_test['x_demand']]
        self.x_cost_only_test = Data[:,self.data_col_test['x_cost_only']]
        self.x_cost_test = Data[:,self.data_col_test['x_cost']]
        self.share_test = Data[:,self.data_col_test['share']].flatten()
        self.price_test = Data[:,self.data_col_test['price']].flatten()      
        self.mktid_test = Data[:,self.data_col_test['mktid']].flatten()
        self.prodid_test = Data[:,self.data_col_test['prodid']].flatten()

        self.mktid_test = self.ShapeID(self.mktid_test)
               
        #Get Parameters
        self.nobs_test = self.share_test.shape[0]
        self.nmkt_test = np.unique(self.mktid_test).shape[0]
        self.nprod_test = np.unique(self.prodid_test).shape[0]
        
        '''
        #Create IV
        if self.ivtype==0:
            self.IV_temp_test = np.zeros([self.nobs_test, 1+(self.nchar_cost-1)*self.nprod])
            #self.IV_temp[:,0] = 1.
            for i in range(self.nmkt_test):
                #a = self.x_cost[self.mktid==i,1:].T
                #b = np.repeat(a, self.nprod).reshape([ (self.nchar_cost - 1)*self.nprod ,self.nprod]).T
                c = np.ones(self.nprod)
                for j in range(1,self.nchar_cost):
                    a = self.x_cost_test[self.mktid_test==i,j]
                    b = a
                    for k in range(1,self.nprod):
                        b = np.c_[b, np.roll(a,-k)]
                    c = np.c_[c, b]
                self.IV_temp_test[self.mktid_test==i,:] = c
        if self.ivtype==1:
            self.IV_temp_test=self.x_cost_test

        self.IV_test = np.tile(self.IV_temp_test, (2,1))
        self.niv_test = self.IV_test.shape[1]
        '''
        #Create IV
        if self.ivtype==0:
            self.IV_temp_test = np.zeros([self.nobs_test, (self.nchar_cost-1)*(self.nprod )]) 
            for i in range(self.nmkt_test):
                c = np.ones(self.nprod)
                for j in range(1,self.nchar_cost): #ommiting constant
                    a = self.x_cost[self.mktid_test==i,j]
                    b = a
                    for k in range(1,self.nprod):
                        b = np.c_[b, np.roll(a,-k)]
                    c = np.c_[c, b]
                c = np.delete(c,0,1) #drop constant
                self.IV_temp_test[self.mktid_test==i,:] = c
        if self.ivtype==1:
            self.IV_temp = self.x_cost        

        self.IV_test = np.tile(self.IV_temp_test, (2,1))
        self.niv_test = self.IV_test.shape[1] 
        
        #---Calc Score----
        #Calc delta
        self.outsh_test = ( 1- self.SumByGroup(self.mktid_test , self.share_test) ).T
        self.delta_test = np.log( self.share_test ) + np.log(self.outsh_test)
        #Calc error term
        xi = self.CalcXi_test(self.Dpara_sol)
        lam = self.CalcLam_test(self.Dpara_sol,self.Spara_sol)
        gmmresid = np.append(xi,lam)
        
        temp5 = np.dot(gmmresid.T, self.IV_test)
        print('gmmresid.shape'+str(gmmresid.shape))
        print('self.IV_test.shape'+str(self.IV_test.shape))
        print('temp5.shape'+str(temp5.shape))
        print('self.invA.shape'+str(self.invA.shape))
        f = np.dot(np.dot(temp5, self.invA),temp5.T) #or invA_test??
        return f
        
    #---for score------
    def CalcXi_test(self, Dpara):
        p_char = np.c_[self.price_test, self.x_demand_test] #Dpara[0] for price, not for constant
        return self.delta_test - np.dot(p_char, Dpara)
            
    def CalcLam_test(self, Dpara, Spara):
        MC = self.SolveMC_test(Dpara)
        return MC - np.dot(self.x_cost_test,Spara)

    def Share_mkt_test(self,Pricevec_mkt, m_id, Dpara):
        p_char = np.c_[Pricevec_mkt,self.char_test[self.mktid_test==m_id]]
        delta = np.dot(p_char,Dpara) + self.xi_test[self.mktid_test==m_id]
        expdelta = np.exp(delta)
        expdelta_sum = np.sum(expdelta) +1
        share = expdelta/expdelta_sum
        return share

    def DeltaInv_test(self,Pricevec_mkt, m_id, Dpara):
        Pricevec_mkt = Pricevec_mkt.flatten()
        Delta = np.empty([self.nprod,self.nprod])
        
        OwnD = Dpara[0]*self.share_test[self.mktid_test==m_id]*(1. - self.share_test[self.mktid_test==m_id] ) 
        a = np.tile(self.share_test[self.mktid_test==m_id] ,self.nprod).reshape([self.nprod,self.nprod])
        b = np.repeat(self.share_test[self.mktid_test==m_id] ,self.nprod).reshape([self.nprod,self.nprod])
        c = - Dpara[0] * a * b ###Dpara[0] for price not constant
        Delta = c * self.colmat        
        np.fill_diagonal( Delta, OwnD)        
        return -np.linalg.inv(Delta)

    def SolveMC_test(self,Dpara):
        #Calc marginal cost mc
        MC = np.zeros(self.nobs_test)
        for i in range(self.nmkt_test):
            m_id = i
            Pricevec_mkt = self.price_test[self.mktid_test==i]
            c = np.dot( self.DeltaInv_test(Pricevec_mkt, m_id, Dpara), self.share_test[self.mktid_test==i] ) 
            MC[self.mktid_test==i] = Pricevec_mkt - c            
        return MC
    
        