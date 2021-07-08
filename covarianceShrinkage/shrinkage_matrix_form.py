import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Covariance_Shrinkage:
    def __init__(self,stock_returns):
        self.stock_returns = stock_returns
        self.t = len(stock_returns)
        self.n = len(stock_returns.iloc[0,:])
        self.meanx = stock_returns.mean()
        self.x = stock_returns - self.meanx

    def get_sample_cov_mat(self):
        sample = (1/self.t)*(self.x.transpose() @ self.x)
        return sample
    
    def get_shrunk_cov_mat(self,target,constant):
        sample_covariance = self.get_sample_cov_mat()
        return constant*target + (1-constant)*sample_covariance

    def get_optimal_shrunk_cov_mat(self):
        #prior (F)
        sample = self.get_sample_cov_mat()
        var = np.diag(sample)
        sqrtvar = np.sqrt(var)
        #getting rBar
        a = np.array(self.n*[list(sqrtvar)]).transpose()
        b = sample / (a*a.transpose())
        c = np.sum(b)
        d = np.sum(c) - self.n
        rBar = d/(self.n*(self.n-1))
        #getting F
        prior = rBar*(a*a.transpose())
        np.fill_diagonal(prior,var)


        #pi-hat
        y = self.x**2
        phiMat = y.transpose()@y/self.t - 2*(self.x.transpose()@self.x)*sample/self.t + sample**2
        phi = np.sum(np.sum(phiMat))

        #rho-hat
        term1 = ((self.x**3).transpose() @ self.x) / self.t
        help_mat = self.x.transpose() @ self.x/self.t
        helpDiag = np.diag(help_mat)
        term2 = np.array(self.n*[list(helpDiag)]).transpose() * sample
        term3 = help_mat * np.array(self.n*[list(var)]).transpose()
        term4 = np.array(self.n*[list(var)]).transpose() * sample
        thetaMat = np.array(term1 - term2 - term3 +term4)
        np.fill_diagonal(thetaMat,0)
        sqrtvar1 = pd.DataFrame(sqrtvar)
        rho = np.sum(np.diag(phiMat)) + rBar * np.sum(np.sum( ((1/sqrtvar1) @ sqrtvar1.transpose()) *thetaMat ))

        #gamma-hat
        gamma = np.linalg.norm(sample-prior,'fro')**2

        #shrinkage constant
        kappa = (phi-rho)/gamma
        shrinkage = max(0,min(1,kappa/self.t))
        sigma = shrinkage*prior + (1-shrinkage)*sample
        return sigma, shrinkage

    def get_pca_cov_mat(self,var):
        sample = self.get_sample_cov_mat()
        D, U = np.linalg.eigh(sample)
        D = D[::-1]
        U = U[:,::-1]
        var_contribution = D/np.sum(D)
        cum_var_contribution = np.cumsum(var_contribution)
        threshold_index = next(x for x, val in enumerate(cum_var_contribution) if val > var)
        D_pca = np.diag(list(D[0:threshold_index+1]) + (len(D) - threshold_index-1)*[0])
        return U@D_pca@U.T

if __name__=="__main__":
    
    import pandas_datareader as web
    stock_prices = web.DataReader(['WFC','BRK-B','TMUS','F','SONY','JPM','OGZPY'],'yahoo','01-01-2019','01-01-2020')['Adj Close']
    stock_returns = stock_prices.pct_change(1).dropna()
    test = Covariance_Shrinkage(stock_returns)
    sample_cov = test.get_sample_cov_mat()
    shrunk_mat, shrink = test.get_optimal_shrunk_cov_mat()
    target_shrunk_mat = test.get_shrunk_cov_mat(target = np.diag(np.diag(sample_cov)),constant=0.3)
    print("Sample\n",sample_cov)
    print("Optimal Shrunk\n",shrunk_mat)
    print("Target Shrunk\n",target_shrunk_mat)

    