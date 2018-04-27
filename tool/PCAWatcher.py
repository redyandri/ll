import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import plotly.plotly as py
from plotly.graph_objs import *


class PCAWatcher():
    def readCSVFile(self,path):
        dataFrame=pd.read_csv(filepath_or_buffer=path,sep=",",header=None)
        dataFrame.dropna(how="all",inplace=True)
        return dataFrame

    def standardizeData(self,dataFrame):
        (r,c)=dataFrame.shape
        data=dataFrame.ix[:,0:c-2].values
        label=dataFrame.ix[:,c-1].values
        dataStandard=StandardScaler().fit_transform(data)
        #print "arr:%s\nstd:%s" %(data[0:4],dataStandard[0:4])
        return dataStandard

    def getCovatrianceMatrix(self,data):
        data=(pd.DataFrame)(data)
        (r,c)=data.shape
        dataMean=np.mean(data,axis=0)
        covMat=((data-dataMean).T.dot((data-dataMean)))/(r-1)
        #print "COVMAT:%s" %(covMat)
        return covMat

    def getEigen(self,covMat):
        (eigenVals,eigenVecs)=np.linalg.eig(covMat)
        #print "eigval:%s...eigVec:%s" %(eigenVals,eigenVecs)
        return eigenVals,eigenVecs

    def sortEigenVals(self,eigenVals,eigenVecs):
        eig_pairs = [(np.abs(eigenVals[i]), eigenVecs[:, i]) for i in range(len(eigenVals))]
        #print "EIG PAIR:%s" %eig_pairs
        # Sort the (eigenvalue, eigenvector) tuples from high to low
        eig_pairs.sort()
        eig_pairs.reverse()

        # Visually confirm that the list is correctly sorted by decreasing eigenvalues
        # print('Eigenvalues in descending order:')
        # for i in eig_pairs:
        #     print(i[0])
        return eig_pairs

    def showVarienceExplained(self,eigenPairs):
        print [e for e,v in eigenPairs]
        tot = sum(e  for e,v in eigenPairs)
        var_exp = [(i / tot) * 100 for i,j in sorted(eigenPairs, reverse=True)]
        cum_var_exp = np.cumsum(var_exp)

        trace1 = Bar(
            x=['PC %s' % i for i in range(1, 5)],
            y=var_exp,
            showlegend=False)

        trace2 = Scatter(
            x=['PC %s' % i for i in range(1, 5)],
            y=cum_var_exp,
            name='cumulative explained variance')

        data = Data([trace1, trace2])

        layout = Layout(
            yaxis=YAxis(title='Explained variance in percent'),
            title='Explained variance by different principal components')

        fig = Figure(data=data, layout=layout)
        py.iplot(fig)

csvPath="../dataset/iris.data"
pca=PCAWatcher()
df=pca.readCSVFile(csvPath)
std=pca.standardizeData(df)
covMat=pca.getCovatrianceMatrix(std)
eigVal,eigVec=pca.getEigen(covMat)
eig_pair=pca.sortEigenVals(eigVal,eigVec)
pca.showVarienceExplained(eig_pair)

