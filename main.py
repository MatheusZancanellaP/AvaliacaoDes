import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import seaborn as sns
from scipy.stats import norm
# a)
array = []

with open('saida.csv', 'r') as arq:
    for number in arq:
        array.append(int(number))
# c)
x = np.sort(array)
y = 1. * np.arange(len(array))/(len(array) - 1)
bins = [1024, 10240, 102400, 1048576, 10485760, 104857600]
labels = ['1 KB', '10 KB', '100 KB', '1 MB', '10 MB', '100 MB']
ylabels =  ['0.00','0.05', '0.10', '0.15', '0.2', '0.25']
hist, _bins = np.histogram(array, bins=bins)
cum_hist=np.cumsum(hist)
norm_cum_hist = cum_hist/float(cum_hist.max())
ax2 = plt.subplot()
ax2.set_title("Distribuição CDF")
ax2.plot(range(len(bins[1:])),norm_cum_hist, marker="s", c='black')
ax2.set_xticks(range(len(bins[1:])))
ax2.set_xticklabels(labels[1:] ,rotation=0, horizontalalignment="center")
plt.ylabel("Probabilidade")
plt.xlabel("Tamanho dos Arquivos")
plt.show()
mean = np.mean(array)
variance = np.var(array)
stddev = np.std(array)
median = np.median(array)
function_cv = lambda x: np.std(x, ddof=1) / np.mean(x) * 100
cv = function_cv(array)
# b)
first_quantile = np.quantile(array, 0.25)
second_quantile = np.quantile(array, 0.50)
third_quantile = np.quantile(array, 0.75)
one_percentile = np.percentile(array, 0.01)
ten_percentile = np.percentile(array, 0.1)
ninety_percentile = np.percentile(array, 0.90)
ninetyNine_percentile = np.percentile(array, 0.99)
print("Mean(MB):")
print(mean/1048576)
print("Variance(MB):")
print(variance/1048576)
print("Standard Deviation(MB):")
print(stddev/1048576)
print("Median(MB):")
print(median/1048576)
print("CV(MB):")
print(cv)
print(x)
# c)
data_set = pd.read_csv("saida.csv")
data = pd.DataFrame(data_set)
res = sns.kdeplot(x);
res.set_xlim(left=0);
plt.ylabel("Densidade")
plt.xlabel("Tamanho dos Arquivos")
res.set_title("Distribuiçao PDF")
res.set_xticklabels(labels[0:] ,rotation=0, horizontalalignment="center")
res.set_yticklabels(ylabels[0:] , rotation=0)
plt.show()

# d) Pela análise da CDF e da PDF, podemos observar que o tamanho dos arquivos está bem concentrado em arquivos pequenos(menores do que 1 MB). Isso ocorre, principalmente, pela alta concentração de arquivos de texto e imagens presentes na área de trabalho analisada.


