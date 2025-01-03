import numpy as np
import scipy

sim_par = np.load("git/multlearn-sns/Modelling/Recovery/simulatedParams.npy")
rec_par = np.load("git/multlearn-sns/Modelling/Recovery/recoveredParams.npy")
nll = np.load("git/multlearn-sns/Modelling/Recovery/NLL_array.npy")

print(sim_par[0,:])
print(rec_par[0,:])
print(scipy.stats.pearsonr(sim_par[:,0], rec_par[:,0]))
print(scipy.stats.pearsonr(sim_par[:,1], rec_par[:,1]))
print(scipy.stats.pearsonr(sim_par[:,2], rec_par[:,2]))
print(scipy.stats.pearsonr(sim_par[:,3], rec_par[:,3]))
print(scipy.stats.pearsonr(sim_par[:,4], rec_par[:,4]))