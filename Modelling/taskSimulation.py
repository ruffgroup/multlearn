from TaskDesign import task_Design
import numpy.ma as ma
import seaborn as sns
sns.set()
import pandas as pd
import scipy.stats
import numpy as np
import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

## Have to be multiples of 60
mainTrials = 60
additionalTrials = 0
taskSimulation = task_Design(mainTrials,  additionalTrials)

# filehandler = open(
#     '/Users/sbedi/Desktop/multisensory-project-rl/Human task design/filename', 'wb')
# pickle.dump(taskSimulation,  filehandler)


taskSimulation.taskStructure()
#taskSimulation.RLloops()
taskSimulation.statisticalLearning()
 
# """1000 subjects correlation plots"""
# # SubjectNum = 1000
# # PearsonCorr = np.empty((SubjectNum, 2))
# # PearsonCorr[:] = np.nan
# # for i in range(0, SubjectNum):
# #     taskSimulation = task_Design(mainTrials,  additionalTrials)
# #     taskSimulation.taskStructure()
# #     taskSimulation.RLloops()
# #     taskSimulation.statisticalLearning()
# #     print((taskSimulation.alpha, taskSimulation.beta))
# #     PearsonCorr[i] = scipy.stats.pearsonr(
# #         taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3], taskSimulation.V_option0[1:, 2, 2])
# #
# #
# # print("mean pea5", np.mean(PearsonCorr[:, 0]))
# # plt.hist(PearsonCorr[:, 0])
# # plt.axvline(np.mean(PearsonCorr[:, 0]), color='r')
# # plt.xlabel("Pearson Correlations")
# # plt.ylabel("Frequency")
# # plt.title("Hisogram of Pearson Correlations for {}+{} trails".format(taskSimulation.mainTrials, taskSimulation.additionalTrials))
# # plt.show()
# #
# # print("error", taskSimulation.errorPercentage, "%")


# """Surprise plots for statistical learning"""
# #
# #
# # plt.scatter(taskSimulation.statSurpriseRow[~np.isnan(taskSimulation.statSurpriseRow)],
# #             taskSimulation.statSurprise[~np.isnan(taskSimulation.statSurprise)])
# # plt.xlabel("row Surprise")
# # plt.ylabel("Total surprise")
# # plt.title(
# #     "Row surprise vs Total statistical surprise for {} + {} trials".format(mainTrials, additionalTrials))
# # plt.show()
# #
# # plt.scatter(taskSimulation.statSurpriseColumn[~np.isnan(taskSimulation.statSurpriseColumn)],
# #             taskSimulation.statSurprise[~np.isnan(taskSimulation.statSurprise)])
# # plt.xlabel("Column Surprise")
# # plt.ylabel("Total surprise")
# # plt.title("Column surprise vs Total statistical surprise for {} + {} trials".format(mainTrials, additionalTrials))
# # plt.show()
# #
# # plt.scatter(taskSimulation.statSurpriseRow[~np.isnan(taskSimulation.statSurpriseRow)],
# #             taskSimulation.statSurpriseColumn[~np.isnan(taskSimulation.statSurpriseColumn)])
# # plt.xlabel("Row Surprise")
# # plt.ylabel("Column surprise")
# # plt.title("Row surprise vs Column surprise for {} + {} trials".format(mainTrials, additionalTrials))
# # plt.show()

# # #
plt.plot(taskSimulation.statSurprise[~np.isnan(taskSimulation.statSurprise)])
plt.xlabel("trials")
plt.ylabel("Total surprise")
plt.title("Statistical surprise signal")
plt.show()
# # #
# # #
# # # # print(taskSimulation.statSurprise[~np.isnan(taskSimulation.statSurprise)])
# # # # plt.scatter(range(0, taskSimulation.trials+taskSimulation.additionalTrials), taskSimulation.statSurprise[~np.isnan(taskSimulation.statSurprise)])
# # # # plt.xlabel("trials")
# # # # plt.ylabel("Total surprise")
# # # # plt.title("Total statistical surprise (one value for a particular cell at each trial)")
# # # # plt.show()
# # #

plt.plot(taskSimulation.statSurprise[~np.isnan(taskSimulation.statSurprise)])
plt.plot(ma.masked_invalid(taskSimulation.statSurprise[:, 0, 0]), label="00")
plt.plot(ma.masked_invalid(taskSimulation.statSurprise[:, 0, 1]), label="01")
plt.plot(ma.masked_invalid(taskSimulation.statSurprise[:, 0, 2]), label="02")
plt.plot(ma.masked_invalid(taskSimulation.statSurprise[:, 1, 0]), label="10")
plt.plot(ma.masked_invalid(taskSimulation.statSurprise[:, 1, 1]), label="11")
plt.plot(ma.masked_invalid(taskSimulation.statSurprise[:, 1, 2]), label="12")
plt.plot(ma.masked_invalid(taskSimulation.statSurprise[:, 2, 0]), label="20")
plt.plot(ma.masked_invalid(taskSimulation.statSurprise[:, 2, 1]), label="21")
plt.plot(ma.masked_invalid(taskSimulation.statSurprise[:, 2, 2]), label="22")
plt.xlabel("Trials")
plt.ylabel("Total surprise")
#plt.legend(bbox_to_anchor=(0.98, 0.9))
plt.title("Statistical surprise signal")
fig1 = plt.gcf()
plt.show()
plt.draw()
# fig1.savefig('Surp.png', dpi=300, bbox_inches='tight')

# # #
# # # # Row surprise
# # #
# # # plt.plot(taskSimulation.statSurpriseRow[~np.isnan(taskSimulation.statSurpriseRow)])
# # # plt.xlabel("trials")
# # # plt.ylabel("Row surprise")
# # # plt.title("Rowwise statistical surprise (one value for a particular cell at each trial)")
# # # plt.show()
# # #
# # # # plt.scatter(range(0, taskSimulation.trials+taskSimulation.additionalTrials),
# # # #             taskSimulation.statSurpriseRow[~np.isnan(taskSimulation.statSurpriseRow)])
# # # # plt.xlabel("trials")
# # # # plt.ylabel("Row surprise")
# # # # plt.title("Rowwise surprise (one value for a particular cell at each trial)")
# # # # plt.show()
# # #
# # #
# # # # Column surprise
# # #
# # # plt.plot(taskSimulation.statSurpriseColumn[~np.isnan(taskSimulation.statSurpriseColumn)])
# # # plt.xlabel("trials")
# # # plt.ylabel("Column surprise")
# # # plt.title("Columnwise statistical surprise (one value for a particular cell at each trial)")
# # # plt.show()
# # #
# # # plt.scatter(range(0, taskSimulation.trials+taskSimulation.additionalTrials),
# # #             taskSimulation.statSurpriseColumn[~np.isnan(taskSimulation.statSurpriseColumn)])
# # # plt.xlabel("trials")
# # # plt.ylabel("Column surprise")
# # # plt.title("Columnwise surprise (one value for a particular cell at each trial)")
# # # plt.show()
# # #
# # #
# # # """Beliefs plots"""
# # # # Modality 1 = 0
plt.figure(figsize=(8, 6))
plt.plot(range(0, taskSimulation.mainTrials+taskSimulation.additionalTrials+1), taskSimulation.beliefsStat[:, 0, 0], label="1A")
plt.plot(range(0, taskSimulation.mainTrials+taskSimulation.additionalTrials+1), taskSimulation.beliefsStat[:, 0, 1], label="1B")
plt.plot(range(0, taskSimulation.mainTrials+taskSimulation.additionalTrials+1), taskSimulation.beliefsStat[:, 0, 2], label="1C")
plt.title("Learning of statistical structure (beliefs) by Bayesian observer")
plt.xlabel("trials")
plt.ylabel("Beliefs of probabilities co-occurence")
#plt.show()
# # Modality 1 = 1
plt.plot(range(0, taskSimulation.mainTrials+taskSimulation.additionalTrials+1), taskSimulation.beliefsStat[:, 1, 0], label="2A")
plt.plot(range(0, taskSimulation.mainTrials+taskSimulation.additionalTrials+1), taskSimulation.beliefsStat[:, 1, 1], label="2B")
plt.plot(range(0, taskSimulation.mainTrials+taskSimulation.additionalTrials+1), taskSimulation.beliefsStat[:, 1, 2], label="2C")
#plt.title("Learning of statistical structure by Bayesian observer for 2 visual modality")
#plt.xlabel("trials")
#plt.ylabel("Beliefs of probabilities co-occurence")
#plt.show()
# # Modality 1 = 2
plt.plot(range(0, taskSimulation.mainTrials+taskSimulation.additionalTrials+1), taskSimulation.beliefsStat[:, 2, 0], label="3A")
plt.plot(range(0, taskSimulation.mainTrials+taskSimulation.additionalTrials+1), taskSimulation.beliefsStat[:, 2, 1], label="3B")
plt.plot(range(0, taskSimulation.mainTrials+taskSimulation.additionalTrials+1), taskSimulation.beliefsStat[:, 2, 2], label="3C")
#plt.title("Statistical learning of 9 combinations")
#plt.xlabel("trials")
#plt.ylabel("Beliefs")
plt.legend(bbox_to_anchor=(0.99, 0.65))
plt.axhline(y=taskSimulation.modality0C*0.33, color='r', linestyle='-')
plt.axhline(y=taskSimulation.modality1B*0.33, color='g', linestyle='-')
plt.axhline(y=taskSimulation.modality1C*0.33, color='b', linestyle='-')
fig1 = plt.gcf()
plt.show()
plt.draw()
# fig1.savefig('beliefs.png', dpi=300, bbox_inches='tight')

# """Reinforcement Learning Plots"""


# plt.plot(range(0, taskSimulation.mainTrials+1), taskSimulation.V_option0[:, 0, 0], label="1A")
# plt.plot(range(0, taskSimulation.mainTrials+1), taskSimulation.V_option0[:, 0, 1], label="1B")
# plt.plot(range(0, taskSimulation.mainTrials+1), taskSimulation.V_option0[:, 0, 2], label="1C")
# plt.plot(range(0, taskSimulation.mainTrials+1), taskSimulation.V_option0[:, 1, 0], label="2A")
# plt.plot(range(0, taskSimulation.mainTrials+1), taskSimulation.V_option0[:, 1, 1], label="2B")
# plt.plot(range(0, taskSimulation.mainTrials+1), taskSimulation.V_option0[:, 1, 2], label="2C")
# plt.plot(range(0, taskSimulation.mainTrials+1), taskSimulation.V_option0[:, 2, 0], label="3A")
# plt.plot(range(0, taskSimulation.mainTrials+1), taskSimulation.V_option0[:, 2, 1], label="3B")
# plt.plot(range(0, taskSimulation.mainTrials+1), taskSimulation.V_option0[:, 2, 2], label="3C")
# plt.legend(bbox_to_anchor=(1.15, 0.65))
# plt.xlabel("Trials")
# plt.ylabel("Successful action value")
# plt.title("Reinforcement Learning, omega = {:.2f}, beta = {:.2f}".format(taskSimulation.omega, taskSimulation.beta))
# fig1 = plt.gcf()
# plt.show()
# plt.draw()
# fig1.savefig('RL.png', dpi=300, bbox_inches='tight')


# print(taskSimulation.rewardPE[~np.isnan(taskSimulation.rewardPE)])
# print(np.diff(taskSimulation.alphaPearce))

#plt.scatter(taskSimulation.rewardPE[~np.isnan(taskSimulation.rewardPE)][1:], np.diff(taskSimulation.alphaPearce))
#plt.show()


# plt.plot(range(0, taskSimulation.mainTrials+1), taskSimulation.V_option1[:, 0, 0], label="1A")
# plt.plot(range(0, taskSimulation.mainTrials+1), taskSimulation.V_option1[:, 0, 1], label="1B")
# plt.plot(range(0, taskSimulation.mainTrials+1), taskSimulation.V_option1[:, 0, 2], label="1C")
# plt.plot(range(0, taskSimulation.mainTrials+1), taskSimulation.V_option1[:, 1, 0], label="2A")
# plt.plot(range(0, taskSimulation.mainTrials+1), taskSimulation.V_option1[:, 1, 1], label="2B")
# plt.plot(range(0, taskSimulation.mainTrials+1), taskSimulation.V_option1[:, 1, 2], label="2C")
# plt.plot(range(0, taskSimulation.mainTrials+1), taskSimulation.V_option1[:, 2, 0], label="3A")
# plt.plot(range(0, taskSimulation.mainTrials+1), taskSimulation.V_option1[:, 2, 1], label="3B")
# plt.plot(range(0, taskSimulation.mainTrials+1), taskSimulation.V_option1[:, 2, 2], label="3C")
# plt.legend(bbox_to_anchor=(1.15, 0.65))
# plt.xlabel("Trials")
# plt.ylabel("Unsuccessful action value")
# plt.title("Reinforcement Learning, alpha = {:.2f}, beta = {:.2f}".format(taskSimulation.alpha, taskSimulation.beta))
# fig1 = plt.gcf()
# plt.show()
# plt.draw()
# fig1.savefig('RL2.png', dpi=300, bbox_inches='tight')


# """Reward Prediction error"""

# plt.plot(range(0, taskSimulation.mainTrials), taskSimulation.rewardPE[~np.isnan(taskSimulation.rewardPE)])
# # plt.plot(range(0, taskSimulation.mainTrials), taskSimulation.rewardPE[:, 0, 0], label="00")
# # plt.plot(range(0, taskSimulation.mainTrials), taskSimulation.rewardPE[:, 0, 1], label="01")
# # plt.plot(range(0, taskSimulation.mainTrials), taskSimulation.rewardPE[:, 0, 2], label="02")
# # plt.plot(range(0, taskSimulation.mainTrials), taskSimulation.rewardPE[:, 1, 0], label="10")
# # plt.plot(range(0, taskSimulation.mainTrials), taskSimulation.rewardPE[:, 1, 1], label="11")
# # plt.plot(range(0, taskSimulation.mainTrials), taskSimulation.rewardPE[:, 1, 2], label="12")
# # plt.plot(range(0, taskSimulation.mainTrials), taskSimulation.rewardPE[:, 2, 0], label="20")
# # plt.plot(range(0, taskSimulation.mainTrials), taskSimulation.rewardPE[:, 2, 1], label="21")
# # plt.plot(range(0, taskSimulation.mainTrials), taskSimulation.rewardPE[:, 2, 2], label="22")
# # plt.legend(bbox_to_anchor=(1.15, 0.65))
# plt.xlabel("Trials")
# plt.ylabel("Reward Prediction error")
# plt.title("RPE for Q-learning model with\nalpha = {:.2f}, beta = {:.2f}".format(taskSimulation.alpha, taskSimulation.beta))
# fig1 = plt.gcf()
# plt.show()
# plt.draw()
# fig1.savefig('RPE.png', dpi=300, bbox_inches='tight')


# """Surprise vs prediction errors"""
# # # Cell 00
# # plt.scatter(taskSimulation.statSurprise[taskSimulation.additionalTrials:, 0, 0], taskSimulation.rewardPE[:, 0, 0])
# # PearsonCorr = scipy.stats.pearsonr(taskSimulation.statSurprise[taskSimulation.additionalTrials:, 0, 0][~np.isnan(
# #     taskSimulation.statSurprise[taskSimulation.additionalTrials:, 0, 0])], taskSimulation.rewardPE[:, 0, 0][~np.isnan(taskSimulation.rewardPE[:, 0, 0])])
# # SpearmanCorr = scipy.stats.spearmanr(ma.masked_invalid(
# #     taskSimulation.statSurprise[taskSimulation.additionalTrials:, 0, 0]), ma.masked_invalid(taskSimulation.rewardPE[:, 0, 0]), nan_policy='omit')
# # regr = scipy.stats.linregress(taskSimulation.statSurprise[taskSimulation.additionalTrials:, 0, 0][~np.isnan(
# #     taskSimulation.statSurprise[taskSimulation.additionalTrials:, 0, 0])], taskSimulation.rewardPE[:, 0, 0][~np.isnan(taskSimulation.rewardPE[:, 0, 0])])
# # x = np.linspace(np.nanmin(taskSimulation.statSurprise[taskSimulation.additionalTrials:, 0, 0]), np.nanmax(
# #     taskSimulation.statSurprise[taskSimulation.additionalTrials:, 0, 0]), 500)
# # plt.plot(x, regr[0]*x+regr[1], '-',  color='k')
# # plt.xlabel("Statistical surprise (only counting main trials)")
# # plt.ylabel("Reward prediction errors")
# # plt.title(
# #     "PearsonCorrelation :{:.4f}, p-value :{:.4f}\n  cell 00".format(PearsonCorr[0], PearsonCorr[1]), fontweight="bold", fontsize=15)
# # plt.show()
# #
# # # Cell 01
# # plt.scatter(taskSimulation.statSurprise[taskSimulation.additionalTrials:, 0, 1], taskSimulation.rewardPE[:, 0, 1])
# # PearsonCorr = scipy.stats.pearsonr(taskSimulation.statSurprise[taskSimulation.additionalTrials:, 0, 1][~np.isnan(
# #     taskSimulation.statSurprise[taskSimulation.additionalTrials:, 0, 1])], taskSimulation.rewardPE[:, 0, 1][~np.isnan(taskSimulation.rewardPE[:, 0, 1])])
# # SpearmanCorr = scipy.stats.spearmanr(ma.masked_invalid(
# #     taskSimulation.statSurprise[taskSimulation.additionalTrials:, 0, 1]), ma.masked_invalid(taskSimulation.rewardPE[:, 0, 1]), nan_policy='omit')
# # regr = scipy.stats.linregress(taskSimulation.statSurprise[taskSimulation.additionalTrials:, 0, 1][~np.isnan(
# #     taskSimulation.statSurprise[taskSimulation.additionalTrials:, 0, 1])], taskSimulation.rewardPE[:, 0, 1][~np.isnan(taskSimulation.rewardPE[:, 0, 1])])
# # x = np.linspace(np.nanmin(taskSimulation.statSurprise[taskSimulation.additionalTrials:, 0, 1]), np.nanmax(
# #     taskSimulation.statSurprise[taskSimulation.additionalTrials:, 0, 1]), 500)
# # plt.plot(x, regr[0]*x+regr[1], '-',  color='k')
# # plt.xlabel("Statistical surprise (only main trials)")
# # plt.ylabel("Reward prediction errors")
# # plt.title(
# #     "PearsonCorrelation :{:.4f}, p-value :{:.4f}\n  cell 01".format(PearsonCorr[0], PearsonCorr[1]), fontweight="bold", fontsize=15)
# # plt.show()
# #
# # # Cell 02
# # plt.scatter(taskSimulation.statSurprise[taskSimulation.additionalTrials:, 0, 2], taskSimulation.rewardPE[:, 0, 2])
# # PearsonCorr = scipy.stats.pearsonr(taskSimulation.statSurprise[taskSimulation.additionalTrials:, 0, 2][~np.isnan(
# #     taskSimulation.statSurprise[taskSimulation.additionalTrials:, 0, 2])], taskSimulation.rewardPE[:, 0, 2][~np.isnan(taskSimulation.rewardPE[:, 0, 2])])
# # SpearmanCorr = scipy.stats.spearmanr(ma.masked_invalid(
# #     taskSimulation.statSurprise[taskSimulation.additionalTrials:, 0, 2]), ma.masked_invalid(taskSimulation.rewardPE[:, 0, 2]), nan_policy='omit')
# # regr = scipy.stats.linregress(taskSimulation.statSurprise[taskSimulation.additionalTrials:, 0, 2][~np.isnan(
# #     taskSimulation.statSurprise[taskSimulation.additionalTrials:, 0, 2])], taskSimulation.rewardPE[:, 0, 2][~np.isnan(taskSimulation.rewardPE[:, 0, 2])])
# # x = np.linspace(np.nanmin(taskSimulation.statSurprise[taskSimulation.additionalTrials:, 0, 2]), np.nanmax(
# #     taskSimulation.statSurprise[taskSimulation.additionalTrials:, 0, 2]), 500)
# # plt.plot(x, regr[0]*x+regr[1], '-',  color='k')
# # plt.xlabel("Statistical surprise")
# # plt.ylabel("Reward prediction errors")
# # plt.title(
# #     "PearsonCorrelation :{:.4f}, p-value :{:.4f}\n  cell 02".format(PearsonCorr[0], PearsonCorr[1]), fontweight="bold", fontsize=15)
# # plt.show()
# #
# # # Cell 10
# # plt.scatter(taskSimulation.statSurprise[taskSimulation.additionalTrials:, 1, 0], taskSimulation.rewardPE[:, 1, 0])
# # PearsonCorr = scipy.stats.pearsonr(taskSimulation.statSurprise[taskSimulation.additionalTrials:, 1, 0][~np.isnan(
# #     taskSimulation.statSurprise[taskSimulation.additionalTrials:, 1, 0])], taskSimulation.rewardPE[:, 1, 0][~np.isnan(taskSimulation.rewardPE[:, 1, 0])])
# # SpearmanCorr = scipy.stats.spearmanr(ma.masked_invalid(
# #     taskSimulation.statSurprise[taskSimulation.additionalTrials:, 1, 0]), ma.masked_invalid(taskSimulation.rewardPE[:, 1, 0]), nan_policy='omit')
# # regr = scipy.stats.linregress(taskSimulation.statSurprise[taskSimulation.additionalTrials:, 1, 0][~np.isnan(
# #     taskSimulation.statSurprise[taskSimulation.additionalTrials:, 1, 0])], taskSimulation.rewardPE[:, 1, 0][~np.isnan(taskSimulation.rewardPE[:, 1, 0])])
# # x = np.linspace(np.nanmin(taskSimulation.statSurprise[taskSimulation.additionalTrials:, 1, 0]), np.nanmax(
# #     taskSimulation.statSurprise[taskSimulation.additionalTrials:, 1, 0]), 500)
# # plt.plot(x, regr[0]*x+regr[1], '-',  color='k')
# # plt.xlabel("Statistical surprise")
# # plt.ylabel("Reward prediction errors")
# # plt.title(
# #     "PearsonCorrelation :{:.4f}, p-value :{:.4f}\n  cell 10".format(PearsonCorr[0], PearsonCorr[1]), fontweight="bold", fontsize=15)
# # plt.show()
# #
# # # Cell 11
# # plt.scatter(taskSimulation.statSurprise[taskSimulation.additionalTrials:, 1, 1], taskSimulation.rewardPE[:, 1, 1])
# # PearsonCorr = scipy.stats.pearsonr(taskSimulation.statSurprise[taskSimulation.additionalTrials:, 1, 1][~np.isnan(
# #     taskSimulation.statSurprise[taskSimulation.additionalTrials:, 1, 1])], taskSimulation.rewardPE[:, 1, 1][~np.isnan(taskSimulation.rewardPE[:, 1, 1])])
# # SpearmanCorr = scipy.stats.spearmanr(ma.masked_invalid(
# #     taskSimulation.statSurprise[taskSimulation.additionalTrials:, 1, 1]), ma.masked_invalid(taskSimulation.rewardPE[:, 1, 1]), nan_policy='omit')
# # regr = scipy.stats.linregress(taskSimulation.statSurprise[taskSimulation.additionalTrials:, 1, 1][~np.isnan(
# #     taskSimulation.statSurprise[taskSimulation.additionalTrials:, 1, 1])], taskSimulation.rewardPE[:, 1, 1][~np.isnan(taskSimulation.rewardPE[:, 1, 1])])
# # x = np.linspace(np.nanmin(taskSimulation.statSurprise[taskSimulation.additionalTrials:, 1, 1]), np.nanmax(
# #     taskSimulation.statSurprise[taskSimulation.additionalTrials:, 1, 1]), 500)
# # plt.plot(x, regr[0]*x+regr[1], '-',  color='k')
# # plt.xlabel("Statistical surprise")
# # plt.ylabel("Reward prediction errors")
# # plt.title(
# #     "PearsonCorrelation :{:.4f}, p-value :{:.4f}\n  cell 11".format(PearsonCorr[0], PearsonCorr[1]), fontweight="bold", fontsize=15)
# # plt.show()
# #
# # # Cell 12
# # plt.scatter(taskSimulation.statSurprise[taskSimulation.additionalTrials:, 1, 2], taskSimulation.rewardPE[:, 1, 2])
# # PearsonCorr = scipy.stats.pearsonr(taskSimulation.statSurprise[taskSimulation.additionalTrials:, 1, 2][~np.isnan(
# #     taskSimulation.statSurprise[taskSimulation.additionalTrials:, 1, 2])], taskSimulation.rewardPE[:, 1, 2][~np.isnan(taskSimulation.rewardPE[:, 1, 2])])
# # SpearmanCorr = scipy.stats.spearmanr(ma.masked_invalid(
# #     taskSimulation.statSurprise[taskSimulation.additionalTrials:, 1, 2]), ma.masked_invalid(taskSimulation.rewardPE[:, 1, 2]), nan_policy='omit')
# # regr = scipy.stats.linregress(taskSimulation.statSurprise[taskSimulation.additionalTrials:, 1, 2][~np.isnan(
# #     taskSimulation.statSurprise[taskSimulation.additionalTrials:, 1, 2])], taskSimulation.rewardPE[:, 1, 2][~np.isnan(taskSimulation.rewardPE[:, 1, 2])])
# # x = np.linspace(np.nanmin(taskSimulation.statSurprise[taskSimulation.additionalTrials:, 1, 2]), np.nanmax(
# #     taskSimulation.statSurprise[taskSimulation.additionalTrials:, 1, 2]), 500)
# # plt.plot(x, regr[0]*x+regr[1], '-',  color='k')
# # plt.xlabel("Statistical surprise")
# # plt.ylabel("Reward prediction errors")
# # plt.title(
# #     "PearsonCorrelation :{:.4f}, p-value :{:.4f}\n  cell 12".format(PearsonCorr[0], PearsonCorr[1]), fontweight="bold", fontsize=15)
# # plt.show()
# #
# # # Cell 20
# # plt.scatter(taskSimulation.statSurprise[taskSimulation.additionalTrials:, 2, 0], taskSimulation.rewardPE[:, 2, 0])
# # PearsonCorr = scipy.stats.pearsonr(taskSimulation.statSurprise[taskSimulation.additionalTrials:, 2, 0][~np.isnan(
# #     taskSimulation.statSurprise[taskSimulation.additionalTrials:, 2, 0])], taskSimulation.rewardPE[:, 2, 0][~np.isnan(taskSimulation.rewardPE[:, 2, 0])])
# # SpearmanCorr = scipy.stats.spearmanr(ma.masked_invalid(
# #     taskSimulation.statSurprise[taskSimulation.additionalTrials:, 2, 0]), ma.masked_invalid(taskSimulation.rewardPE[:, 2, 0]), nan_policy='omit')
# # regr = scipy.stats.linregress(taskSimulation.statSurprise[taskSimulation.additionalTrials:, 2, 0][~np.isnan(
# #     taskSimulation.statSurprise[taskSimulation.additionalTrials:, 2, 0])], taskSimulation.rewardPE[:, 2, 0][~np.isnan(taskSimulation.rewardPE[:, 2, 0])])
# # x = np.linspace(np.nanmin(taskSimulation.statSurprise[taskSimulation.additionalTrials:, 2, 0]), np.nanmax(
# #     taskSimulation.statSurprise[taskSimulation.additionalTrials:, 2, 0]), 500)
# # plt.plot(x, regr[0]*x+regr[1], '-',  color='k')
# # plt.xlabel("Statistical surprise")
# # plt.ylabel("Reward prediction errors")
# # plt.title(
# #     "PearsonCorrelation :{:.4f}, p-value :{:.4f}\n  cell 20".format(PearsonCorr[0], PearsonCorr[1]), fontweight="bold", fontsize=15)
# # plt.show()
# #
# # # Cell 21
# # plt.scatter(taskSimulation.statSurprise[taskSimulation.additionalTrials:, 2, 1], taskSimulation.rewardPE[:, 2, 1])
# # PearsonCorr = scipy.stats.pearsonr(taskSimulation.statSurprise[taskSimulation.additionalTrials:, 2, 1][~np.isnan(
# #     taskSimulation.statSurprise[taskSimulation.additionalTrials:, 2, 1])], taskSimulation.rewardPE[:, 2, 1][~np.isnan(taskSimulation.rewardPE[:, 2, 1])])
# # SpearmanCorr = scipy.stats.spearmanr(ma.masked_invalid(
# #     taskSimulation.statSurprise[taskSimulation.additionalTrials:, 2, 1]), ma.masked_invalid(taskSimulation.rewardPE[:, 2, 1]), nan_policy='omit')
# # regr = scipy.stats.linregress(taskSimulation.statSurprise[taskSimulation.additionalTrials:, 2, 1][~np.isnan(
# #     taskSimulation.statSurprise[taskSimulation.additionalTrials:, 2, 1])], taskSimulation.rewardPE[:, 2, 1][~np.isnan(taskSimulation.rewardPE[:, 2, 1])])
# # x = np.linspace(np.nanmin(taskSimulation.statSurprise[taskSimulation.additionalTrials:, 2, 1]), np.nanmax(
# #     taskSimulation.statSurprise[taskSimulation.additionalTrials:, 2, 1]), 500)
# # plt.plot(x, regr[0]*x+regr[1], '-',  color='k')
# # plt.xlabel("Statistical surprise")
# # plt.ylabel("Reward prediction errors")
# # plt.title(
# #     "PearsonCorrelation :{:.4f}, p-value :{:.4f}\n  cell 21".format(PearsonCorr[0], PearsonCorr[1]), fontweight="bold", fontsize=15)
# # plt.show()
# #
# # # Cell 22
# # plt.scatter(taskSimulation.statSurprise[taskSimulation.additionalTrials:, 2, 2], taskSimulation.rewardPE[:, 2, 2])
# # PearsonCorr = scipy.stats.pearsonr(taskSimulation.statSurprise[taskSimulation.additionalTrials:, 2, 2][~np.isnan(
# #     taskSimulation.statSurprise[taskSimulation.additionalTrials:, 2, 2])], taskSimulation.rewardPE[:, 2, 2][~np.isnan(taskSimulation.rewardPE[:, 2, 2])])
# # SpearmanCorr = scipy.stats.spearmanr(ma.masked_invalid(
# #     taskSimulation.statSurprise[taskSimulation.additionalTrials:, 2, 2]), ma.masked_invalid(taskSimulation.rewardPE[:, 2, 2]), nan_policy='omit')
# # regr = scipy.stats.linregress(taskSimulation.statSurprise[taskSimulation.additionalTrials:, 2, 2][~np.isnan(
# #     taskSimulation.statSurprise[taskSimulation.additionalTrials:, 2, 2])], taskSimulation.rewardPE[:, 2, 2][~np.isnan(taskSimulation.rewardPE[:, 2, 2])])
# # x = np.linspace(np.nanmin(taskSimulation.statSurprise[taskSimulation.additionalTrials:, 2, 2]), np.nanmax(
# #     taskSimulation.statSurprise[taskSimulation.additionalTrials:, 2, 2]), 500)
# # plt.plot(x, regr[0]*x+regr[1], '-',  color='k')
# # plt.xlabel("Statistical surprise")
# # plt.ylabel("Reward prediction errors")
# # plt.title(
# #     "PearsonCorrelation :{:.4f}, p-value :{:.4f}\n  cell 22".format(PearsonCorr[0], PearsonCorr[1]), fontweight="bold", fontsize=15)
# # plt.show()


# # print(ma.masked_invalid(taskSimulation.statSurprise[:, 2, 2]))
# # print(np.mean(ma.masked_invalid(taskSimulation.statSurprise[:, 2, 2])))
# # print(ma.masked_invalid(taskSimulation.rewardPE[:, 2, 2]))
# # taskSimulation.statSurprise[:, 2, 2][~np.isnan(taskSimulation.statSurprise[:, 2, 2])]


# """Correlations"""


# # All cells
PearsonCorr = scipy.stats.pearsonr(taskSimulation.statSurprise[taskSimulation.additionalTrials:, :, :][~np.isnan(
     taskSimulation.statSurprise[taskSimulation.additionalTrials:, :, :])], taskSimulation.rewardPE[:, :, :][~np.isnan(taskSimulation.rewardPE[:, :, :])])
print("pearson", PearsonCorr)

# SpearmanCorr = scipy.stats.spearmanr(ma.masked_invalid(
#     taskSimulation.statSurprise[taskSimulation.additionalTrials:, :, :]), ma.masked_invalid(taskSimulation.rewardPE[:, :, :]), nan_policy='omit')
# print("spearman", SpearmanCorr)

regr = scipy.stats.linregress(taskSimulation.statSurprise[taskSimulation.additionalTrials:, 2, 2][~np.isnan(
    taskSimulation.statSurprise[taskSimulation.additionalTrials:, 2, 2])], taskSimulation.rewardPE[:, 2, 2][~np.isnan(taskSimulation.rewardPE[:, 2, 2])])
print("regression", regr)
print(regr[1])

# r22 = scipy.stats.spearmanr((taskSimulation.statSurprise[taskSimulation.additionalTrials:, 2, 2], taskSimulation.rewardPE[:, 2, 2]))
# print("r", r22)


# df = pd.DataFrame(taskSimulation.simulatedData,  columns=['visual', 'auditory', 'prederror', 'surprise'])
# print(df)

# df.to_csv('/Users/sbedi/Desktop/multisensory-project-rl/Human task design/simulated.csv')
#
# print(taskSimulation.simulatedData)

#
plt.scatter(taskSimulation.simulatedData[taskSimulation.additionalTrials:, 2], taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3])
PearsonCorr = scipy.stats.pearsonr(
    taskSimulation.simulatedData[taskSimulation.additionalTrials:, 2], taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3])
SpearmanCorr = scipy.stats.spearmanr(
    taskSimulation.simulatedData[taskSimulation.additionalTrials:, 2], taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3])
regr = scipy.stats.linregress(
    taskSimulation.simulatedData[taskSimulation.additionalTrials:, 2], taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3])
x = np.linspace(np.min(taskSimulation.simulatedData[taskSimulation.additionalTrials:, 2]), np.max(
    taskSimulation.simulatedData[taskSimulation.additionalTrials:, 2]), 500)
plt.plot(x, regr[0]*x+regr[1], '-',  color='k')
plt.xlabel("Reward prediction errors")
plt.ylabel("Statistical surprise")
plt.title("All cells - PearsonCorr:{:.4f}, p-value :{:.4f}\nalpha = {}, beta = {}, error% = {:.1f}".format(
    PearsonCorr[0], PearsonCorr[1], taskSimulation.alpha, taskSimulation.beta, taskSimulation.errorPercentage), fontweight="bold", fontsize=15)
fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig('corr.png', dpi=300, bbox_inches='tight')

# """Value vs surprise"""
# # plt.scatter(taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3], taskSimulation.V_option1[1:, 0, 0])
# # PearsonCorr = scipy.stats.pearsonr(
# #     taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3], taskSimulation.V_option1[1:, 0, 0])
# # SpearmanCorr = scipy.stats.spearmanr(
# #     taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3], taskSimulation.V_option1[1:, 0, 0])
# # regr = scipy.stats.linregress(
# #     taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3], taskSimulation.V_option1[1:, 0, 0])
# # x = np.linspace(np.min(taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3]), np.max(
# #     taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3]), 500)
# # plt.plot(x, regr[0]*x+regr[1], '-',  color='k')
# # plt.xlabel("Surprise")
# # plt.ylabel("Value for option 1")
# # plt.title("Cell00 - PearsonCorr:{:.4f}, p-value :{:.4f}\nalpha = {}, beta = {}, error% = {:.1f}".format(
# #     PearsonCorr[0], PearsonCorr[1], taskSimulation.alpha, taskSimulation.beta, taskSimulation.errorPercentage), fontweight="bold", fontsize=15)
# # plt.show()
# #
# # plt.scatter(taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3], taskSimulation.V_option1[1:, 0, 1])
# # PearsonCorr = scipy.stats.pearsonr(
# #     taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3], taskSimulation.V_option1[1:, 0, 1])
# # SpearmanCorr = scipy.stats.spearmanr(
# #     taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3], taskSimulation.V_option1[1:, 0, 1])
# # regr = scipy.stats.linregress(
# #     taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3], taskSimulation.V_option1[1:, 0, 1])
# # x = np.linspace(np.min(taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3]), np.max(
# #     taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3]), 500)
# # plt.plot(x, regr[0]*x+regr[1], '-',  color='k')
# # plt.xlabel("Surprise")
# # plt.ylabel("Value for option 1")
# # plt.title("Cell01 - PearsonCorr:{:.4f}, p-value :{:.4f}\nalpha = {}, beta = {}, error% = {:.1f}".format(
# #     PearsonCorr[0], PearsonCorr[1], taskSimulation.alpha, taskSimulation.beta, taskSimulation.errorPercentage), fontweight="bold", fontsize=15)
# # plt.show()
# #
# #
# # plt.scatter(taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3], taskSimulation.V_option1[1:, 0, 2])
# # PearsonCorr = scipy.stats.pearsonr(
# #     taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3], taskSimulation.V_option1[1:, 0, 2])
# # SpearmanCorr = scipy.stats.spearmanr(
# #     taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3], taskSimulation.V_option1[1:, 0, 2])
# # regr = scipy.stats.linregress(
# #     taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3], taskSimulation.V_option1[1:, 0, 2])
# # x = np.linspace(np.min(taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3]), np.max(
# #     taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3]), 500)
# # plt.plot(x, regr[0]*x+regr[1], '-',  color='k')
# # plt.xlabel("Surprise")
# # plt.ylabel("Value for option 1")
# # plt.title("Cell02 - PearsonCorr:{:.4f}, p-value :{:.4f}\nalpha = {}, beta = {}, error% = {:.1f}".format(
# #     PearsonCorr[0], PearsonCorr[1], taskSimulation.alpha, taskSimulation.beta, taskSimulation.errorPercentage), fontweight="bold", fontsize=15)
# # plt.show()
# #
# #
# # plt.scatter(taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3], taskSimulation.V_option1[1:, 1, 0])
# # PearsonCorr = scipy.stats.pearsonr(
# #     taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3], taskSimulation.V_option1[1:, 1, 0])
# # SpearmanCorr = scipy.stats.spearmanr(
# #     taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3], taskSimulation.V_option1[1:, 1, 0])
# # regr = scipy.stats.linregress(
# #     taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3], taskSimulation.V_option1[1:, 1, 0])
# # x = np.linspace(np.min(taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3]), np.max(
# #     taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3]), 500)
# # plt.plot(x, regr[0]*x+regr[1], '-',  color='k')
# # plt.xlabel("Surprise")
# # plt.ylabel("Value for option 1")
# # plt.title("Cell10 - PearsonCorr:{:.4f}, p-value :{:.4f}\nalpha = {}, beta = {}, error% = {:.1f}".format(
# #     PearsonCorr[0], PearsonCorr[1], taskSimulation.alpha, taskSimulation.beta, taskSimulation.errorPercentage), fontweight="bold", fontsize=15)
# # plt.show()
# #
# #
# # plt.scatter(taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3], taskSimulation.V_option1[1:, 1, 1])
# # PearsonCorr = scipy.stats.pearsonr(
# #     taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3], taskSimulation.V_option1[1:, 1, 1])
# # SpearmanCorr = scipy.stats.spearmanr(
# #     taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3], taskSimulation.V_option1[1:, 1, 1])
# # regr = scipy.stats.linregress(
# #     taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3], taskSimulation.V_option1[1:, 1, 1])
# # x = np.linspace(np.min(taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3]), np.max(
# #     taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3]), 500)
# # plt.plot(x, regr[0]*x+regr[1], '-',  color='k')
# # plt.xlabel("Surprise")
# # plt.ylabel("Value for option 1")
# # plt.title("Cell11 - PearsonCorr:{:.4f}, p-value :{:.4f}\nalpha = {}, beta = {}, error% = {:.1f}".format(
# #     PearsonCorr[0], PearsonCorr[1], taskSimulation.alpha, taskSimulation.beta, taskSimulation.errorPercentage), fontweight="bold", fontsize=15)
# # plt.show()
# #
# #
# # plt.scatter(taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3], taskSimulation.V_option1[1:, 1, 2])
# # PearsonCorr = scipy.stats.pearsonr(
# #     taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3], taskSimulation.V_option1[1:, 1, 2])
# # SpearmanCorr = scipy.stats.spearmanr(
# #     taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3], taskSimulation.V_option1[1:, 1, 2])
# # regr = scipy.stats.linregress(
# #     taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3], taskSimulation.V_option1[1:, 1, 2])
# # x = np.linspace(np.min(taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3]), np.max(
# #     taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3]), 500)
# # plt.plot(x, regr[0]*x+regr[1], '-',  color='k')
# # plt.xlabel("Surprise")
# # plt.ylabel("Value for option 1")
# # plt.title("Cell12 - PearsonCorr:{:.4f}, p-value :{:.4f}\nalpha = {}, beta = {}, error% = {:.1f}".format(
# #     PearsonCorr[0], PearsonCorr[1], taskSimulation.alpha, taskSimulation.beta, taskSimulation.errorPercentage), fontweight="bold", fontsize=15)
# # plt.show()
# #
# #
# # plt.scatter(taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3], taskSimulation.V_option1[1:, 2, 0])
# # PearsonCorr = scipy.stats.pearsonr(
# #     taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3], taskSimulation.V_option1[1:, 2, 0])
# # SpearmanCorr = scipy.stats.spearmanr(
# #     taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3], taskSimulation.V_option1[1:, 2, 0])
# # regr = scipy.stats.linregress(
# #     taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3], taskSimulation.V_option1[1:, 2, 0])
# # x = np.linspace(np.min(taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3]), np.max(
# #     taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3]), 500)
# # plt.plot(x, regr[0]*x+regr[1], '-',  color='k')
# # plt.xlabel("Surprise")
# # plt.ylabel("Value for option 1")
# # plt.title("Cell20 - PearsonCorr:{:.4f}, p-value :{:.4f}\nalpha = {}, beta = {}, error% = {:.1f}".format(
# #     PearsonCorr[0], PearsonCorr[1], taskSimulation.alpha, taskSimulation.beta, taskSimulation.errorPercentage), fontweight="bold", fontsize=15)
# # plt.show()
# #
# #
# # plt.scatter(taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3], taskSimulation.V_option1[1:, 2, 1])
# # PearsonCorr = scipy.stats.pearsonr(
# #     taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3], taskSimulation.V_option1[1:, 2, 1])
# # SpearmanCorr = scipy.stats.spearmanr(
# #     taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3], taskSimulation.V_option1[1:, 2, 1])
# # regr = scipy.stats.linregress(
# #     taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3], taskSimulation.V_option1[1:, 2, 1])
# # x = np.linspace(np.min(taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3]), np.max(
# #     taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3]), 500)
# # plt.plot(x, regr[0]*x+regr[1], '-',  color='k')
# # plt.xlabel("Surprise")
# # plt.ylabel("Value for option 1")
# # plt.title("Cell21 - PearsonCorr:{:.4f}, p-value :{:.4f}\nalpha = {}, beta = {}, error% = {:.1f}".format(
# #     PearsonCorr[0], PearsonCorr[1], taskSimulation.alpha, taskSimulation.beta, taskSimulation.errorPercentage), fontweight="bold", fontsize=15)
# # plt.show()
# #
# #
# # plt.scatter(taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3], taskSimulation.V_option1[1:, 2, 2])
# # PearsonCorr = scipy.stats.pearsonr(
# #     taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3], taskSimulation.V_option1[1:, 2, 2])
# # SpearmanCorr = scipy.stats.spearmanr(
# #     taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3], taskSimulation.V_option1[1:, 2, 2])
# # regr = scipy.stats.linregress(
# #     taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3], taskSimulation.V_option1[1:, 2, 2])
# # x = np.linspace(np.min(taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3]), np.max(
# #     taskSimulation.simulatedData[taskSimulation.additionalTrials:, 3]), 500)
# # plt.plot(x, regr[0]*x+regr[1], '-',  color='k')
# # plt.xlabel("Surprise")
# # plt.ylabel("Value for option 1")
# # plt.title("Cell22 - PearsonCorr:{:.4f}, p-value :{:.4f}\nalpha = {}, beta = {}, error% = {:.1f}".format(
# #     PearsonCorr[0], PearsonCorr[1], taskSimulation.alpha, taskSimulation.beta, taskSimulation.errorPercentage), fontweight="bold", fontsize=15)
# # plt.show()
