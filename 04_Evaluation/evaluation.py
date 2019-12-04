#!/usr/bin/env python
# title           :reloadModel.py
# description     :
# author          :Michael Bowyer
# date            :20191126
# version         :0.0
# usage           :
# notes           :
# python_version  :Python 3.7.3
# ==============================================================================

# """ Generate Error Metrics for overall Training data """
# EvalCols = evaluationDf.columns.values
# MAECols = [col for col in EvalCols if 'MAE' in col]
# MAPECols = [col for col in EvalCols if 'MAPE' in col]

# MAEMean = []
# MAEStd = []
# for MAECol in MAECols:
#     MAEMean.append(evaluationDf[MAECol].mean())
#     MAEStd.append(evaluationDf[MAECol].std())

# MAPEMean = []
# MAPEStd = []
# for MAPECol in MAPECols:
#     MAPEMean.append(evaluationDf[MAPECol].mean())
#     MAPEStd.append(evaluationDf[MAPECol].std())

# plt.subplot(121)
# plt.errorbar(MAECols, MAEMean, MAEStd, capsize=15,
#              capthick=3, barsabove=True, linestyle='None')
# plt.xlabel('Predicted Months (tx, with x=number of months in future)')
# plt.ylabel('Mean Absolute Error')
# plt.title('Overall Training Mean Asolute Error Mean and Standard Deviations')
# plt.subplot(122)
# plt.errorbar(MAPECols, MAPEMean, MAPEStd, capsize=15,
#              capthick=3, barsabove=True, linestyle='None')
# plt.xlabel('Predicted Months (tx, with x=number of months in future)')
# plt.ylabel('Mean Absolute Percentage Error')
# plt.title(
#     'Overall Training Mean Asolute Percentage Error Mean and Standard Deviations')
# plt.show()

# """ Find Errors Per Neighborhoods """
# neighborhoods = list(evaluationDf['ZillowNeighborhood'].unique())
# for neighborhood in neighborhoods:
#     neighborhoodDF = evaluationDf[evaluationDf['ZillowNeighborhood']
#                                   == neighborhood]
#     MAPEMean = []
#     MAPEStd = []
#     for MAPECol in MAPECols:
#         MAPEMean.append(neighborhoodDF[MAPECol].mean())
#         MAPEStd.append(evaluationDf[MAPECol].std())
#     print(MAPEMean)
#     print(MAEMean)
# # plt.scatter(Albany['Date'],Albany['ZHVI_t0'])
