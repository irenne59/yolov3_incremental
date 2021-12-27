from utils.utils import *

'''
Input: results.txt
Output: result.png
'''

inputPath = 'E:/research/gcp/output/model221/results_0-99.txt'
outputPath = 'E:/research/gcp/output/model221/results_0-99.png'
startEpoch = 0
endEpoch = 99

plot_results(inputPath, outputPath, startEpoch, endEpoch)