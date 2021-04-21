# open work dirs using glob
import glob as glob
import os
import re
import sys


patt = '{}*'.format(sys.argv[1])
 
dirs = glob.glob(patt)
print (dirs)

for f in dirs:
    logs = [e for e in glob.glob(f+'/'+'*.log') if e.endswith('.log')]
    logs.sort(key=os.path.getmtime)
    newest_log = logs[-1]
    scores = []
    epochs = []
    max_score = 0

    with open(newest_log) as fi:
        for line in fi:
            if 'AP:' in line:
                ind = line.index('AP:')
                try:
                    score = float(line[ind+3:ind+9])
                except:
                    score = 0
                if score > max_score:
                    max_score = score

                    
                scores.append(score)
                
        scores.sort()

        if len(scores)!=0:
            
            print ('at file {} best mAP is {}'.format(newest_log,scores[-1]))
        


