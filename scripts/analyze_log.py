# open work dirs using glob
import glob as glob
import os
import re
patt = '5animal*245'
 
dirs = glob.glob(patt)


for f in dirs:
    logs = [e for e in glob.glob(f+'/'+'*.log') if e.endswith('.log')]
    logs.sort(key=os.path.getmtime)
    newest_log = logs[-1]
    scores = []
    epochs = []
    with open(newest_log) as fi:
        for line in fi:
            if 'AP:' in line:
                ind = line.index('AP:')
                
                scores.append(float(line[ind+3:ind+9]))
                
        scores.sort()
        
        print ('at file {} best mAP is {}'.format(newest_log,scores[-1]))
        


