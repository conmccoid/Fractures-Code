from EX2_Test import main as EX
import sys
from Utilities import plotNX

def main(linesearch_list=['fp','ls','2step'],WriteSwitch=False):

    if WriteSwitch:
        EX('AltMin',WriteSwitch=True)
        for linesearch in linesearch_list:
            EX('Newton',linesearch=linesearch,WriteSwitch=True)
    
    plotNX('Test',linesearch_list)

if __name__== "__main__":
    main()
    sys.exit()