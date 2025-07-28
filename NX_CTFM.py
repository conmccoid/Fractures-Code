from EX2_CTFM import main as EX
from Utilities import plotNX
import sys

def main(linesearch_list=['fp','ls','2step'],WriteSwitch=False):

    if WriteSwitch:
        EX('AltMin',WriteSwitch=True)
        for linesearch in linesearch_list:
            EX('Newton',linesearch=linesearch,WriteSwitch=True)
    
    plotNX('CTFM',linesearch_list)

if __name__== "__main__":
    main()
    sys.exit()