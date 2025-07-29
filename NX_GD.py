from EX2_GD import main as EX
from Utilities import plotNX
import sys

def main(linesearch_list=['tr','ls','2step'],WriteSwitch=False):

    if WriteSwitch:
        EX('AltMin',WriteSwitch=True)
        for linesearch in linesearch_list:
            EX('Newton',linesearch=linesearch,WriteSwitch=True)
    
    plotNX('GD',linesearch_list)

if __name__== "__main__":
    main()
    sys.exit()