from EX2_CTFM import main as EX
from Utilities import plotNX
import sys

def main(id_list=['AltMin','CubicBacktracking','Parallelogram'], en_list=None, WriteSwitch=True):

    if WriteSwitch:
        en1, id1=EX('AltMin',WriteSwitch=True)
        en2, id2=EX('CubicBacktracking',WriteSwitch=True)
        en3, id3=EX('Parallelogram',WriteSwitch=True)
        # for linesearch in linesearch_list:
        #     EX('Newton',linesearch=linesearch,WriteSwitch=True)
        id_list=[id1,id2,id3]
        en_list=[en1,en2,en3]
    
    plotNX('CTFM',id_list, en_list)

if __name__== "__main__":
    main()
    sys.exit()