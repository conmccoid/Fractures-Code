from EX_GradientDamage import main as EX
import matplotlib.pyplot as plt
import sys

def main():
    output_AltMin=EX('AltMin')
    linesearch_list=['bt','tr','ls','2step']
    for linesearch in linesearch_list:
        output=EX('NewtonLS',linesearch)
        fig, ax=plt.subplots()
        ax.plot(output[0,:],output[1,:],label='Total inner iterations')
        ax.plot(output[0,:],output[2,:],label='Outer iterations')
        ax.set_xlabel('t')
        ax.set_ylabel('Iterations')
        ax.legend()

if __name__== "__main__":
    main()
    sys.exit()