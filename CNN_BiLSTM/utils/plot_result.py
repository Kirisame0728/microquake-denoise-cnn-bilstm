from matplotlib import pyplot as plt
import numpy as np

def plot_figure(time, noisy_signal, clean_signal, name):
    fig, axes = plt.subplots(2, 1, sharex=True)
    fig.set_size_inches(fig.get_size_inches() * [1.3, 1.1])

    axes[0].plot(time, noisy_signal,label='Noisy signal',linewidth=0.3,color='k')
    axes[0].legend(loc='upper right')
    axes[0].margins(x=0)
    axes[0].locator_params(axis='y', nbins=5)

    axes[1].plot(time, clean_signal, label='Clean signal', linewidth=0.3, color='k')
    axes[1].legend(loc='upper right')
    axes[1].margins(x=0)
    axes[1].locator_params(axis='y', nbins=5)

    fig.text(0.5, 0.04, 'Time (s)', ha='center', va='center', fontsize=12)
    fig.text(0.06, 0.5, 'Amplitude', ha='center', va='center', rotation='vertical', fontsize=12)

    fig.savefig(name, dpi=300, bbox_inches='tight')
    plt.show()

