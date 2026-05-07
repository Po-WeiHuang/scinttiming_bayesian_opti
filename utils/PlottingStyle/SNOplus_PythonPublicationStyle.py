import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import matplotlib.font_manager as fm

# --- FONT REGISTRATION ---
# This runs as soon as you import the script
prop_path = '/home/huangp/pytor_bayesian_opti/utils/PlottingStyle/times.ttf'

if os.path.exists(prop_path):
    fm.fontManager.addfont(prop_path)
    # We create a font properties object to use for manual labels
    prop_font = fm.FontProperties(fname=prop_path, size=26.)
    _FONT_NAME = prop_font.get_name()
else:
    print(f"CRITICAL WARNING: Font file not found at {prop_path}")
    prop_font = fm.FontProperties(family='serif', size=26.)
    _FONT_NAME = 'serif'

def SNOplus_style():
    """Apply the global rcParams for SNO+ publication style."""
    matplotlib.rcParams.update({'font.size': 18})
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = [_FONT_NAME]
    
    # Global fix for the minus sign on linear axes
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    # Standard SNO+ Publication Specs
    matplotlib.rcParams['lines.markersize'] = 3
    matplotlib.rcParams['lines.linewidth'] = 2.5
    matplotlib.rcParams['patch.linewidth'] = 2.5
    matplotlib.rcParams['axes.labelsize'] = 22
    matplotlib.rcParams['legend.fontsize'] = 18
    matplotlib.rcParams['legend.framealpha'] = 0
    matplotlib.rcParams['xtick.labelsize'] = 16
    matplotlib.rcParams['ytick.labelsize'] = 16
    matplotlib.rcParams['xtick.major.size'] = 8
    matplotlib.rcParams['xtick.major.width'] = 1.5
    matplotlib.rcParams['xtick.minor.size'] = 6.4  
    matplotlib.rcParams['xtick.minor.width'] = 0.8
    matplotlib.rcParams['ytick.major.size'] = 8
    matplotlib.rcParams['ytick.major.width'] = 1.5
    matplotlib.rcParams['ytick.minor.size'] = 4.0
    matplotlib.rcParams['ytick.minor.width'] = 0.8
    matplotlib.rcParams['axes.linewidth'] = 1.5
    matplotlib.rcParams['axes.labelpad'] = 10
    matplotlib.rcParams['figure.facecolor'] = 'white'
    matplotlib.rcParams['figure.figsize'] = (10, 10)
    matplotlib.rcParams['xtick.major.pad'] = '6'
    matplotlib.rcParams['ytick.major.pad'] = '6'
    
    # Pad margins: same as ROOT SetPadLeftMargin(0.1), SetPadRightMargin(0.1), etc.
    matplotlib.rcParams['figure.subplot.left'] = 0.1
    matplotlib.rcParams['figure.subplot.right'] = 0.9
    matplotlib.rcParams['figure.subplot.bottom'] = 0.1
    matplotlib.rcParams['figure.subplot.top'] = 0.9
    
    # Draw ticks on all four sides (like ROOT's SetPadTickX(1) and SetPadTickY(1))
    matplotlib.rcParams['xtick.top'] = True
    matplotlib.rcParams['xtick.bottom'] = True
    matplotlib.rcParams['ytick.left'] = True
    matplotlib.rcParams['ytick.right'] = True
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'

    # Ensure minor ticks are enabled by default (axes still need minorticks_on() or
    # minor locators, but this makes minor ticks visible when present)
    matplotlib.rcParams['xtick.minor.visible'] = True
    matplotlib.rcParams['ytick.minor.visible'] = True

    # Fix for Log Scale Exponents: minimize mathtext usage to avoid cursive/special fonts
    matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'
    matplotlib.rcParams['mathtext.default'] = 'regular'

def textsetting(ax1, x=0.9, y=0.9, s=''):
    """Utility to add text using the SNO+ font."""
    ax1.text(x=x, y=y, s=s, fontproperties=prop_font, 
             transform=ax1.transAxes, fontweight=1000)

def axsetting(ax1, xlabel="", ylabel="", legendloc=(0.95, 0.95)):
    """Apply specific axis formatting, labels, and the log-minus-sign fix."""
    ax1.set_xlabel(xlabel, fontproperties=prop_font, size=32, x=1, ha='right')
    ax1.set_ylabel(ylabel, fontproperties=prop_font, size=32, y=1, ha='right')

    # THE LOG FIX: Force the Log Formatter to ignore Unicode minus
    # We do this here because axsetting is called after the scale is set
    if ax1.get_xscale() == 'log':
        ax1.xaxis.set_major_formatter(mticker.LogFormatterMathtext(unicode_minus=False))
    if ax1.get_yscale() == 'log':
        ax1.yaxis.set_major_formatter(mticker.LogFormatterMathtext(unicode_minus=False))

    # Apply font to tick labels manually to ensure they use the TTF
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontproperties(prop_font)

    # Legend handling
    handles, labels = ax1.get_legend_handles_labels()
    if handles:
        ax1.legend(handles, labels, loc=legendloc, fancybox=False, 
                   numpoints=1, prop=prop_font, frameon=False)

    ax1.minorticks_on()
    ax1.tick_params(which='both', direction='in', top=True, right=True, width=1)