'''
Created on Apr 27, 2017

@author: optas
'''
import subprocess
import warnings

def run_command(cmd):
    """Run command, return output as string."""
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    return output.decode("ascii")


def grep_files(pattern, top_dir='./', match_whole_word=True, options=None):
    '''Example: grep_files('tf.norm', '../../../tf_lab/',  match_whole_word=True, options='--include=\*.py ')
    TODO: check if include, exclude works.
    '''
    def enclose_with_quotes(var):
        return '\'' + var + '\''
    
    cmd = 'grep  -rn'        # r: recursice, n: print line number

    if match_whole_word:
        cmd += 'w' 

    cmd += ' ' + enclose_with_quotes(top_dir)
    cmd += ' -e ' + enclose_with_quotes(pattern)

    
    if options is not None:
        cmd += ' ' + options
        #enclose_with_quotes(options)
    
    print cmd
    
    
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    out, err = proc.communicate()
    
    if err is not None:
        warnings.warn(err) 
        
    print out
    
    
def mnemonics():
    rules = [
    'Start interactive matplotlib plots: %matplotlib nbagg',
    'Use plotly: from plotly.offline import init_notebook_mode; init_notebook_mode(connected=True)',
    'Many figure plot/saving. Turn off interactive plotting.\nplt.ioff() and plt.close() in for-loop.',
    'Debugger set brkpoint: import ipdb; ipdb.set_trace()',
    'Reset jupyter\'s signature database: cd $(jupyter --data-dir); rm nbsignatures.db',
    'Plot multiple images inside single cell.\nfrom IPython.display import display; display(Fig1), display(Fig2)'        
    ]
    for i, r in enumerate(rules):
        print str(i) + '.\t', r + '\n'
