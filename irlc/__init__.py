# Do not import Matplotlib (or imports which import matplotlib) in case you have to run in headless mode.
import shutil
import inspect
import compress_pickle
import numpy as np
import os

# Global imports from across the API. Allows imports like
# > from irlc import Agent, train
from irlc.utils.irlc_plot import main_plot as main_plot
from irlc.utils.irlc_plot import plot_trajectory as plot_trajectory
from irlc.ex01.agent import Agent as Agent, train as train
try:
    from irlc.ex09.rl_agent import TabularAgent, ValueAgent
except ImportError:
    pass
from irlc.utils.video_monitor import VideoMonitor as VideoMonitor
from irlc.utils.player_wrapper import PlayWrapper as PlayWrapper
from irlc.utils.lazylog import LazyLog
from irlc.utils.timer import Timer


def get_irlc_base():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return dir_path


def get_students_base():
    return os.path.join(get_irlc_base(), "../../../02465students/")


def pd2latex_(pd, index=False, escape=False, **kwargs):
    for c in pd.columns:
        if pd[c].values.dtype == 'float64' and all(pd[c].values - np.round(pd[c].values)==0):
            # format column
            pd[c] = pd[c].astype(int)
    ss = pd.to_latex(index=index, escape=escape)
    return fix_bookstabs_latex_(ss)

def fix_bookstabs_latex_(ss, linewidth=True, first_column_left=True):
    if linewidth:
        ss.replace("tabular", "tabularx")
    lines = ss.split("\n")
    hd = lines[0].split("{")
    adj = ('L' if first_column_left else 'C') + ("".join(["C"] * (len(hd[-1][:-1])-1)))
    if linewidth:
        lines[0] = "\\begin{tabularx}{\\linewidth}{" + adj + "}"
    else:
        lines[0] = "\\begin{tabular}{" + adj.lower() + "}"

    ss = '\n'.join(lines)
    return ss


def savepdf(pdf, verbose=False, watermark=False):
    '''
    magic save command for generating figures. No need to read this code.
    '''
    import matplotlib.pyplot as plt
    pdf = pdf.strip()
    pdf = pdf+".pdf" if not pdf.endswith(".pdf") else pdf
    filename = None
    for k in range(5):
        frame = inspect.stack()[-1-k]
        module = inspect.getmodule(frame[0])
        filename = module.__file__
        if not any([filename.endswith(f) for f in ["pydevd.py", "_pydev_execfile.py"] ]):
            # print("breaking c. debugger", filename)
            break
    if any( [filename.endswith(f) for f in ["pydevd.py", "_pydev_execfile.py"]]):
        print("pdf path could not be resolved due to debug mode being active in pycharm", filename)
        return
    # print(filename)
    wd = os.path.dirname(filename)
    pdf_dir = wd +"/pdf"
    if filename.endswith("_RUN_OUTPUT_CAPTURE.py"):
        return
    if not os.path.isdir(pdf_dir):
        os.mkdir(pdf_dir)

    irlc_base = os.path.dirname(__file__)
    # if os.path.exists(os.getcwd()+ "/../../../Exercises") and os.path.exists(os.getcwd()+ "/../../../pdf_out"):
    if os.path.exists(irlc_base+ "/../../Exercises") and os.path.exists(irlc_base+ "/../../pdf_out") and "irlc" in os.path.abspath(pdf):
        lecs = [os.path.join(irlc_base, "../../shared/output")]
        od = lecs+[pdf_dir]
        for f in od:
            if not os.path.isdir(f):
                os.makedirs(f)

        on = od[0] + "/" + pdf
        # print("save to ", on)
        plt.savefig(fname=on)
        from thtools.slider import convert
        print("converting", on)
        convert.pdfcrop(on, fout=on)
        print("copying..")
        for f in od[1:]:
            shutil.copy(on, f +"/"+pdf)
    else:
        plt.savefig(fname=wd+"/"+pdf)
    outf = os.path.abspath(pdf)
    print("> [savepdf]", pdf + (f" [full path: {outf}]" if verbose else ""))

    if watermark:
        try:
            from thtools.slider import convert
            convert.pdfcrop(outf, fout=outf)
            if watermark:
                from thtools.plot.plot_helpers import watermark_plot
                watermark_plot()
                savepdf(pdf[:-4] + "_watermark.pdf", verbose=verbose, watermark=False)

        except ImportError as e:
            pass  # module doesn't exist, deal with it.
    return outf


def _move_to_output_directory(file):
    """
    Hidden function: Move file given file to static output dir.
    """
    if not is_this_my_computer():
        return
    CDIR = os.path.dirname(os.path.realpath(__file__)).replace('\\', '/')
    shared_output_dir = CDIR + "/../../shared/output"
    shutil.copy(file, shared_output_dir + "/"+ os.path.basename(file) )

def is_o_mode():
    return False

def bmatrix(a):

    if is_o_mode():
        return a.__str__()
    else:
        np.set_printoptions(suppress=True)
        """Returns a LaTeX bmatrix
        :a: numpy array
        :returns: LaTeX bmatrix as a string
        """
        if len(a.shape) > 2:
            raise ValueError('bmatrix can at most display two dimensions')
        lines = str(a).replace('[', '').replace(']', '').splitlines()
        rv = [r'\begin{bmatrix}']
        rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
        rv +=  [r'\end{bmatrix}']
        return '\n'.join(rv)


def is_this_my_computer():
    CDIR = os.path.dirname(os.path.realpath(__file__)).replace('\\', '/')
    return os.path.exists(CDIR + "/../../Exercises")


def cache_write(object, file_name, only_on_professors_computer=False, verbose=True, protocol=-1): # -1 is default protocol. Fix crash issue with large files.
    if only_on_professors_computer and not is_this_my_computer():
        """ Probably for your own good :-). """
        return

    dn = os.path.dirname(file_name)
    if not os.path.exists(dn):
        os.mkdir(dn)
    if verbose: print("Writing cache...", file_name)
    with open(file_name, 'wb') as f:
        compress_pickle.dump(object, f, compression="lzma", protocol=protocol)
    if verbose: print("Done!")


def cache_exists(file_name):
    return os.path.exists(file_name)

def cache_read(file_name):
    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            return compress_pickle.load(f, compression="lzma")
    else:
        return None
