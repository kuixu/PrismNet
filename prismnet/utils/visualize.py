import os, sys
import numpy as np
import matplotlib as mpl
mpl.use("pdf")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.misc import imresize

package_directory = os.path.dirname(os.path.abspath(__file__))
acgu_path = os.path.join(package_directory,'acgu.npz')
chars = np.load(acgu_path,allow_pickle=True)['data']

def normalize_pwm(pwm, factor=None, MAX=None):
    if MAX is None:
        MAX = np.max(np.abs(pwm))
    pwm = pwm/MAX
    if factor:
        pwm = np.exp(pwm*factor)
    norm = np.outer(np.ones(pwm.shape[0]), np.sum(np.abs(pwm), axis=0))
    return pwm/norm

def get_nt_height(pwm, height, norm):

    def entropy(p):
        s = 0
        for i in range(len(p)):
            if p[i] > 0:
                s -= p[i]*np.log2(p[i])
        return s

    num_nt, num_seq = pwm.shape
    heights = np.zeros((num_nt,num_seq))
    for i in range(num_seq):
        if norm == 1:
            total_height = height
        else:
            total_height = (np.log2(num_nt) - entropy(pwm[:, i]))*height
        
        heights[:,i] = np.floor(pwm[:,i]*np.minimum(total_height, height*2))

    return heights.astype(int)

def seq_logo(pwm, height=30, nt_width=10, norm=0, alphabet='rna', colormap='standard'):

    heights = get_nt_height(pwm, height, norm)
    num_nt, num_seq = pwm.shape
    width = np.ceil(nt_width*num_seq).astype(int)
    
    max_height = height*2
    logo = np.ones((max_height, width, 3)).astype(int)*255
    for i in range(num_seq):
        nt_height = np.sort(heights[:,i])
        index = np.argsort(heights[:,i])
        remaining_height = np.sum(heights[:,i])
        offset = max_height-remaining_height

        for j in range(num_nt):
            if nt_height[j] <=0 :
                continue
            # resized dimensions of image
            nt_img = imresize(chars[index[j]], (nt_height[j], nt_width))
            # determine location of image
            height_range = range(remaining_height-nt_height[j], remaining_height)
            width_range = range(i*nt_width, i*nt_width+nt_width)
            # 'annoying' way to broadcast resized nucleotide image
            if height_range:
                for k in range(3):
                    for m in range(len(width_range)):
                        logo[height_range+offset, width_range[m],k] = nt_img[:,m,k]

            remaining_height -= nt_height[j]

    return logo.astype(np.uint8)

def plot_saliency(X, W, nt_width=100, norm_factor=3, str_null=None, outdir="results/"):
    # filter out zero-padding
    plot_index = np.where(np.sum(X[:4,:], axis=0)!=0)[0]
    num_nt = len(plot_index)
    trace_width = num_nt*nt_width
    trace_height = 400
    
    seq_str_mode = False
    if X.shape[0]>4:
        seq_str_mode = True
        assert str_null is not None, "Null region is not provided."

    # sequence logo
    img_seq_raw = seq_logo(X[:4, plot_index], height=nt_width, nt_width=nt_width)

    if seq_str_mode:
        # structure line
        str_raw = X[4, plot_index]
        if str_null.sum() > 0:
            str_raw[str_null.T==1] = -0.01

        line_str_raw = np.zeros(trace_width)
        for v in range(str_raw.shape[0]):
            line_str_raw[v*nt_width:(v+1)*nt_width] = (1-str_raw[v])*trace_height 
            # i+=1
    
    # sequence saliency logo
    seq_sal = normalize_pwm(W[:4, plot_index], factor=norm_factor)
    img_seq_sal_logo = seq_logo(seq_sal, height=nt_width*5, nt_width=nt_width)
    img_seq_sal = imresize(W[:4, plot_index], size=(trace_height, trace_width))

    if seq_str_mode:
        # structure saliency logo
        str_sal = W[4, plot_index].reshape(1,-1)
        img_str_sal = imresize(str_sal, size=(trace_height, trace_width))


    # plot    
    fig = plt.figure(figsize=(10,2))
    gs = gridspec.GridSpec(nrows=4, ncols=1, height_ratios=[2, 1, 0.5, 1])
    cmap_reversed = mpl.cm.get_cmap('jet')

    ax = fig.add_subplot(gs[0, 0])
    ax.axis('off')
    ax.imshow(img_seq_sal_logo)

    ax = fig.add_subplot(gs[1, 0]) 
    ax.axis('off')
    ax.imshow(img_seq_sal, cmap=cmap_reversed)

    ax = fig.add_subplot(gs[2, 0]) 
    ax.axis('off')
    ax.imshow(img_seq_raw)

    if seq_str_mode:
        ax = fig.add_subplot(gs[3, 0]) 
        ax.axis('off')
        ax.imshow(img_str_sal, cmap=cmap_reversed)
        ax.plot(line_str_raw, '-', color='r', linewidth=1)
        
        # plot balck line to hide the -1(NULL structure score)
        x = (np.zeros(trace_width) + (1+0.01))*trace_height  +1.5
        ax.plot(x, '-', color='white', linewidth=1.2)

    plt.subplots_adjust(wspace=0, hspace=0)
    
    # save figure
    filepath = outdir
    fig.savefig(filepath, format='pdf', dpi=300, bbox_inches='tight')
    plt.close('all')
