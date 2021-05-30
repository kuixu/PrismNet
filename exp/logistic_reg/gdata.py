import numpy as np

import pandas as pd

def g_rand_data(N=5000, M=128):
    label = np.random.randint(2,size=N)
    data  = np.random.rand(N,M)
    line = ""
    for i in range(N):
        datastr = " ".join(["{:d}:{:.3f}".format(j, data[i,j]) for j in range(M) ])
        line += "{:d} {:s}\n".format(label[i], datastr)
    print(line)

def save_file(data, filepath):
    print("Y min,max:", data[:,0].min(), data[:,0].max())
    print("X min,max:", data[:,1:].min(), data[:,1:].max())
    print("p/n:", data[:,0].sum()/data.shape[0],)
    N,M = data.shape
    with open(filepath,"w") as f:
        line = ""
        for i in range(N):
            datastr = " ".join(["{:d}:{:f}".format(j, data[i,j]) for j in range(1, M) ])
            line += "{:d} {:s}\n".format(int(data[i,0]), datastr)
        print(line, file=f)
def concat_data(filepath):
    raw_data = pd.read_csv(filepath,sep="\t")
    data=raw_data.to_numpy()[:,1:]

    # import pdb; pdb.set_trace()
    

    t_path = filepath.replace(".txt",".train")
    e_path = filepath.replace(".txt",".test")
    
    # import pdb; pdb.set_trace()
    # data = abs(np.concatenate((pos_samples, neg_samples)))
    # data = np.concatenate((pos_samples, neg_samples))
    print("min,max:", data[:,1:].min(), data[:,1:].max())
    dmin = data[:,0].min()
    dwid = data[:,0].max() - dmin
    # data[:,0] = (data[:,0] - dmin)/dwid
    # norm 
    data[:,1:] = (data[:,1:] - data[:,1:].mean())/data[:,1:].std()
    N,M = data.shape
    perm =np.random.permutation(N)
    t_N = int(0.8*N)


    # save_file(data[perm][:t_N,:], t_path)
    np.savez_compressed(t_path+".npz", x=data[perm][:t_N,1:], y=data[perm][:t_N,0],dmin=dmin,dwid=dwid)
    print("Training file saved into:", t_path,",", t_N," samples.")
    # save_file(data[perm][t_N:,:], e_path)
    np.savez_compressed(e_path+".npz", x=data[perm][t_N:,1:], y=data[perm][t_N:,0],dmin=dmin,dwid=dwid)
    print("Testing file saved into:", e_path,",", N-t_N," samples.")
    
    


if __name__ == "__main__":
    # g_rand_data()
    import glob
    for f in glob.glob("data/regu6/*.txt"):
        print(f)
        try:
            concat_data(f)
        except TypeError:
            pass
        
