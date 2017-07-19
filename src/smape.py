import numpy as np

def Num_Sampe(a, b):
    if a or b:
        return 100.0*abs(a-b)/(abs(a)+abs(b))
    else:
        return 0.0

def sampe(y_true, y_pred):
    vsampe = np.vectorize(Num_Sampe)
    return vsampe(y_true,y_pred).mean()
