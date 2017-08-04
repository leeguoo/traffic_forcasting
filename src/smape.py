import numpy as np

def Num_Smape(a, b):
    if a or b:
        return 200.0*abs(a-b)/(abs(a)+abs(b))
    else:
        return 0.0

def smape(y_true, y_pred):
    vsmape = np.vectorize(Num_Smape)
    return vsmape(y_true,y_pred).mean()

def XGBsmape(preds,dtrain):
    y = dtrain.get_label()
    return 'smape', smape(y,preds)
