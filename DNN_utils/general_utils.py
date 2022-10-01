import pickle
def save_model(model,modelname,path="models/"):
    with open(path + modelname + ".pkl", 'wb') as outp:  
        pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)
def load_model(modelname,path="models/"):
    with open(path + modelname + ".pkl", 'rb') as inp:
        return pickle.load(inp)