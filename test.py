import pickle

def verify_pickle(file_path):
    with open(file_path, 'rb') as file_obj:
        obj = pickle.load(file_obj)
        print(f"Object type from {file_path}: {type(obj)}")
        return obj

verify_pickle('artifacts/preprocessor.pkl')
verify_pickle('artifacts/model.pkl')
