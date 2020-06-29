import _pickle as pickle


def save_object(obj, filepath):
    """This function takes an object and a filepath with filename and saves the object to that location"""
    if filepath[-3:] == '.pk1':
        filepath = filepath
    elif '.' in filepath[-4:]:
        filepath[-3:] = 'pk1'
    else:
        filepath = filepath + '.pk1'
    with open(filepath, 'wb') as output:  # Overwrites any existing file
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)