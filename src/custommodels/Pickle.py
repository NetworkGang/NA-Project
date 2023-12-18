import os
import pickle as pkl

def log(text, silent = False):
    """print"""
    if not silent:
        print(text)

def pickle(function, silent = False):
    """Save/load data from a function. Name of the function is used as the name of the file."""
    # Get name of function
    funcname = function.__name__
    location = '../data/picklejar/' + funcname + '.pkl'
    if os.path.isfile(location):
        if not silent:
            print("Found data, loading from picklejar")
        # Load data
        with open(location, 'rb') as f:
            data = pkl.load(f)
    else:
        if not silent:
            print("No data found, creating new data")
        # Save data
        data = function()
        with open(location, 'wb') as f:
            pkl.dump(data, f)
    return data

def load(name, silent = False, relative_path = True):
    """Load data from picklejar"""
    location = ('../data/picklejar/' if relative_path else 'data/picklejar/') + name + '.pkl'
    if os.path.isfile(location):
        if not silent:
            print("Found data, loading from picklejar")
        # Load data
        with open(location, 'rb') as f:
            data = pkl.load(f)
    else:
        raise Exception("No data found in \"" + location + "\"")
    return data

def save(function):
    """Save data to picklejar"""
    # Get name of function
    funcname = function.__name__
    location = '../data/picklejar/' + funcname + '.pkl'
    # Save data
    data = function()
    with open(location, 'wb') as f:
        pkl.dump(data, f)
    return data

def delete(name):
    """Delete data from picklejar"""
    location = '../data/picklejar/' + name + '.pkl'
    if os.path.isfile(location):
        print("Found data, deleting from picklejar")
        os.remove(location)
    else:
        raise Exception("No data to delete found in \"" + location + "\"")
    return None