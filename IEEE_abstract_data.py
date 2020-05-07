import numpy as np
import general_OS_functions as gof
import matplotlib.pyplot as plt
import seaborn as sns
import glob

#%% Flat field air

def flatfield_air():
    folder = 'C:/Users/10376/Documents/IEEE Abstract/Analysis Data/Flat Field/'

    print('1w           4w')
    w1 = np.load(folder + '/flatfield_1wA0.npy')
    w4 = np.load(folder + '/flatfield_4wA0.npy')

    w1 = np.squeeze(w1)
    w4 = np.squeeze(w4)

    w1 = np.sum(w1, axis=1)
    w4 = np.sum(w4, axis=1)

    for i in np.arange(5):
        print('Bin' + str(i))
        #print(np.nanstd(w1[i+6])/np.mean(w1[i+6]))
        #print(np.nanstd(w4[i+6])/np.mean(w4[i+6]))
        print(np.mean(w1[i+6]))
        print(np.mean(w4[i+6]))

    return w1
#x = flatfield_air()


#%% Flatfield phantoms (correct for air)

def flatfield_phantoms():
    directory = 'C:/Users/10376/Documents/IEEE Abstract/'
    load_folder = 'Raw Data/Flat Field\\'
    save_folder = 'Analysis Data/Flat Field/'

    load_path = directory + load_folder
    air1w = np.squeeze(np.load(load_path + '/flatfield_1w.npy'))
    air4w = np.squeeze(np.load(load_path + '/flatfield_4w.npy'))

    blue1w = np.squeeze(np.load(load_path + 'bluebelt_1w.npy'))
    blue4w = np.squeeze(np.load(load_path + 'bluebelt_4w.npy'))

    plexi1w = np.squeeze(np.load(load_path + 'plexiglass_1w.npy'))
    plexi4w = np.squeeze(np.load(load_path + 'plexiglass_4w.npy'))

    blue1w = np.sum(blue1w, axis=1)
    blue4w = np.sum(blue4w, axis=1)
    plexi1w = np.sum(plexi1w, axis=1)
    plexi4w = np.sum(plexi4w, axis=1)
    air1w = np.sum(air1w, axis=1)
    air4w = np.sum(air4w, axis=1)

    blue1w = np.divide(blue1w, air1w)
    blue4w = np.divide(blue4w, air4w)

    plexi1w = np.divide(plexi1w, air1w)
    plexi4w = np.divide(plexi4w, air4w)

    blue1w = -1*np.log(blue1w)
    blue4w = -1*np.log(blue4w)

    plexi1w = -1*np.log(plexi1w)
    plexi4w = -1*np.log(plexi4w)

    #np.save(directory + save_folder + 'bluebelt_1w.npy', blue1w)
    #np.save(directory + save_folder + 'bluebelt_4w.npy', blue4w)

    #np.save(directory + save_folder + 'plexiglass_1w.npy', plexi1w)
    #np.save(directory + save_folder + 'plexiglass_4w.npy', plexi4w)


    return blue1w, blue4w, plexi1w, plexi4w

b1, b4, p1, p4 = flatfield_phantoms()
