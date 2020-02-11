import numpy as np
#import matplotlib.pyplot as plt
import csv
import os
import sys
#import astra
import scipy.io as sio
import glob
import pywt
from numpy.fft import fftshift, ifftshift, fft, ifft

def main_multiplex(energy_name='EC',directory_air ='.',dir_home='.',
                    translations=np.array(["10", "0.5", "-9", "-18.5", "0.5"])):
    '''
    This is for the multiplexed 3D-printed phantom (Au, Gd, I)
    :return:
    '''

    np.seterr(divide='ignore', invalid='ignore')

    directory_air = "C:/Users/drich/PycharmProjects/Reconstruction/sCT Scan Data/Au_4_29_19/Air/"
    dir_home = "C:/Users/drich/PycharmProjects/Reconstruction/sCT Scan Data/Au_4_29_19/"

    pixel_corr = np.load('deadpixel_mask.npy')

    number_projections = 181
    #energy_name = "SEC5"  #"SEC1", "SEC2", "EC", etc. This doesn't loop so run this code several times for each energy bin

    # CZT translates to these position, similar to a panoramic photo
    translations = np.array(["10", "0.5", "-9", "-18.5", "0.5"])

    files1 = os.listdir(dir_home + "Rot_" + translations[0] + "/")
    files2 = os.listdir(dir_home + "Rot_" + translations[1] + "/")
    files3 = os.listdir(dir_home + "Rot_" + translations[2] + "/")
    files4 = os.listdir(dir_home + "Rot_" + translations[3] + "/")
    #files5 = os.listdir(dir_home + "Rot_" + translations[4] + "/")

    files1.sort()
    files2.sort()
    files3.sort()
    files4.sort()
    #files5.sort()

    dark_image = generateImageByReadingCSVdata(directory_air, "dark.csv", energyname=energy_name)
    air_image1 = generateImageByReadingCSVdata(directory_air, "air_" + translations[0] + ".csv", energyname=energy_name)
    air_image1 = np.subtract(air_image1, dark_image)
    #air_image1 = np.multiply(air_image1, pixel_corr)

    air_image2 = generateImageByReadingCSVdata(directory_air, "air_" + translations[1] + ".csv", energyname=energy_name)
    air_image2 = np.subtract(air_image2, dark_image)
    #air_image2 = np.multiply(air_image2, pixel_corr)

    air_image3 = generateImageByReadingCSVdata(directory_air, "air_" + translations[2] + ".csv", energyname=energy_name)
    air_image3 = np.subtract(air_image3, dark_image)
    #air_image3 = np.multiply(air_image3, pixel_corr)

    air_image4 = generateImageByReadingCSVdata(directory_air, "air_" + translations[3] + ".csv", energyname=energy_name)
    air_image4 = np.subtract(air_image4, dark_image)
    #air_image4 = np.multiply(air_image4, pixel_corr)

    #import pdb;pdb.set_trace()
    #air_image5 = generateImageByReadingCSVdata(directory_air, "air_" + translations[4] + ".csv", energyname=energy_name)

    #only plot the minimum number of projections. Assuming they start at same initial position
    sizes = np.array([np.size(files1), np.size(files2), np.size(files3), np.size(files4)])
    min_proj = np.min(sizes)
    # import pdb;pdb.set_trace()

    for i in np.arange(0, min_proj):
        proj_image1 = generateImageByReadingCSVdata(dir_home + "Rot_" + translations[0] + "/", files1[i],
                                                    energyname=energy_name)
        proj_image1 = np.subtract(proj_image1, dark_image)
        #proj_image1 = np.multiply(proj_image1, pixel_corr)

        proj_image2 = generateImageByReadingCSVdata(dir_home + "Rot_" + translations[1] + "/", files2[i],
                                                    energyname=energy_name)
        proj_image2 = np.subtract(proj_image2, dark_image)
        #proj_image2 = np.multiply(proj_image2, pixel_corr)

        proj_image3 = generateImageByReadingCSVdata(dir_home + "Rot_" + translations[2] + "/", files3[i],
                                                    energyname=energy_name)
        proj_image3 = np.subtract(proj_image3, dark_image)
        #proj_image3 = np.multiply(proj_image3, pixel_corr)

        proj_image4 = generateImageByReadingCSVdata(dir_home + "Rot_" + translations[3] + "/", files4[i],
                                                    energyname=energy_name)
        proj_image4 = np.subtract(proj_image4, dark_image)
        #proj_image4 = np.multiply(proj_image4, pixel_corr)

        #proj_image5 = generateImageByReadingCSVdata(
            #dir_home + "Rot_" + translations[4] + "/",
            #files5[i], energyname=energy_name)

        ct_image1 = -np.log(proj_image1 / air_image1)
        ct_image2 = -np.log(proj_image2 / air_image2)
        ct_image3 = -np.log(proj_image3 / air_image3)
        ct_image4 = -np.log(proj_image4 / air_image4)
        #ct_image5 = -np.log(proj_image5 / air_image5)

        #manually setting this underresponding pixel to be the average of the nearest neighbours.
        #dead_pixel_x = 9
        #dead_pixel_y = 32
        #ct_image1[dead_pixel_x,dead_pixel_y] = getAveragePixelValue(ct_image1, [dead_pixel_x,dead_pixel_y])
        #ct_image2[dead_pixel_x,dead_pixel_y] = getAveragePixelValue(ct_image2, [dead_pixel_x,dead_pixel_y])
        #ct_image3[dead_pixel_x,dead_pixel_y] = getAveragePixelValue(ct_image3, [dead_pixel_x,dead_pixel_y])
        #ct_image4[dead_pixel_x,dead_pixel_y] = getAveragePixelValue(ct_image4, [dead_pixel_x,dead_pixel_y])
        #ct_image5[dead_pixel_x,dead_pixel_y] = getAveragePixelValue(ct_image5, [dead_pixel_x,dead_pixel_y])
        # plt.imshow(ct_image2, interpolation='none')
        # plt.title("projection "+str(i)+", offset="+str(translations[2]))
        # plt.show()
        # plt.imshow(proj_image2,interpolation='none')
        # plt.show()

        adict = {}
        adict['ct_image1'] = ct_image1
        adict['ct_image2'] = ct_image2
        adict['ct_image3'] = ct_image3
        adict['ct_image4'] = ct_image4
        #adict['ct_image5'] = ct_image5

        sio.savemat(dir_home + "projection_multiplex_" + str(energy_name)+"_rot"+str(i)+".mat", adict)

        # np.save("/home/chelsea/Desktop/UVICnotes/data_AuCT/stitchedrot_projections_5px/projectionAuCT_rot"+str(i)+".npy",
        #         ct_image)
        # ii = "%03d"%i
        # plt.imshow(ct_image, extent=(-34.44,7.13,-3.96,3.96))
        # plt.xlabel("x [mm]")
        # plt.ylabel("y [mm]")
        # plt.title("New rotation "+str(ii))
        # plt.savefig("/home/chelsea/Desktop/UVICnotes/data_AuCT/stitchedrot_projections_5px/projectionAuCT_"+
        #             str(energy_name)+"_rot"+str(ii)+".png",
        #             dpi=100)
        # #plt.show()
        # plt.close()

    #@todo: IMAGE RECONSTRUCTION USING MATLAB

    #np.save("projections_sample.npy",PROJECTIONS)


def getAveragePixelValue(img, pixels):
    '''
    Averages the dead pixel using the 8 nearest neighbours
    :param img: the projection image
    :param pixels: the problem pixels (is a 2-tuple)
    :return:
    '''
    x, y = pixels
    n1 = img[x,y+1]
    n2 = img[x,y-1]
    n3 = img[x+1,y]
    n4 = img[x-1,y]
    n5 = img[x+1,y+1]
    n6 = img[x+1,y-1]
    n7 = img[x-1,y+1]
    n8 = img[x-1,y-1]
    avg = np.average(np.array([n1,n2,n3,n4,n5,n6,n7,n8]))

    return avg


def stitchProjectionImages(proj1, proj2, proj3, proj4):
    '''
    I used to stitch in Python, now I do it in matlab. ignore this
    :param proj1:
    :param proj2:
    :param proj3:
    :param proj4:
    :return:
    '''
    new_image = np.zeros([24,(36*4) - (3*6)])
    new_image[:,0:31] = proj1[:,0:31]
    new_image[:, 31:62] = proj2[:,0:31]
    new_image[:, 62:93] = proj3[:,0:31]
    new_image[:, 93::] = proj4[:,0:33]
    #plt.imshow(new_image)
    #plt.show()

    return new_image


def generateImageByReadingCSVdata(directory, filename, energyname = "EC"):
    '''
    Reads in the file to produce a projection image from the csv data
    :param directory:
    :param filename:
    :param energyname: one of 'EC', 'SEC0-5'
    :return:
    '''
    energy_dict = {'SUMCC': 18,
                   'CC0' :  11,
                   'CC1' : 10,
                   'CC2' : 9,
                   'CC3' : 8,
                   'CC4' : 7,
                   'CC5' : 6,
                   'EC' : 5,
                   'SEC5' : 12,
                   'SEC4' : 13,
                   'SEC3' : 14,
                   'SEC2' : 15,
                   'SEC1' : 16,
                   'SEC0' : 17
                   }
    rows = 24
    columns = 72
    old_pixel_module = -1
    projection_image = np.zeros([rows, columns])
    with open(directory+filename, 'rt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for csv_row in spamreader:
            current_pixel_module = csv_row[0]
            if current_pixel_module != "Pixel":

                row = int(csv_row[2])
                column = int(csv_row[3])
                if energyname != 'Kedge':
                    entry = np.float(csv_row[energy_dict[energyname]])
                else:
                    #This should be unreachable. This was when I did the K-edge subtraction earlier
                    cc4entry = np.float(csv_row[energy_dict['CC4']])
                    cc3entry = np.float(csv_row[energy_dict['CC3']])
                    entry = cc4entry - cc3entry
                if old_pixel_module != current_pixel_module:
                    projection_image[row, column] += entry
                else:
                    # Jericho quick fix for making the input take both detectors
                    # comment out if you want just one detector
                    projection_image[row, 71 - column] += entry
                old_pixel_module = current_pixel_module
    return projection_image

def interpolateCSVdata(directory, filename, neighbour_low, neighbour_high):
    '''
    if missing data file in scan
    :param directory:
    :param filename:
    :return:
    '''
    with open(directory+neighbour_low, 'r') as csvlow, open(directory+neighbour_high, 'r') as csvhigh:
        filelow = csvlow.readlines()
        filehigh = csvhigh.readlines()

    with open(directory+filename, "w") as outfile:
        header = filelow[0]
        outfile.write(header)
        for i_row in np.arange(1, np.size(filelow)):
            rowlow = filelow[i_row].split(",")
            rowhigh = filehigh[i_row].split(",")
            row_avg = np.copy(rowhigh)
            row_avg_str = ""
            # average the rows
            for value in np.arange(0, 19):
                if value > 4:
                    row_avg_value = str(((int(rowlow[value]) + int(rowhigh[value]))/2))
                    if value == 18:
                        row_avg_value += "\r\n"
                    else:
                        row_avg_value += ","
                    row_avg_str += row_avg_value
                else:
                    row_avg_str += row_avg[value]+","

            # now write the average row to csv
            outfile.write(row_avg_str)

def find_missing(dir_to_search = 'Rot*/',interpolate=False):
    '''
    This is a function to intepolate over the missing entries
    dir_to_search: str,optional
    Term to search for, should be the prefix of the directories
    interpolate: bool, optional
    Whether or not to interpolate, False just shows what files will be replaced
    '''
    count = 1
    count2 = 1
    temp2 = 0

    Dirs = glob.glob("{}".format(dir_to_search))

    for directory in Dirs:

        files = sorted(glob.glob("{}{}".format(directory, '*.csv')))
        # print files
        
        for file in files:
            
            # print file
            temp1 = np.mod(round(int(file[-6:-4]), -1), 60)
            if temp1 != np.mod(temp2+10, 60) and count != 1:
                count2 = count2 + 1
                
                filename = '{}{}.csv'.format(file[:-6], "%02d" % int(np.mod(temp2+10, 60)))
                if interpolate:
                    interpolateCSVdata('', filename, files[count-2], files[count-1])
                    print(filename, 'created')
                else:
                    print(filename, 'will be created')

            count = count + 1
            temp2 = temp1

        count = 1
        count2 = 1
        
    if not interpolate:
        print('To replace these files run the next block of code, if not skip it')
        
def remove_stripe(img, level, wname='db5', sigma=1.5):
    """
    Suppress horizontal stripe in a sinogram using the Fourier-Wavelet based
    method by Munch et al. [2]_.

    Parameters
    ----------
    img : 2d array
        The two-dimensional array representig the image or the sinogram to de-stripe.

    level : int
        The highest decomposition level.

    wname : str, optional
        The wavelet type. Default value is ``db5``

    sigma : float, optional
        The damping factor in the Fourier space. Default value is ``1.5``

    Returns
    -------
    out : 2d array
        The resulting filtered image.

    References
    ----------
    .. [2] B. Munch, P. Trtik, F. Marone, M. Stampanoni, Stripe and ring artifact removal with
           combined wavelet-Fourier filtering, Optics Express 17(10):8567-8591, 2009.
    """

    nrow, ncol = img.shape

    # wavelet decomposition.
    cH = []; cV = []; cD = []

    for i in range(0, level):
        img, (cHi, cVi, cDi) = pywt.dwt2(img, wname)
        cH.append(cHi)
        cV.append(cVi)
        cD.append(cDi)

    # FFT transform of horizontal frequency bands
    for i in range(0, level):
        # FFT
        fcV=fftshift(fft(cV[i], axis=0))
        my, mx = fcV.shape

        # damping of vertical stripe information
        yy2  = (np.arange(-np.floor(my/2), -np.floor(my/2) + my))**2
        damp = - np.expm1( - yy2 / (2.0*(sigma**2)) )
        fcV  = fcV * np.tile(damp.reshape(damp.size, 1), (1,mx))

        #inverse FFT
        cV[i] = np.real( ifft( ifftshift(fcV), axis=0) )

    # wavelet reconstruction
    for i in  range(level-1, -1, -1):
        img = img[0:cH[i].shape[0], 0:cH[i].shape[1]]
        img = pywt.idwt2((img, (cH[i], cV[i], cD[i])), wname)

    return img[0:nrow, 0:ncol]

def main_dark_cor(energy_name='EC',directory_air ='.',dir_home='.',
                    translations=np.array(["10", "0.5", "-9", "-18.5", "0.5"]),
                    dark_file='./Dark&Air/Dark/Dark_1.csv'):
    '''
    This is for the multiplexed 3D-printed phantom (Au, Gd, I)
    :return:
    '''

    number_projections = 181
    # energy_name = "EC"  #"SEC1", "SEC2", "EC", etc. This doesn't loop so run this code several times for each energy bin

    # CZT translates to these position, similar to a panoramic photo
    # translations = np.array(["10", "0.5", "-9", "-18.5", "0.5"])

    files1 = os.listdir(dir_home + "Rot_" + translations[0] + "/")
    files2 = os.listdir(dir_home + "Rot_" + translations[1] + "/")
    files3 = os.listdir(dir_home + "Rot_" + translations[2] + "/")
    files4 = os.listdir(dir_home + "Rot_" + translations[3] + "/")
    # files5 = os.listdir(dir_home + "Rot_" + translations[4] + "/")

    files1.sort()
    files2.sort()
    files3.sort()
    files4.sort()
    # files5.sort()

    air_image1 = generateImageByReadingCSVdata(directory_air, "air_" + translations[0] + ".csv", energyname=energy_name)
    air_image2 = generateImageByReadingCSVdata(directory_air, "air_" + translations[1] + ".csv", energyname=energy_name)
    air_image3 = generateImageByReadingCSVdata(directory_air, "air_" + translations[2] + ".csv", energyname=energy_name)
    air_image4 = generateImageByReadingCSVdata(directory_air, "air_" + translations[3] + ".csv", energyname=energy_name)
    dark_image = generateImageByReadingCSVdata('',dark_file, energyname=energy_name)

    # only plot the minimum number of projections. Assuming they start at same initial position
    sizes = np.array([np.size(files1), np.size(files2), np.size(files3), np.size(files4)])
    min_proj = np.min(sizes)

    for i in np.arange(0,min_proj):
        proj_image1 = generateImageByReadingCSVdata(
            dir_home + "Rot_" + translations[0] + "/",
            files1[i], energyname=energy_name)
        proj_image2 = generateImageByReadingCSVdata(
            dir_home + "Rot_" + translations[1] + "/",
            files2[i], energyname=energy_name)
        proj_image3 = generateImageByReadingCSVdata(
            dir_home + "Rot_" + translations[2]  + "/",
            files3[i], energyname=energy_name)
        proj_image4 = generateImageByReadingCSVdata(
            dir_home + "Rot_" + translations[3] +  "/",
            files4[i], energyname=energy_name)
        # proj_image5 = generateImageByReadingCSVdata(
            # dir_home + "Rot_" + translations[4] + "/",
            # files5[i], energyname=energy_name)

        ct_image1 = -np.log((proj_image1-dark_image) / (air_image1-dark_image))
        ct_image2 = -np.log((proj_image2-dark_image) / (air_image2-dark_image))
        ct_image3 = -np.log((proj_image3-dark_image) / (air_image3-dark_image))
        ct_image4 = -np.log((proj_image4-dark_image) / (air_image4-dark_image))
        # ct_image5 = -np.log(proj_image5 / air_image5)

        # manually setting this underresponding pixel to be the average of the nearest neighbours.
        dead_pixel_x = 9
        dead_pixel_y = 32
        ct_image1[dead_pixel_x, dead_pixel_y] = getAveragePixelValue(ct_image1, [dead_pixel_x, dead_pixel_y])
        ct_image2[dead_pixel_x, dead_pixel_y] = getAveragePixelValue(ct_image2, [dead_pixel_x, dead_pixel_y])
        ct_image3[dead_pixel_x, dead_pixel_y] = getAveragePixelValue(ct_image3, [dead_pixel_x, dead_pixel_y])
        ct_image4[dead_pixel_x, dead_pixel_y] = getAveragePixelValue(ct_image4, [dead_pixel_x, dead_pixel_y])

        adict = {}
        adict['ct_image1'] = ct_image1
        adict['ct_image2'] = ct_image2
        adict['ct_image3'] = ct_image3
        adict['ct_image4'] = ct_image4
        # adict['ct_image5'] = ct_image5

        sio.savemat(dir_home + "projection_multiplex_"+
                    str(energy_name)+"_rot"+str(i)+".mat", adict)


main_multiplex(energy_name='EC')
