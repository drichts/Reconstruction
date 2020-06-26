import os


def create_folder(folder_name, directory_path):
    """
    This function creates a new folder with the name given within the directory given if it hasn't already been created
    :param folder_name: The name of the folder to create or look for
    :param directory_path: The directory within to create the folder
    :return:
    """
    # Check to make sure the directory exists
    if os.path.exists(directory_path):

        path = directory_path + '/' + folder_name

        # Check to see if the folder already exists in the directory, if not, create it
        if os.path.exists(path):
            print("Folder already exists:", path)
        else:
            try:
                os.mkdir(path)
            except OSError:
                print("Creation of the directory %s failed" % path)
            else:
                print("Successfully created the directory %s " % path)

    # If the directory does not exist raise an error
    else:
        try:
            raise OSError('The directory does not exist.')
        except OSError:
            print('Possible problem...')
            raise
