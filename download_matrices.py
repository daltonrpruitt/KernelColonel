#!/apps/python-3.9.2/bin/python3

import os, shutil, sys

import requests
import tarfile

# https://stackoverflow.com/questions/8428954/move-child-folder-contents-to-parent-folder-in-python
def move_to_root_folder(root_path, cur_path):
    for filename in os.listdir(cur_path):
        if os.path.isfile(os.path.join(cur_path, filename)):
            shutil.move(os.path.join(cur_path, filename), os.path.join(root_path, filename))
        elif os.path.isdir(os.path.join(cur_path, filename)):
            move_to_root_folder(root_path, os.path.join(cur_path, filename))
        else:
            sys.exit("Should never reach here.")
    # remove empty folders
    if cur_path != root_path:
        os.rmdir(cur_path)

suitesparse_base_url = "https://suitesparse-collection-website.herokuapp.com/MM/"
matrix_urls = {
    "bcsstk13"      : suitesparse_base_url + "HB/bcsstk13.tar.gz",
    "bcsstm13"      : suitesparse_base_url + "HB/bcsstm13.tar.gz",
    "StocF-1465"    : suitesparse_base_url + "Janna/StocF-1465.tar.gz",
    "Goodwin_127"   : suitesparse_base_url + "Goodwin/Goodwin_127.tar.gz",
    "RM07R"         : suitesparse_base_url + "Fluorem/RM07R.tar.gz",
    "rma10"         : suitesparse_base_url + "Bova/rma10.tar.gz",
    "mixtank_new"   : suitesparse_base_url + "POLYFLOW/mixtank_new.tar.gz",
    "cfd2"          : suitesparse_base_url + "Rothberg/cfd2.tar.gz",
    "cfd1"          : suitesparse_base_url + "Rothberg/cfd1.tar.gz",
}

def find_matrix_directory():
    for dirpath, dirnames, filenames in os.walk(os.getcwd()):
        for dirname in dirnames:
            if dirname == "matrices":
                return os.path.join(dirpath, dirname)                
    return None

def find_file_in_dir(start_dir, filename):
    for dirpath, dirnames, filenames in os.walk(os.getcwd()):
        for f in filenames:
            if f == filename:
                return True                
    return False
    

def download_extract_matrix(matrix_name, urls_dict, extract_dir):
    
    if find_file_in_dir(extract_dir, matrix_name+".mtx"):
        print("Matrix", matrix_name, "already exists!")
        return True
    print(matrix_name, ": begin processing")
    response = requests.get(urls_dict[matrix_name], stream=True)
    if response.status_code == 200:
        print("\tGot response", end=" - ")
        try: 
            file = tarfile.open(fileobj=response.raw, mode="r|gz")
            print("Opened tarfile", end=" - ")
            file.extractall(path=extract_dir)
            print("tarfile extracted successfully",  end=" - ")
            return True
        except Exception as e:
            print("\nFailed! :",e)
            return False
    else: 
        print("Bad response code:",response.status_code)
        return False

def main():
    matrices_dir = find_matrix_directory()
    if matrices_dir is None: 
        print("Could not find 'matrices' directory!")
        exit(-1)
    # print("Matrices dir =", matrices_dir)
    
    
    for m in matrix_urls.keys(): 
        if not download_extract_matrix(m, matrix_urls, matrices_dir):
            print("Downloading failed!")
   
    print("All downloads processed successfully!")
    print("Moving *.mtx files to matrices dir directly...")
    move_to_root_folder(matrices_dir, matrices_dir)


if __name__=="__main__":
    main()
