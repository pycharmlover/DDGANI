import os
import glob
import shutil
import platform

from dependencies_util import Attribute
import psutil
from datetime import datetime


def get_max_ram():
    return int(psutil.virtual_memory().total / (1024 ** 3))


def copy_file(source, destination):
    shutil.copyfile(source, destination)

def copy_file_to_folder(source, folder):
    filename = os.path.basename(source)
    copy_file(source, join_path(folder, filename))

def clean_directory(dir_path):
    files = glob.glob(os.path.join(dir_path, '*'))
    for f in files:
        os.remove(f)

def get_file_first_file(dir_path):
    files=glob.glob(os.path.join(dir_path, '*'))
    if len(files)>0:
        return files[0]
    return None


def get_most_recent_file(dir_path: str) -> str:
    files = glob.glob(os.path.join(dir_path, '*'))


    if not files:
        raise FileNotFoundError(f"Nessun file trovato in {dir_path}")

    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def drop_directory(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)

def delete_file(file_path):
    os.remove(file_path)

def delete_files(files):
    for f in files:
        delete_file(f)

def get_all_files(dir_path, filter="*"):
    return glob.glob(os.path.join(dir_path, filter))

def check_and_create_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def join_path(folder, file):
    return os.path.join(folder, file)

def cast_attribute(attribute_string):
    splitted = attribute_string.split('@')
    return Attribute(splitted[0].strip(), float(splitted[1].strip()))

def is_dir(dir_path):
    return os.path.isdir(dir_path)


def profile_f(f, *positional, **parameters):
    start_time = datetime.now()
    res_function=f(*positional,**parameters)
    end_time = datetime.now()
    return end_time-start_time, res_function

def get_null_redirect_output():
    sys=str(platform.system())
    if sys == "Windows":
        return "NUL"
    elif sys =="Linux":
        return "/dev/null"
    else:
        return "NULL"