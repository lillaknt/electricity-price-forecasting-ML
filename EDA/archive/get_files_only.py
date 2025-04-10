
import pandas as pd
import numpy as np
import pytz
import BachelorProject.ETL.dictionary_mappings as dm
import glob
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
xdrive_path = os.path.join(parent_dir, 'xdrive')
sys.path.append(xdrive_path)
import get_files_from_xdrive as gxdrive


def get_file_from_xdrive(file_path):
    return gxdrive.gxdrive.read_file_from_xdrive_as_df(file_path)