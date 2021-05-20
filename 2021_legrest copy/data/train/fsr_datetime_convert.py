import pandas as pd
import numpy as np
import dateutil.parser as dp
import sys


def datetime_convert(target_file):
    
    # set filepath
    filepath = "./"+target_file+"-1d.csv"
    print("converting file datetime format : ", filepath)

    # time format conversion (iso format to sec)
    ref_datetime = pd.to_datetime([dp.isoparse(target_file)]).astype(int)/10**9
    #print("Start Datetime(sec) :",ref_datetime[0])

    # read fsr data
    fsr_data = pd.read_csv(filepath, header=0, index_col=False)

    # get time column and time conversion
    fsr_data_time_s = fsr_data["Measurement Time (sec)"]+ref_datetime[0]

    # converting to timestamp format
    converted_datetime = pd.to_datetime(fsr_data_time_s, unit='s')
    #print(converted_datetime)

    # insert new column for time
    fsr_data.insert(0, "mtime", converted_datetime, True)
    #print(fsr_data)

    # write file
    fsr_data.to_csv("./"+target_file+"-1dm.csv", index=False)


if __name__ == '__main__':
    args = sys.argv
    del args[0]
    if len(args)==1:
        datetime_convert(args[0].split('-')[0])
        print("done.")
    else:
        print("too many arguments. only 1 argument(ex. 20210303T160024)")