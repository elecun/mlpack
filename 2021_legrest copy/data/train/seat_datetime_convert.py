import pandas as pd
import numpy as np
import dateutil.parser as dp
import sys


def datetime_convert(target_file):
    
    # set filepath
    filepath = "./"+target_file+".xls"
    print("converting file datetime format : ", filepath)

    # time format conversion (iso format to sec)
    #ref_datetime = pd.to_datetime([dp.isoparse(target_file)]).astype(int)/10**9
    #print("Start Datetime(sec) :",ref_datetime[0])

    # read seat data
    seat_data = pd.read_excel(filepath, header=0, index_col=False)
    #fsr_data = pd.read_csv(filepath, header=0, index_col=False)

    # get time column and time conversion
    seat_datetime = seat_data["Measurement time"]
    #print(seat_datetime.values)

    # split string
    splited = [i.split('-') for i in seat_datetime.values]
    converted = ['-'.join(i[0:3])+" "+':'.join(i[3:6])+"."+i[6] for i in splited]
    #print(converted)

    # insert new column for time
    seat_data.insert(0, "mtime", converted, True)
    #print(seat_data)

    # write file
    seat_data.to_csv("./"+target_file+"m.csv", index=False)


if __name__ == '__main__':
    args = sys.argv
    del args[0]
    if len(args)==1:
        datetime_convert(args[0])
        print("done.")
    else:
        print("too many arguments. only 1 argument(ex. data_01)")