
'''
 Air Cleaner Control/Sensor Monitoring Program
'''

import gatt
from threading import Thread
import tkinter as tk
import tkinter.font as tkFont
import tkinter.ttk as ttk
import time
from influxdb import InfluxDBClient
import sys
import struct
import math
import numpy as np

manager = gatt.DeviceManager(adapter_name='hci0')
db_thread_start_flag = False
db_interval = 5 # sec
aqi_interval = 10 #sec
aqi_query_dt = 60 # sec
db_client = InfluxDBClient('localhost', 8086, 'hwang', 'qudgns', 'air')
air_service_uuid = '0000aabb-0000-1000-8000-00805f9b34fb'

characteristic_uuids = {
        'mode':         '0000aa01-0000-1000-8000-00805f9b34fb',
        'temperature':  '0000aa02-0000-1000-8000-00805f9b34fb',
        'humidity':     '0000aa03-0000-1000-8000-00805f9b34fb',
        'pm25':         '0000aa04-0000-1000-8000-00805f9b34fb',
        'co2':          '0000aa05-0000-1000-8000-00805f9b34fb',
        #'fan':          '0000aa06-0000-1000-8000-00805f9b34fb',
    }

'''
calc aqi with 
'''
def cal_aqi_pm25(cp):
    bp = (0, 15)
    i = (0, 50)
    if cp>=0 and cp<=15:
        bp = (0, 15)
        i = (0, 50)
    elif cp>=16 and cp<=35:
        bp = (16, 35)
        i = (51, 100)
    elif cp>=36 and cp<=75:
        bp = (36, 75)
        i = (101, 250)
    else:
        bp = (76, 500)
        i = (251, 500)

    return (i[1]-i[0])/(bp[1]-bp[0])*(cp-bp[0])+i[0]

'''
calc humidex (from E.C.Thom)
'''
def calc_humidex(h, t):
    #tdp = t-(100-h)/5 #approximation
    tdp = (243.12*(math.log(h/100)+(17.62*t)/(243.12+t)))/(17.62-(math.log(h/100)+(17.62*t)/(243.12+t)))
    return t+0.555*(6.11*math.exp(5417.7530*(1/273.16-1/(273.15+tdp)))-10)
    

class ACController(gatt.Device):
    
    def read_characteristic_value(self, ch_uuid):
        for s in self.services:
            if s.uuid == air_service_uuid:
                self.device_service = s
                break
            else:
                self.device_service = None
        
        if self.device_service:
            for c in self.device_service.characteristics:
                c.read_value()
                time.sleep(0.5)

    def device_discovered(self, device):
        print("Discovered [%s] %s" % (device.mac_address, device.alias()))

    def connect_succeeded(self):
        super().connect_succeeded()
        print("[%s] Connected" % (self.mac_address))

    def connect_failed(self, error):
        super().connect_failed(error)
        print("[%s] Connection failed: %s" % (self.mac_address, str(error)))

    def disconnect_succeeded(self):
        super().disconnect_succeeded()
        print("[%s] Disconnected" % (self.mac_address))

    def services_resolved(self):
        super().services_resolved()
        global db_thread_start_flag
        db_thread_start_flag = True

        print("[%s] Resolved services" % (self.mac_address))

        for service in self.services:
            print("[%s]  Service [%s]" % (self.mac_address, service.uuid))
            for characteristic in service.characteristics:
                print("[%s]  Characteristic [%s]" % (self.mac_address, characteristic.uuid))

    def characteristic_value_updated(self, characteristic, value):
        vlist = [bytes([v]) for v in value]
        str_hex = ''.join([v.hex() for v in vlist[::-1]])
        # print("updated ", characteristic.uuid, " : ", str_hex)
        global characteristic_uuids

        if characteristic.uuid == characteristic_uuids['mode']:
            print("mode : ", int(str_hex, 16))
            db_client.write_points([{"measurement": "sensor","fields": {"mode": int(str_hex, 16)}}])
        elif characteristic.uuid == characteristic_uuids['temperature']:
            print("temperature :", struct.unpack('!f', bytes.fromhex(str_hex))[0])
            db_client.write_points([{"measurement": "sensor","fields": {"temperature": struct.unpack('!f', bytes.fromhex(str_hex))[0]}}])
        elif characteristic.uuid == characteristic_uuids['humidity']:
            print("humidity : ", struct.unpack('!f', bytes.fromhex(str_hex))[0])
            db_client.write_points([{"measurement": "sensor","fields": {"humidity": struct.unpack('!f', bytes.fromhex(str_hex))[0]}}])
        elif characteristic.uuid == characteristic_uuids['pm25']:
            print("pm25 : ", int(str_hex, 16))
            db_client.write_points([{"measurement": "sensor","fields": {"pm25": int(str_hex, 16)}}])
        elif characteristic.uuid == characteristic_uuids['co2']:
            print("co2 : ", int(str_hex, 16))
            db_client.write_points([{"measurement": "sensor","fields": {"co2": int(str_hex, 16)}}])
        elif characteristic.uuid == characteristic_uuids['fan']:
            print("fan : ", int(str_hex, 16))
            db_client.write_points([{"measurement": "sensor","fields": {"co2": int(str_hex, 16)}}])

    def characteristic_read_value_failed(self, characteristic, error):
        if (characteristic.uuid in list(characteristic_uuids.values())):
            print("Read Error : ", characteristic.uuid, error)
        
            


# global
device = ACController(mac_address='8C:AA:B5:BE:CA:2E', manager=manager)

'''
Bluetooth handler
'''
def bt_thread_work(manager, interval_s):
    manager.run()

'''
GUI window
'''
def win_thread_work():
    window = tk.Tk()
    window.title("Air Quality Monitoring System")
    window.geometry("1024x600+50+50") # screen resolution
    fontStyle = tkFont.Font(family="Lucida Grande", size=20) # font style

    tab_window = ttk.Notebook(window, width=900, height=550)
    tab_window.pack()

    frame_temperature = tkinter.Frame(window)

    top_label = tk.Label(window, text="Air Quality Monitoring System", font=fontStyle)
    top_label.grid(row=0, column=0, sticky="n")
    # top_label.pack()

    window.attributes('-fullscreen', True)
    window.mainloop()


'''
DB interoperability
'''
def db_thread_work():

    global db_thread_start_flag
    global db_interval

    while True:
        if db_thread_start_flag==True:
            device.read_characteristic_value('null')
            
        time.sleep(db_interval)


'''
Thread function for AQI
'''
def aqi_thread_work():
    global aqi_interval
    global aqi_query_dt

    while True:
        if db_thread_start_flag==True:
            fields = ', '.join(["mean({0}) as {0}".format(v) for v in list(characteristic_uuids.keys())])
            queryset = "SELECT {fields} FROM air.autogen.sensor WHERE time > now()-{dt}s AND time < now()".format(fields=fields, dt=aqi_query_dt)
            result = db_client.query(queryset)
            query_result = result.raw['series'][0]
            out = dict(zip(query_result["columns"], query_result["values"][0]))

            if out["temperature"]!=None and out["humidity"]!=None:
                humidex = calc_humidex(h=out["humidity"], t=out["temperature"])
                db_client.write_points([{"measurement": "aqi","fields": {"humidex": humidex}}])

            if out["pm25"]!=None:
                ip = cal_aqi_pm25(out["pm25"])
                db_client.write_points([{"measurement": "aqi","fields": {"aqi_pm25": ip}}])
            

        time.sleep(aqi_interval)


if __name__ == '__main__':

    try:
        # BLE device connect
        device.connect()

        bt_thread = Thread(target=bt_thread_work, args=(manager, 10))
        # win_thread = Thread(target=win_thread_work)
        db_thread = Thread(target=db_thread_work)
        aqi_thread = Thread(target=aqi_thread_work)

        # starting thread
        bt_thread.start()
        db_thread.start()
        aqi_thread.start()
        #win_thread.start()

        # join thread
        bt_thread.join()
        db_thread.join()
        aqi_thread.join()
        #win_thread.join()
    except KeyboardInterrupt:
        device.disconnect()
        sys.exit(0)

