
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

manager = gatt.DeviceManager(adapter_name='hci0')
db_thread_start_flag = False
db_interval = 3
db_client = InfluxDBClient('localhost', 8086, 'hwang', 'qudgns', 'air.sensor')
air_service_uuid = '0000aabb-0000-1000-8000-00805f9b34fb'

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
                # if c.uuid == ch_uuid:
                #     c.read_value()
                #     break
                # else:
                #     pass

        #device_service = next((s for s in self.services if s.uuid == air_service_uuid), None)
        # attr = next(c for c in device_service.characteristics if c.uuid == ch_uuid)
        # attr.read_value()

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
        print("value:", characteristic.uuid, ":%s" % [bytes([v]) for v in value])

    def characteristic_read_value_failed(self, characteristic, error):
        print(characteristic, "has error : ", error)

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

    characteristic_uuids = {
        'mode':         '0000aa01-0000-1000-8000-00805f9b34fb',
        'temperature':  '0000aa02-0000-1000-8000-00805f9b34fb',
        'humidity':     '0000aa03-0000-1000-8000-00805f9b34fb',
        'pm25':         '0000aa04-0000-1000-8000-00805f9b34fb',
        'co2':          '0000aa05-0000-1000-8000-00805f9b34fb',
        'fan':          '0000aa06-0000-1000-8000-00805f9b34fb',
    }
    
    points = [
         {
            "measurement": "cpu_load_short",
            "tags": {
                "host": "hwang-Alpha"
            },
            #"time": "2009-11-10T23:00:00Z",
            # "fields": {
            #     "mode": 0.64
            # }
        }
    ]

    global db_thread_start_flag
    global db_interval

    while True:
        if db_thread_start_flag==True:
            fields = {}
            device.read_characteristic_value('null')
            # for ch in characteristic_uuids:
            #     device.read_characteristic_value(characteristic_uuids[ch])
            #     time.sleep(1)

            # byte array conversion

            # insert into database
            #client.write_points(body)
            print("in loop")
            
        time.sleep(db_interval)


if __name__ == '__main__':

    try:
        # BLE device connect
        device.connect()

        bt_thread = Thread(target=bt_thread_work, args=(manager, 10))
        # win_thread = Thread(target=win_thread_work)
        db_thread = Thread(target=db_thread_work)

        # starting thread
        bt_thread.start()
        db_thread.start()
        #win_thread.start()

        # join thread
        bt_thread.join()
        db_thread.join()
        #win_thread.join()
    except KeyboardInterrupt:
        device.disconnect()
        sys.exit(0)

