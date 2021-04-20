
'''
 Air Cleaner Control/Sensor Monitoring Program
'''

import gatt
from threading import Thread
import tkinter as tk

manager = gatt.DeviceManager(adapter_name='hci0')

class ACController(gatt.Device):
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

        print("[%s] Resolved services" % (self.mac_address))

        for service in self.services:
            print("[%s]  Service [%s]" % (self.mac_address, service.uuid))
            for characteristic in service.characteristics:
                print("[%s]    Characteristic [%s]" % (self.mac_address, characteristic.uuid))


        device_service = next(
            s for s in self.services
            if s.uuid == '0000aabb-0000-1000-8000-00805f9b34fb')

        attr = next(
            c for c in device_service.characteristics
            if c.uuid == '0000aa02-0000-1000-8000-00805f9b34fb')

        attr.read_value()

    def characteristic_value_updated(self, characteristic, value):
        print("updated value:", value.decode("utf-8"))


# device = AnyDevice(mac_address='8C:AA:B5:BE:CA:2E', manager=manager)
# device = AnyDevice(mac_address='40:A3:CC:E9:8E:D1', manager=manager)
# device.connect()
# manager.run()

# manager = AnyDevice(adapter_name='hci0')
# manager.start_discovery()

def bt_thread_work(manager, interval_s):
    manager.run()

def win_thread_work():
    window = tk.Tk()
    t = tk.Label(text="Air Quality Monitoring System")
    t.pack()

    window.mainloop()





if __name__ == '__main__':

    # BLE device connect
    device = ACController(mac_address='8C:AA:B5:BE:CA:2E', manager=manager)
    device.connect()

    bt_thread = Thread(bt_thread_work, args=(manager, 10))
    bt_window = Thread(win_thread_work, args=())

    # starting thread
    bt_thread.start()
    win_thread.start()

    # join thread
    bt_thread.join()
    win_thread.join()

