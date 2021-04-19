import gatt

manager = gatt.DeviceManager(adapter_name='auto')

class AnyDevice(gatt.Device):
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


device = AnyDevice(mac_address='8C:AA:B5:BE:CA:2E', manager=manager)
device.connect()

manager.run()