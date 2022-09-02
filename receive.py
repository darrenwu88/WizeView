import asyncio
from bleak import BleakClient, BleakScanner

address = "6B:62:70:1F:F9:72"
switch_id = "19B10001-E8F2-537E-4F6C-D104768A1214"
count_id = "19C20000-E8F2-537E-4F6C-D104768A1214"
on = bytearray([0x01])
off = bytearray([0x00])

async def discover():
    devices = await BleakScanner.discover()
    for d in devices:
        print(d)


async def main():
    global address,switch_id, count_id,on,off
    #connects
    async with BleakClient(address) as client:
        print("connected to "+address)
        switch_state = await client.read_gatt_char(switch_id)
        if switch_state == off:
            print('now it is off')
        while(True):
            await start(client, switch_id)

        # print("Model Number: {0}".format("".join(map(chr, model_number))))
        # while(True):
        #     count = await client.read_gatt_char(count_id)
        #     print("Count: {0}".format("".join(map(chr, count))))
            # print(count)

async def start(client, switch_id):
    global on,off
    await client.write_gatt_char(switch_id,on)

asyncio.run(main())

# asyncio.run(discover())