implementing a crnn in pytorch for recognizing telugu printed text and handwritten

Disable Wifi (Uncheck Enable Wi-Fi)
Go to network connection (Edit Connections...)
Click "Add"
Choose "Wi-Fi" and click "Create"
Type in Connection name like "wifi-hotspot"
Type in SSID as you wish
Choose Device MAC Address from the dropdown (wlan0)
Wifi Security select "WPA & WPA2 Personal" and set a password.
Go to IPv4 Settings tab, from Method drop-down box select Shared to other computers.
Then save and close.

Open Terminal (Ctrl+Alt+T) and type in the following command with your connection name used in step 5.

sudo gedit /etc/NetworkManager/system-connections/wifi-hotspot

Find mode=infrastructure and change it to mode=ap

Now check the network section where wi-fi will be connected to the created hotspot automatically. If you can not find it, go to Connect to Hidden Network... Find the connection and connect to i
