from uiutils import *
import random
import string
import socket
import os

sys.path.append("..")


def generate_wifi_ssid():
    prefix = "xgo-"
    suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    return prefix + suffix


def generate_wifi_password():
    return "".join(random.choices(string.ascii_letters + string.digits, k=8))


font1 = ImageFont.truetype("/home/pi/RaspberryPi-CM4/model/msyh.ttc", 20)


def lcd_text(x, y, content):
    draw.text((x, y), content, fill="WHITE", font=font1)


def get_ip(ifname):
    import socket, struct, fcntl

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return socket.inet_ntoa(
        fcntl.ioctl(
            s.fileno(), 0x8915, struct.pack("256s", bytes(ifname[:15], "utf-8"))
        )[20:24]
    )


fm_logo = Image.open("/home/pi/RaspberryPi-CM4/pics/wifi@2x.png")
re_logo = Image.open("/home/pi/RaspberryPi-CM4/pics/redian@2x.png")


splash_theme_color = (15, 21, 46)
purple = (24, 47, 223)
draw.rectangle([(0, 0), (320, 240)], fill=splash_theme_color)


ssid = generate_wifi_ssid()
password = generate_wifi_password()

lcd_text(77, 115, "SSID:" + ssid)
lcd_text(77, 150, "PWD:" + password)
splash.paste(fm_logo, (77, 188), fm_logo)
splash.paste(re_logo, (115, 15), re_logo)
display.ShowImage(splash)

hotspot_cmd = "sudo nmcli device wifi hotspot ssid {} password {}".format(
    ssid, password
)
print(hotspot_cmd)
os.system(hotspot_cmd)
resume_cmd = "sudo nmcli con delete Hotspot"
ip_address = get_ip("wlan0")
lcd_text(102, 185, ip_address)
display.ShowImage(splash)

while True:
    if button.press_a() or button.press_b() or button.press_c() or button.press_d():
        # os._exit(0)
        break

os.system(resume_cmd)
