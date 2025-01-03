#ALL LIBS NEED 2024-11-8
import RPi.GPIO as GPIO
from xgolib import XGO
import spidev as SPI
import subprocess
import os, socket, sys, time, json
from PIL import Image, ImageDraw, ImageFont
import xgoscreen.LCD_2inch as LCD_2inch

class Button:
    def __init__(self):
        self.key1 = 24
        self.key2 = 23
        self.key3 = 17
        self.key4 = 22
        # GPIO.setup(self.key1, GPIO.IN, GPIO.PUD_UP)
        # GPIO.setup(self.key2, GPIO.IN, GPIO.PUD_UP)
        # GPIO.setup(self.key3, GPIO.IN, GPIO.PUD_UP)
        # GPIO.setup(self.key4, GPIO.IN, GPIO.PUD_UP)
        os.system("sudo pinctrl set 24 ip")
        os.system("sudo pinctrl set 23 ip")
        os.system("sudo pinctrl set 17 ip")
        os.system("sudo pinctrl set 22 ip")

    def press_a(self):
        #last_state = GPIO.input(self.key1)
        result = subprocess.run(["sudo", "pinctrl", "level", "24"],capture_output=True,text=True).stdout
        if result[0]=="1":
            last_state = True
        elif result[0]=="0":
            last_state = False
        if last_state:
            return False
        else:
            while 1:
                result = subprocess.run(["sudo", "pinctrl", "level", "24"],capture_output=True,text=True).stdout
                if result[0]=="1":
                    break
                time.sleep(0.01)
            # while not GPIO.input(self.key1):
            #     time.sleep(0.02)
            return True

    def press_b(self):
        result = subprocess.run(["sudo", "pinctrl", "level", "23"],capture_output=True,text=True).stdout
        if result[0]=="1":
            last_state = True
        elif result[0]=="0":
            last_state = False
        #last_state = GPIO.input(self.key2)
        if last_state:
            return False
        else:
            while 1:
                result = subprocess.run(["sudo", "pinctrl", "level", "23"],capture_output=True,text=True).stdout
                if result[0]=="1":
                    break
                time.sleep(0.01)
            # while not GPIO.input(self.key2):
            #     time.sleep(0.02)
            os.system("pkill mplayer")
            return True

    def press_c(self):
        result = subprocess.run(["sudo", "pinctrl", "level", "17"],capture_output=True,text=True).stdout
        if result[0]=="1":
            last_state = True
        elif result[0]=="0":
            last_state = False
        #last_state = GPIO.input(self.key3)
        if last_state:
            return False
        else:
            while 1:
                result = subprocess.run(["sudo", "pinctrl", "level", "17"],capture_output=True,text=True).stdout
                if result[0]=="1":
                    break
                time.sleep(0.01)
            # while not GPIO.input(self.key3):
            #     time.sleep(0.02)
            return True

    def press_d(self):
        result = subprocess.run(["sudo", "pinctrl", "level", "22"],capture_output=True,text=True).stdout
        if result[0]=="1":
            last_state = True
        elif result[0]=="0":
            last_state = False
        #last_state = GPIO.input(self.key4)
        if last_state:
            return False
        else:
            while 1:
                result = subprocess.run(["sudo", "pinctrl", "level", "22"],capture_output=True,text=True).stdout
                if result[0]=="1":
                    break
                time.sleep(0.01)
            # while not GPIO.input(self.key4):
            #     time.sleep(0.02)
            return True

def load_language():
    current_dir="/home/pi/RaspberryPi-CM4/"
    language_ini_path = "/home/pi/RaspberryPi-CM4/language/language.ini"
    with open(language_ini_path, "r") as f:  # r为标识符，表示只读
        language = f.read()
        print(language)
    language_pack = os.path.join(current_dir, "language", language + ".la")
    with open(language_pack, "r") as f:  # r为标识符，表示只读
        language_json = f.read()
    language_dict = json.loads(language_json)
    return language_dict


def language():
    language_ini_path = "/home/pi/RaspberryPi-CM4/language/language.ini"
    with open(language_ini_path, "r") as f:  # r为标识符，表示只读
        language = f.read()
        print(language)
    return language


la = load_language()

# define colors
btn_selected = (24, 47, 223)
btn_unselected = (20, 30, 53)
txt_selected = (255, 255, 255)
txt_unselected = (76, 86, 127)
splash_theme_color = (15, 21, 46)
color_black = (0, 0, 0)
color_white = (255, 255, 255)
color_red = (238, 55, 59)
# display init
display = LCD_2inch.LCD_2inch()
display.Init()
# button
button = Button()
# type
os.system("sudo chmod 777 -R /dev/ttyAMA0")
dog = XGO(port="/dev/ttyAMA0", version="xgorider")
fm = dog.version
print(fm)
if fm[0] == "M":
    print("XGO-MINI")
    dog_type = "M"
elif fm[0] == "L":
    print("XGO-LITE")
    dog_type = "L"
elif fm[0] == "R":
    print("XGO-RIDER")
    dog_type = "R"

# font
font1 = ImageFont.truetype("/home/pi/RaspberryPi-CM4/model/msyh.ttc", 15)
font2 = ImageFont.truetype("/home/pi/RaspberryPi-CM4/model/msyh.ttc", 22)
font3 = ImageFont.truetype("/home/pi/RaspberryPi-CM4/model/msyh.ttc", 30)
font4 = ImageFont.truetype("/home/pi/RaspberryPi-CM4/model/msyh.ttc", 40)
# init splash
splash = Image.new("RGB", (display.height, display.width), splash_theme_color)
draw = ImageDraw.Draw(splash)
# dog

#----------------------------DRAW METHOD--------------------------------
def lcd_draw_string(
    splash,
    x,
    y,
    text,
    color=(255, 255, 255),
    font_size=1,
    scale=1,
    mono_space=False,
    auto_wrap=True,
    background_color=(0, 0, 0),
):
    splash.text((x, y), text, fill=color, font=scale)

def lcd_rect(x, y, w, h, color, thickness):
    draw.rectangle([(x, y), (w, h)], fill=color, width=thickness)
# ----------------------------SOUND UI----------------------------------
import random
import time

def draw_wave(ch):
    start_x = 40
    start_y = 42
    width, height = 80, 50
    y_center = height // 2
    current_y = y_center
    previous_point = (0 + start_x, y_center + start_y)
    draw.rectangle(
        [(start_x - 1, start_y), (start_x + width, start_y + height)],
        fill=splash_theme_color,
    )

    x = 0
    while x < width:
        segment_length = random.randint(7, 25)
        gap_length = random.randint(4, 20)

        for _ in range(segment_length):
            if x >= width:
                break

            amplitude_change = ch
            current_y += amplitude_change

            if current_y < 0:
                current_y = 0
            elif current_y > height - 1:
                current_y = height - 1

            current_point = (x + start_x, current_y + start_y)
            draw.line([previous_point, current_point], fill="white")
            previous_point = current_point
            x += 1

        for _ in range(gap_length):
            if x >= width:
                break

            current_point = (x + start_x, y_center + start_y)
            draw.line([previous_point, current_point], fill="white", width=2)
            previous_point = current_point
            x += 1

    start_x = 210
    start_y = 42
    width, height = 80, 50
    y_center = height // 2
    current_y = y_center
    previous_point = (0 + start_x, y_center + start_y)
    draw.rectangle(
        [(start_x - 1, start_y), (start_x + width, start_y + height)],
        fill=splash_theme_color,
    )

    x = 0
    while x < width:
        segment_length = random.randint(7, 25)
        gap_length = random.randint(4, 20)

        for _ in range(segment_length):
            if x >= width:
                break

            amplitude_change = ch
            current_y += amplitude_change

            if current_y < 0:
                current_y = 0
            elif current_y > height - 1:
                current_y = height - 1

            current_point = (x + start_x, current_y + start_y)
            draw.line([previous_point, current_point], fill="white")
            previous_point = current_point
            x += 1

        for _ in range(gap_length):
            if x >= width:
                break

            current_point = (x + start_x, y_center + start_y)
            draw.line([previous_point, current_point], fill="white", width=2)
            previous_point = current_point
            x += 1


def draw_cir(ch):
    draw.rectangle([(55, 40), (120, 100)], fill=splash_theme_color)
    draw.rectangle([(205, 40), (270, 100)], fill=splash_theme_color)
    radius = 4
    cy = 70

    centers = [(62, cy), (87, cy), (112, cy), (210, cy), (235, cy), (260, cy)]

    for center in centers:
        random_offset = ch
        new_y = center[1] + random_offset
        new_y2 = center[1] - random_offset

        draw.line([center[0], new_y2, center[0], new_y], fill="white", width=11)

        top_left = (center[0] - radius, new_y - radius)
        bottom_right = (center[0] + radius, new_y + radius)
        draw.ellipse([top_left, bottom_right], fill="white")
        top_left = (center[0] - radius, new_y2 - radius)
        bottom_right = (center[0] + radius, new_y2 + radius)
        draw.ellipse([top_left, bottom_right], fill="white")


mic_logo = Image.open("/home/pi/RaspberryPi-CM4/pics/mic.png")
