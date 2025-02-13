# ALL LIBS NEED 2024-11-8
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
        os.system("sudo pinctrl set 24 ip")
        os.system("sudo pinctrl set 23 ip")
        os.system("sudo pinctrl set 17 ip")
        os.system("sudo pinctrl set 22 ip")

    def press_a(self):
        result = subprocess.run(
            ["sudo", "pinctrl", "level", "24"], capture_output=True, text=True
        ).stdout
        if result[0] == "1":
            last_state = True
        elif result[0] == "0":
            last_state = False
        if last_state:
            return False
        else:
            while 1:
                result = subprocess.run(
                    ["sudo", "pinctrl", "level", "24"], capture_output=True, text=True
                ).stdout
                if result[0] == "1":
                    break
                time.sleep(0.01)
            return True

    def press_b(self):
        result = subprocess.run(
            ["sudo", "pinctrl", "level", "23"], capture_output=True, text=True
        ).stdout
        if result[0] == "1":
            last_state = True
        elif result[0] == "0":
            last_state = False
        if last_state:
            return False
        else:
            while 1:
                result = subprocess.run(
                    ["sudo", "pinctrl", "level", "23"], capture_output=True, text=True
                ).stdout
                if result[0] == "1":
                    break
                time.sleep(0.01)
            os.system("pkill mplayer")
            return True

    def press_c(self):
        result = subprocess.run(
            ["sudo", "pinctrl", "level", "17"], capture_output=True, text=True
        ).stdout
        if result[0] == "1":
            last_state = True
        elif result[0] == "0":
            last_state = False
        if last_state:
            return False
        else:
            while 1:
                result = subprocess.run(
                    ["sudo", "pinctrl", "level", "17"], capture_output=True, text=True
                ).stdout
                if result[0] == "1":
                    break
                time.sleep(0.01)
            return True

    def press_d(self):
        result = subprocess.run(
            ["sudo", "pinctrl", "level", "22"], capture_output=True, text=True
        ).stdout
        if result[0] == "1":
            last_state = True
        elif result[0] == "0":
            last_state = False
        if last_state:
            return False
        else:
            while 1:
                result = subprocess.run(
                    ["sudo", "pinctrl", "level", "22"], capture_output=True, text=True
                ).stdout
                if result[0] == "1":
                    break
                time.sleep(0.01)
            return True


def load_language():
    current_dir = "/home/pi/RaspberryPi-CM4/"
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


# ----------------------------DRAW METHOD--------------------------------
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


from xgolib import XGO
import cv2

btn_selected = (24, 47, 223)
btn_unselected = (20, 30, 53)
txt_selected = (255, 255, 255)
txt_unselected = (76, 86, 127)
splash_theme_color = (15, 21, 46)
color_black = (0, 0, 0)
color_white = (255, 255, 255)
color_red = (238, 55, 59)
mic_purple = (24, 47, 223)

# font
font1 = ImageFont.truetype("/home/pi/RaspberryPi-CM4/model/msyh.ttc", 15)
font2 = ImageFont.truetype("/home/pi/RaspberryPi-CM4/model/msyh.ttc", 16)
font3 = ImageFont.truetype("/home/pi/RaspberryPi-CM4/model/msyh.ttc", 18)

import random
import time

mic_logo = Image.open("/home/pi/RaspberryPi-CM4/pics/mic.png")
mic_wave = Image.open("/home/pi/RaspberryPi-CM4/pics/mic_wave.png")
offline_logo = Image.open("/home/pi/RaspberryPi-CM4/pics/offline.png")
draw_logo = Image.open("/home/pi/RaspberryPi-CM4/pics/gpt_draw.png")


def clear_top():
    draw.rectangle([(0, 0), (320, 111)], fill=splash_theme_color)


def clear_bottom():
    draw.rectangle([(0, 111), (320, 240)], fill=splash_theme_color)


def draw_wave(ch):
    if ch > 10:
        ch = 10
    start_x = 40
    start_y = 32
    width, height = 80, 50
    y_center = height // 2
    current_y = y_center
    previous_point = (0 + start_x, y_center + start_y)
    clear_top()
    # draw.rectangle([(0, 0), (320, 240)], fill=splash_theme_color)
    draw.bitmap((145, 40), mic_logo, mic_purple)
    x = 0
    while x < width:
        segment_length = random.randint(7, 25)
        gap_length = random.randint(4, 20)

        for _ in range(segment_length):
            if x >= width:
                break

            amplitude_change = random.randint(-ch, ch)
            current_y += amplitude_change

            if current_y < 0:
                current_y = 0
            elif current_y > height - 1:
                current_y = height - 1

            current_point = (x + start_x, current_y + start_y)
            draw.line([previous_point, current_point], fill=mic_purple)
            previous_point = current_point
            x += 1

        for _ in range(gap_length):
            if x >= width:
                break

            current_point = (x + start_x, y_center + start_y)
            draw.line([previous_point, current_point], fill=mic_purple, width=2)
            previous_point = current_point
            x += 1

    start_x = 210
    start_y = 32
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

            amplitude_change = random.randint(-ch, ch)
            current_y += amplitude_change

            if current_y < 0:
                current_y = 0
            elif current_y > height - 1:
                current_y = height - 1

            current_point = (x + start_x, current_y + start_y)
            draw.line([previous_point, current_point], fill=mic_purple)
            previous_point = current_point
            x += 1

        for _ in range(gap_length):
            if x >= width:
                break

            current_point = (x + start_x, y_center + start_y)
            draw.line([previous_point, current_point], fill=mic_purple, width=2)
            previous_point = current_point
            x += 1


def draw_cir(ch):
    if ch > 15:
        ch = 15
    clear_top()
    draw.bitmap((145, 40), mic_logo, mic_purple)
    radius = 4
    cy = 60

    centers = [(62, cy), (87, cy), (112, cy), (210, cy), (235, cy), (260, cy)]

    for center in centers:
        random_offset = random.randint(0, ch)
        new_y = center[1] + random_offset
        new_y2 = center[1] - random_offset

        draw.line([center[0], new_y2, center[0], new_y], fill=mic_purple, width=11)

        top_left = (center[0] - radius, new_y - radius)
        bottom_right = (center[0] + radius, new_y + radius)
        draw.ellipse([top_left, bottom_right], fill=mic_purple)
        top_left = (center[0] - radius, new_y2 - radius)
        bottom_right = (center[0] + radius, new_y2 + radius)
        draw.ellipse([top_left, bottom_right], fill=mic_purple)


def draw_wait(j):
    center = (161, 56)
    initial_radius = 50 - j * 8
    clear_top()
    for i in range(4 - j):
        radius = initial_radius - i * 8
        color_intensity = 223 - (3 - i) * 50
        blue_color = (24, 47, color_intensity)

        draw.ellipse(
            [
                center[0] - radius,
                center[1] - radius,
                center[0] + radius,
                center[1] + radius,
            ],
            outline=blue_color,
            fill=blue_color,
            width=2,
        )

    draw.bitmap(
        (145, 40),
        mic_wave,
    )
    display.ShowImage(splash)


def draw_play():
    j = 0
    center = (161, 56)
    initial_radius = 50 - j * 8
    # draw.rectangle([(40, 0), (240, 120)], fill=splash_theme_color)
    for i in range(4 - j):
        radius = initial_radius - i * 8
        color_intensity = 223 - (3 - i) * 50
        blue_color = (24, 47, color_intensity)

        draw.ellipse(
            [
                center[0] - radius,
                center[1] - radius,
                center[0] + radius,
                center[1] + radius,
            ],
            outline=blue_color,
            fill=blue_color,
            width=2,
        )

    # draw.bitmap(
    #     (145, 40),
    #     mic_wave,
    # )

    image_height = 50
    line_width = 3
    spacing = 3
    start_x = 146 + spacing
    for _ in range(5):
        line_length = random.randint(3, 20)
        start_y = (image_height - line_length) // 2 + 30
        end_y = start_y + line_length
        draw.line((start_x, start_y, start_x, end_y), fill="white", width=line_width)
        draw.point((start_x, start_y - 1), fill="white")
        draw.point((start_x, end_y + 1), fill="white")
        start_x += line_width + spacing

    display.ShowImage(splash)


def draw_offline():
    draw.bitmap((115, 20), offline_logo, "red")
    warn_text = "Wifi unconnected"
    draw.text((90, 140), warn_text, fill=(255, 255, 255), font=font3)
    display.ShowImage(splash)


def draw_draw(j):
    center = (161, 86)
    initial_radius = 50 - j * 8
    draw.rectangle([(0, 0), (320, 240)], fill=splash_theme_color)
    for i in range(4 - j):
        radius = initial_radius - i * 8
        color_intensity = 223 - (3 - i) * 50
        blue_color = (24, 47, color_intensity)

        draw.ellipse(
            [
                center[0] - radius,
                center[1] - radius,
                center[0] + radius,
                center[1] + radius,
            ],
            outline=blue_color,
            fill=blue_color,
            width=2,
        )

    draw.bitmap(
        (147, 72),
        draw_logo,
    )
    display.ShowImage(splash)


# -------------------------------------------------------------------------------
import threading
from subprocess import Popen
import requests
import base64

from libnyumaya import AudioRecognition, FeatureExtractor
from auto_platform import AudiostreamSource, play_command, default_libpath
from datetime import datetime

from openai import OpenAI
import os
from volcenginesdkarkruntime import Ark

import pyaudio
import wave
import numpy as np
from scipy import fftpack


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


button = Button()
play_anmi = True


def wait_anmi(num):
    global play_anmi
    while 1:
        if play_anmi == False:
            break
        draw_wait(3)
        time.sleep(0.4)
        if play_anmi == False:
            break
        draw_wait(2)
        time.sleep(0.4)
        if play_anmi == False:
            break
        draw_wait(1)
        time.sleep(0.4)
        if play_anmi == False:
            break
        draw_wait(0)
        time.sleep(0.4)


def draw_anmi(num):
    global play_anmi
    while 1:
        if play_anmi == False:
            break
        draw_draw(3)
        time.sleep(0.4)
        if play_anmi == False:
            break
        draw_draw(2)
        time.sleep(0.4)
        if play_anmi == False:
            break
        draw_draw(1)
        time.sleep(0.4)
        if play_anmi == False:
            break
        draw_draw(0)
        time.sleep(0.4)


def speak_anmi(num):
    global play_anmi
    while 1:
        if play_anmi == False:
            break
        draw_play()
        time.sleep(0.05)


def scroll_text_on_lcd(text, x, y, max_lines, delay):
    lines = text.split("\n")
    total_lines = len(lines)
    for i in range(total_lines - max_lines):
        lcd_rect(0, 110, 320, 240, splash_theme_color, -1)
        visible_lines = lines[i : i + max_lines - 1]
        last_line = lines[i + max_lines - 1]

        for j in range(max_lines - 1):
            lcd_draw_string(
                draw,
                x,
                y + j * 20,
                visible_lines[j],
                color=(255, 255, 255),
                scale=font2,
                mono_space=False,
            )
        lcd_draw_string(
            draw,
            x,
            y + (max_lines - 1) * 20,
            last_line,
            color=(255, 255, 255),
            scale=font2,
            mono_space=False,
        )

        display.ShowImage(splash)
        time.sleep(delay)


def get_wav_duration():
    filename = "test.wav"
    with wave.open(filename, "rb") as wav_file:
        n_frames = wav_file.getnframes()
        frame_rate = wav_file.getframerate()

        duration = n_frames / frame_rate
        return duration


def gpt(speech_text):
    global play_anmi
    play_anmi = True
    play_wait_anmi1 = threading.Thread(target=wait_anmi, args=(0,))
    play_wait_anmi1.start()

    client = Ark(api_key=os.environ.get("DOUBAO_API_KEY"))
    completion = client.chat.completions.create(
        model="ep-20250206182007-666hv",
        messages=[
            {"role": "user", "content": speech_text},
        ],
    )
    re = completion.choices[0].message.content
    play_anmi = False
    return re


def start_audio(timel=3, save_file="test.wav"):
    end_threshold = 66666
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = timel
    WAVE_OUTPUT_FILENAME = save_file

    p = pyaudio.PyAudio()
    stream_a = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )
    frames = []
    start_luyin = False
    break_luyin = False
    sum_vol = 0
    audio_stream = AudiostreamSource()
    libpath = "./demos/libnyumaya_premium.so.3.1.0"
    extractor = FeatureExtractor(libpath)
    detector = AudioRecognition(libpath)
    extactor_gain = 1.0
    keywordIdlulu = detector.addModel("./demos/src/lulu_v3.1.907.premium", 0.6)
    bufsize = detector.getInputDataSize()
    audio_stream.start()

    while not break_luyin:
        frame = audio_stream.read(bufsize * 2, bufsize * 2)
        if not frame:
            continue
        features = extractor.signalToMel(frame, extactor_gain)
        prediction = detector.runDetection(features)
        if prediction != 0:
            now = datetime.now().strftime("%d.%b %Y %H:%M:%S")
            if prediction == keywordIdlulu:
                print("lulu detected:" + now)
            os.system(play_command + " ./demos/src/ding.wav")
            break
        data = stream_a.read(CHUNK, exception_on_overflow=False)
        rt_data = np.frombuffer(data, dtype=np.int16)
        fft_temp_data = fftpack.fft(rt_data, rt_data.size, overwrite_x=True)
        fft_data = np.abs(fft_temp_data)[0 : fft_temp_data.size // 2 + 1]
        vol = sum(fft_data) // len(fft_data)
        draw_wave(int(vol / 10000))
        display.ShowImage(splash)
    audio_stream.stop()
    try:
        stream_a.stop_stream()
        stream_a.close()
    except:
        pass
    lcd_rect(30, 40, 320, 90, splash_theme_color, -1)

    stream_a = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    data_list = [99999, 99999, 99999, 99999, 99999, 99999, 99999, 99999, 99999, 99999]
    while not break_luyin:
        data = stream_a.read(CHUNK, exception_on_overflow=False)
        rt_data = np.frombuffer(data, dtype=np.int16)
        fft_temp_data = fftpack.fft(rt_data, rt_data.size, overwrite_x=True)
        fft_data = np.abs(fft_temp_data)[0 : fft_temp_data.size // 2 + 1]
        vol = sum(fft_data) // len(fft_data)
        data_list.pop(0)
        data_list.append(vol)
        kkk = lambda x: float(x) < end_threshold
        if all([kkk(i) for i in data_list]):
            print(data_list)
            break
        frames.append(data)
        print(vol)
        draw_cir(int(vol / 10000))
        display.ShowImage(splash)
    print("auto end")

    try:
        stream_a.stop_stream()
        stream_a.close()
    except:
        pass
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()


camera_on = False
from picamera2 import Picamera2

picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(main={"format": "RGB888", "size": (320, 240)})
)
picam2.start()


def start_audio_camera(timel=3, save_file="test.wav"):
    end_threshold = 66666
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = timel
    WAVE_OUTPUT_FILENAME = save_file

    p = pyaudio.PyAudio()
    stream_a = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )
    frames = []
    start_luyin = False
    break_luyin = False
    sum_vol = 0
    audio_stream = AudiostreamSource()
    libpath = "./demos/libnyumaya_premium.so.3.1.0"
    extractor = FeatureExtractor(libpath)
    detector = AudioRecognition(libpath)
    extactor_gain = 1.0
    keywordIdlulu = detector.addModel("./demos/src/lulu_v3.1.907.premium", 0.6)
    bufsize = detector.getInputDataSize()
    audio_stream.start()

    while not break_luyin:
        frame = audio_stream.read(bufsize * 2, bufsize * 2)
        if not frame:
            continue
        features = extractor.signalToMel(frame, extactor_gain)
        prediction = detector.runDetection(features)
        if prediction != 0:
            now = datetime.now().strftime("%d.%b %Y %H:%M:%S")
            if prediction == keywordIdlulu:
                print("lulu detected:" + now)
            os.system(play_command + " ./demos/src/ding.wav")
            break
        data = stream_a.read(CHUNK, exception_on_overflow=False)
        rt_data = np.frombuffer(data, dtype=np.int16)
        fft_temp_data = fftpack.fft(rt_data, rt_data.size, overwrite_x=True)
        fft_data = np.abs(fft_temp_data)[0 : fft_temp_data.size // 2 + 1]
        vol = sum(fft_data) // len(fft_data)
        draw_wave(int(vol / 10000))
        display.ShowImage(splash)
    audio_stream.stop()
    try:
        stream_a.stop_stream()
        stream_a.close()
    except:
        pass
    lcd_rect(30, 40, 320, 90, splash_theme_color, -1)

    print("take a photo")
    time.sleep(0.5)

    path = "/home/pi/xgoPictures/"
    image = picam2.capture_array()
    filename = "rec"
    cv2.imwrite(path + filename + ".jpg", image)
    image = cv2.resize(image, (320, 240))
    b, g, r = cv2.split(image)
    image = cv2.merge((r, g, b))
    image = cv2.flip(image, 1)
    imgok = Image.fromarray(image)
    display.ShowImage(imgok)
    time.sleep(1)
    cv2.destroyAllWindows()
    print("camera close")

    stream_a = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    data_list = [99999, 99999, 99999, 99999, 99999, 99999, 99999, 99999, 99999, 99999]
    while not break_luyin:
        data = stream_a.read(CHUNK, exception_on_overflow=False)
        rt_data = np.frombuffer(data, dtype=np.int16)
        fft_temp_data = fftpack.fft(rt_data, rt_data.size, overwrite_x=True)
        fft_data = np.abs(fft_temp_data)[0 : fft_temp_data.size // 2 + 1]
        vol = sum(fft_data) // len(fft_data)
        data_list.pop(0)
        data_list.append(vol)
        kkk = lambda x: float(x) < end_threshold
        if all([kkk(i) for i in data_list]):
            break
        frames.append(data)
        print(vol)
        draw_cir(int(vol / 10000))
        display.ShowImage(splash)
    print("auto end")

    try:
        stream_a.stop_stream()
        stream_a.close()
    except:
        pass
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()


def line_break(line):
    LINE_CHAR_COUNT = 19 * 2
    CHAR_SIZE = 20
    TABLE_WIDTH = 4
    ret = ""
    width = 0
    for c in line:
        if len(c.encode("utf8")) == 3:
            if LINE_CHAR_COUNT == width + 1:
                width = 2
                ret += "\n" + c
            else:
                width += 2
                ret += c
        else:
            if c == "\t":
                space_c = TABLE_WIDTH - width % TABLE_WIDTH
                ret += " " * space_c
                width += space_c
            elif c == "\n":
                width = 0
                ret += c
            else:
                width += 1
                ret += c
        if width >= LINE_CHAR_COUNT:
            ret += "\n"
            width = 0
    if ret.endswith("\n"):
        return ret
    return ret + "\n"


def download_image(url, save_path):
    response = requests.get(url)

    if response.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"Image downloaded successfully and saved at {save_path}")
    else:
        print(f"Failed to download image. Status code: {response.status_code}")


def resize_image(image_path, output_path, size=(320, 240)):
    with Image.open(image_path) as img:
        img_resized = img.resize(size)
        img_resized.save(output_path)
        print(f"Image resized successfully and saved at {output_path}")


# ------------------豆包语音识别（一句话识别）----------------------------------
import asyncio
import base64
import gzip
import hmac
import json
import logging
import os
import uuid
import wave
from enum import Enum
from hashlib import sha256
from io import BytesIO
from typing import List
from urllib.parse import urlparse
import time
import websockets

appid = "3984980014"  # 项目的 appid
token = os.environ.get("DOUBAO_TOKEN")  # 项目的 token
cluster = "volcengine_input_common"  # 请求的集群
audio_path = "test.wav"  # 本地音频路径
audio_format = "wav"  # wav 或者 mp3，根据实际音频格式设置

PROTOCOL_VERSION = 0b0001
DEFAULT_HEADER_SIZE = 0b0001

PROTOCOL_VERSION_BITS = 4
HEADER_BITS = 4
MESSAGE_TYPE_BITS = 4
MESSAGE_TYPE_SPECIFIC_FLAGS_BITS = 4
MESSAGE_SERIALIZATION_BITS = 4
MESSAGE_COMPRESSION_BITS = 4
RESERVED_BITS = 8

# Message Type:
CLIENT_FULL_REQUEST = 0b0001
CLIENT_AUDIO_ONLY_REQUEST = 0b0010
SERVER_FULL_RESPONSE = 0b1001
SERVER_ACK = 0b1011
SERVER_ERROR_RESPONSE = 0b1111

# Message Type Specific Flags
NO_SEQUENCE = 0b0000  # no check sequence
POS_SEQUENCE = 0b0001
NEG_SEQUENCE = 0b0010
NEG_SEQUENCE_1 = 0b0011

# Message Serialization
NO_SERIALIZATION = 0b0000
JSON = 0b0001
THRIFT = 0b0011
CUSTOM_TYPE = 0b1111

# Message Compression
NO_COMPRESSION = 0b0000
GZIP = 0b0001
CUSTOM_COMPRESSION = 0b1111


def generate_header(
    version=PROTOCOL_VERSION,
    message_type=CLIENT_FULL_REQUEST,
    message_type_specific_flags=NO_SEQUENCE,
    serial_method=JSON,
    compression_type=GZIP,
    reserved_data=0x00,
    extension_header=bytes(),
):
    """
    protocol_version(4 bits), header_size(4 bits),
    message_type(4 bits), message_type_specific_flags(4 bits)
    serialization_method(4 bits) message_compression(4 bits)
    reserved （8bits) 保留字段
    header_extensions 扩展头(大小等于 8 * 4 * (header_size - 1) )
    """
    header = bytearray()
    header_size = int(len(extension_header) / 4) + 1
    header.append((version << 4) | header_size)
    header.append((message_type << 4) | message_type_specific_flags)
    header.append((serial_method << 4) | compression_type)
    header.append(reserved_data)
    header.extend(extension_header)
    return header


def generate_full_default_header():
    return generate_header()


def generate_audio_default_header():
    return generate_header(message_type=CLIENT_AUDIO_ONLY_REQUEST)


def generate_last_audio_default_header():
    return generate_header(
        message_type=CLIENT_AUDIO_ONLY_REQUEST, message_type_specific_flags=NEG_SEQUENCE
    )


def parse_response(res):
    """
    protocol_version(4 bits), header_size(4 bits),
    message_type(4 bits), message_type_specific_flags(4 bits)
    serialization_method(4 bits) message_compression(4 bits)
    reserved （8bits) 保留字段
    header_extensions 扩展头(大小等于 8 * 4 * (header_size - 1) )
    payload 类似与http 请求体
    """
    protocol_version = res[0] >> 4
    header_size = res[0] & 0x0F
    message_type = res[1] >> 4
    message_type_specific_flags = res[1] & 0x0F
    serialization_method = res[2] >> 4
    message_compression = res[2] & 0x0F
    reserved = res[3]
    header_extensions = res[4 : header_size * 4]
    payload = res[header_size * 4 :]
    result = {}
    payload_msg = None
    payload_size = 0
    if message_type == SERVER_FULL_RESPONSE:
        payload_size = int.from_bytes(payload[:4], "big", signed=True)
        payload_msg = payload[4:]
    elif message_type == SERVER_ACK:
        seq = int.from_bytes(payload[:4], "big", signed=True)
        result["seq"] = seq
        if len(payload) >= 8:
            payload_size = int.from_bytes(payload[4:8], "big", signed=False)
            payload_msg = payload[8:]
    elif message_type == SERVER_ERROR_RESPONSE:
        code = int.from_bytes(payload[:4], "big", signed=False)
        result["code"] = code
        payload_size = int.from_bytes(payload[4:8], "big", signed=False)
        payload_msg = payload[8:]
    if payload_msg is None:
        return result
    if message_compression == GZIP:
        payload_msg = gzip.decompress(payload_msg)
    if serialization_method == JSON:
        payload_msg = json.loads(str(payload_msg, "utf-8"))
    elif serialization_method != NO_SERIALIZATION:
        payload_msg = str(payload_msg, "utf-8")
    result["payload_msg"] = payload_msg
    result["payload_size"] = payload_size
    return result


def read_wav_info(data: bytes = None) -> (int, int, int, int, int):
    with BytesIO(data) as _f:
        wave_fp = wave.open(_f, "rb")
        nchannels, sampwidth, framerate, nframes = wave_fp.getparams()[:4]
        wave_bytes = wave_fp.readframes(nframes)
    return nchannels, sampwidth, framerate, nframes, len(wave_bytes)


class AudioType(Enum):
    LOCAL = 1  # 使用本地音频文件


class AsrWsClient:
    def __init__(self, audio_path, cluster, **kwargs):
        """
        :param config: config
        """
        self.audio_path = audio_path
        self.cluster = cluster
        self.success_code = 1000  # success code, default is 1000
        self.seg_duration = int(kwargs.get("seg_duration", 15000))
        self.nbest = int(kwargs.get("nbest", 1))
        self.appid = kwargs.get("appid", "")
        self.token = kwargs.get("token", "")
        self.ws_url = kwargs.get("ws_url", "wss://openspeech.bytedance.com/api/v2/asr")
        self.uid = kwargs.get("uid", "streaming_asr_demo")
        self.workflow = kwargs.get(
            "workflow", "audio_in,resample,partition,vad,fe,decode,itn,nlu_punctuate"
        )
        self.show_language = kwargs.get("show_language", False)
        self.show_utterances = kwargs.get("show_utterances", False)
        self.result_type = kwargs.get("result_type", "full")
        self.format = kwargs.get("format", "wav")
        self.rate = kwargs.get("sample_rate", 16000)
        self.language = kwargs.get("language", "zh-CN")
        self.bits = kwargs.get("bits", 16)
        self.channel = kwargs.get("channel", 1)
        self.codec = kwargs.get("codec", "raw")
        self.audio_type = kwargs.get("audio_type", AudioType.LOCAL)
        self.secret = kwargs.get("secret", "access_secret")
        self.auth_method = kwargs.get("auth_method", "token")
        self.mp3_seg_size = int(kwargs.get("mp3_seg_size", 10000))

    def construct_request(self, reqid):
        req = {
            "app": {
                "appid": self.appid,
                "cluster": self.cluster,
                "token": self.token,
            },
            "user": {"uid": self.uid},
            "request": {
                "reqid": reqid,
                "nbest": self.nbest,
                "workflow": self.workflow,
                "show_language": self.show_language,
                "show_utterances": self.show_utterances,
                "result_type": self.result_type,
                "sequence": 1,
            },
            "audio": {
                "format": self.format,
                "rate": self.rate,
                "language": self.language,
                "bits": self.bits,
                "channel": self.channel,
                "codec": self.codec,
            },
        }
        return req

    @staticmethod
    def slice_data(data: bytes, chunk_size: int) -> (list, bool):
        """
        slice data
        :param data: wav data
        :param chunk_size: the segment size in one request
        :return: segment data, last flag
        """
        data_len = len(data)
        offset = 0
        while offset + chunk_size < data_len:
            yield data[offset : offset + chunk_size], False
            offset += chunk_size
        else:
            yield data[offset:data_len], True

    def _real_processor(self, request_params: dict) -> dict:
        pass

    def token_auth(self):
        return {"Authorization": "Bearer; {}".format(self.token)}

    def signature_auth(self, data):
        header_dicts = {
            "Custom": "auth_custom",
        }

        url_parse = urlparse(self.ws_url)
        input_str = "GET {} HTTP/1.1\n".format(url_parse.path)
        auth_headers = "Custom"
        for header in auth_headers.split(","):
            input_str += "{}\n".format(header_dicts[header])
        input_data = bytearray(input_str, "utf-8")
        input_data += data
        mac = base64.urlsafe_b64encode(
            hmac.new(self.secret.encode("utf-8"), input_data, digestmod=sha256).digest()
        )
        header_dicts["Authorization"] = (
            'HMAC256; access_token="{}"; mac="{}"; h="{}"'.format(
                self.token, str(mac, "utf-8"), auth_headers
            )
        )
        return header_dicts

    async def segment_data_processor(self, wav_data: bytes, segment_size: int):
        reqid = str(uuid.uuid4())
        # 构建 full client request，并序列化压缩
        request_params = self.construct_request(reqid)
        payload_bytes = str.encode(json.dumps(request_params))
        payload_bytes = gzip.compress(payload_bytes)
        full_client_request = bytearray(generate_full_default_header())
        full_client_request.extend(
            (len(payload_bytes)).to_bytes(4, "big")
        )  # payload size(4 bytes)
        full_client_request.extend(payload_bytes)  # payload
        header = None
        if self.auth_method == "token":
            header = self.token_auth()
        elif self.auth_method == "signature":
            header = self.signature_auth(full_client_request)
        async with websockets.connect(
            self.ws_url, extra_headers=header, max_size=1000000000
        ) as ws:
            # 发送 full client request
            await ws.send(full_client_request)
            res = await ws.recv()
            result = parse_response(res)
            if (
                "payload_msg" in result
                and result["payload_msg"]["code"] != self.success_code
            ):
                return result
            for seq, (chunk, last) in enumerate(
                AsrWsClient.slice_data(wav_data, segment_size), 1
            ):
                # if no compression, comment this line
                payload_bytes = gzip.compress(chunk)
                audio_only_request = bytearray(generate_audio_default_header())
                if last:
                    audio_only_request = bytearray(generate_last_audio_default_header())
                audio_only_request.extend(
                    (len(payload_bytes)).to_bytes(4, "big")
                )  # payload size(4 bytes)
                audio_only_request.extend(payload_bytes)  # payload
                # 发送 audio-only client request
                await ws.send(audio_only_request)
                res = await ws.recv()
                result = parse_response(res)
                if (
                    "payload_msg" in result
                    and result["payload_msg"]["code"] != self.success_code
                ):
                    return result
        return result

    async def execute(self):
        with open(self.audio_path, mode="rb") as _f:
            data = _f.read()
        audio_data = bytes(data)
        if self.format == "mp3":
            segment_size = self.mp3_seg_size
            return await self.segment_data_processor(audio_data, segment_size)
        if self.format != "wav":
            raise Exception("format should in wav or mp3")
        nchannels, sampwidth, framerate, nframes, wav_len = read_wav_info(audio_data)
        size_per_sec = nchannels * sampwidth * framerate
        segment_size = int(size_per_sec * self.seg_duration / 1000)
        return await self.segment_data_processor(audio_data, segment_size)


def execute_one(audio_item, cluster, **kwargs):
    """

    :param audio_item: {"id": xxx, "path": "xxx"}
    :param cluster:集群名称
    :return:
    """
    assert "id" in audio_item
    assert "path" in audio_item
    audio_id = audio_item["id"]
    audio_path = audio_item["path"]
    audio_type = AudioType.LOCAL
    asr_http_client = AsrWsClient(
        audio_path=audio_path, cluster=cluster, audio_type=audio_type, **kwargs
    )
    result = asyncio.run(asr_http_client.execute())
    return {"id": audio_id, "path": audio_path, "result": result}


def SpeechRecognition():
    global play_anmi
    print("gpt speech recognition")
    result = execute_one(
        {"id": 1, "path": audio_path},
        cluster=cluster,
        appid=appid,
        token=token,
        format=audio_format,
    )
    result_text = result["result"]["payload_msg"]["result"][0]["text"]
    play_anmi = False
    return result_text


# --------------------豆包TTS（大模型拟人发音）-----------------------------
import base64
import json
import uuid
import requests


def tts(content):
    global play_anmi
    play_anmi = True
    time.sleep(0.5)
    appid = "3984980014"
    access_token = os.environ.get("DOUBAO_TOKEN")
    cluster = "volcano_tts"

    voice_type = "zh_female_cancan_mars_bigtts"
    host = "openspeech.bytedance.com"
    api_url = f"https://{host}/api/v1/tts"

    header = {"Authorization": f"Bearer;{access_token}"}

    request_json = {
        "app": {"appid": appid, "token": "access_token", "cluster": cluster},
        "user": {"uid": "BigTTS7462686243546714431"},
        "audio": {
            "voice_type": voice_type,
            "encoding": "mp3",
            "speed_ratio": 1.0,
            "volume_ratio": 1.0,
            "pitch_ratio": 1.0,
        },
        "request": {
            "reqid": str(uuid.uuid4()),
            "text": content,
            "text_type": "plain",
            "operation": "query",
            "with_frontend": 1,
            "frontend_type": "unitTson",
        },
    }
    try:
        resp = requests.post(api_url, json.dumps(request_json), headers=header)
        # print(f"resp body: \n{resp.json()}")
        if "data" in resp.json():
            data = resp.json()["data"]
            file_to_save = open("speech.mp3", "wb")
            file_to_save.write(base64.b64decode(data))
    except Exception as e:
        e.with_traceback()
    play_anmi = False
    time.sleep(0.5)
    play_anmi = True
    play_wait_anmi = threading.Thread(target=speak_anmi, args=(0,))
    play_wait_anmi.start()
    proc = Popen("mplayer speech.mp3", shell=True)
    proc.wait()
    time.sleep(0.5)
    play_anmi = False


# ----------------------------豆包视觉理解---------------------------------
import base64
import os
from volcenginesdkarkruntime import Ark


def gpt_rec(speech_text):
    global play_anmi
    play_anmi = True
    play_wait_anmi1 = threading.Thread(target=wait_anmi, args=(0,))
    play_wait_anmi1.start()

    client = Ark(api_key=os.environ.get("DOUBAO_API_KEY"))

    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    image_path = "/home/pi/xgoPictures/rec.jpg"
    base64_image = encode_image(image_path)
    response = client.chat.completions.create(
        model="ep-20250206182007-666hv",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": speech_text,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            # 需要注意：传入Base64编码前需要增加前缀 data:image/{图片格式};base64,{Base64编码}：
                            # PNG图片："url":  f"data:image/png;base64,{base64_image}"
                            # JEPG图片："url":  f"data:image/jpeg;base64,{base64_image}"
                            # WEBP图片："url":  f"data:image/webp;base64,{base64_image}"
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
    )

    re = response.choices[0].message.content
    play_anmi = False
    return re


# ------------------------------豆包文生图---------------------------------------
from volcengine.visual.VisualService import VisualService


def gpt_draw(content):
    global quitmark
    global play_anmi
    play_anmi = True
    play_wait_anmi1 = threading.Thread(target=draw_anmi, args=(0,))
    play_wait_anmi1.start()

    visual_service = VisualService()
    visual_service.set_ak(os.environ.get("DOUBAO_AK"))
    visual_service.set_sk(os.environ.get("DOUBAO_SK"))

    form = {
        "req_key": "high_aes_general_v21_L",
        "prompt": content,
        "model_version": "general_v2.1_L",
        "req_schedule_conf": "general_v20_9B_pe",
        "llm_seed": -1,
        "seed": -1,
        "scale": 3.5,
        "ddim_steps": 25,
        "width": 320,
        "height": 240,
        "use_pre_llm": True,
        "use_sr": True,
        "sr_seed": -1,
        "sr_strength": 0.4,
        "sr_scale": 3.5,
        "sr_steps": 20,
        "is_only_sr": False,
        "return_url": True,
        "logo_info": {
            "add_logo": False,
            "position": 0,
            "language": 0,
            "opacity": 0.3,
            "logo_text_content": "XGO",
        },
    }

    image_url = visual_service.cv_process(form)["data"]["image_urls"][0]
    print(image_url)
    original_image_path = "original.jpg"
    resized_image_path = "resized.jpg"
    download_image(image_url, original_image_path)
    resize_image(original_image_path, resized_image_path)
    play_anmi = False
    time.sleep(0.5)
    image = Image.open("resized.jpg")
    splash.paste(image, (0, 0))
    display.ShowImage(splash)
    draw.rectangle([(0, 0), (320, 240)], fill=splash_theme_color)
    time.sleep(5.5)
