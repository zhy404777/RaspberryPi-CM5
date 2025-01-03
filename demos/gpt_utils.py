# ALL LIBS NEED 2024-11-8
import RPi.GPIO as GPIO
from xgolib import XGO
import spidev as SPI
import subprocess
import os, socket, sys, time, json
from PIL import Image, ImageDraw, ImageFont
import xgoscreen.LCD_2inch as LCD_2inch

# import os

# os.environ["OPENAI_API_KEY"] = "sk-jrXKBk2Oyi2TFgdWUcsAplx2C4WhuYCR0pZwlbmI6e28d20i"
# os.environ["OPENAI_BASE_URL"] = "https://api.feidaapi.com/v1"


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
        # last_state = GPIO.input(self.key1)
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
            # while not GPIO.input(self.key1):
            #     time.sleep(0.02)
            return True

    def press_b(self):
        result = subprocess.run(
            ["sudo", "pinctrl", "level", "23"], capture_output=True, text=True
        ).stdout
        if result[0] == "1":
            last_state = True
        elif result[0] == "0":
            last_state = False
        # last_state = GPIO.input(self.key2)
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
            # while not GPIO.input(self.key2):
            #     time.sleep(0.02)
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
        # last_state = GPIO.input(self.key3)
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
            # while not GPIO.input(self.key3):
            #     time.sleep(0.02)
            return True

    def press_d(self):
        result = subprocess.run(
            ["sudo", "pinctrl", "level", "22"], capture_output=True, text=True
        ).stdout
        if result[0] == "1":
            last_state = True
        elif result[0] == "0":
            last_state = False
        # last_state = GPIO.input(self.key4)
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
            # while not GPIO.input(self.key4):
            #     time.sleep(0.02)
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
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": speech_text},
        ],
    )
    re = response.choices[0].message.content
    play_anmi = False
    return re


def tts(content):
    global play_anmi
    play_anmi = True
    time.sleep(0.5)
    play_wait_anmi2 = threading.Thread(target=wait_anmi, args=(0,))
    play_wait_anmi2.start()
    client = OpenAI()
    speech_file_path = "speech.mp3"
    response = client.audio.speech.create(model="tts-1", voice="nova", input=content)
    response.stream_to_file(speech_file_path)
    play_anmi = False
    time.sleep(0.5)
    play_anmi = True
    play_wait_anmi = threading.Thread(target=speak_anmi, args=(0,))
    play_wait_anmi.start()
    proc = Popen("mplayer speech.mp3", shell=True)
    proc.wait()
    time.sleep(0.5)
    play_anmi = False


def start_audio(timel=3, save_file="test.wav"):
    end_threshold = 10000
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


def start_audio_camera(timel=3, save_file="test.wav"):
    end_threshold = 10000
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
    from picamera2 import Picamera2

    picam2 = Picamera2()
    picam2.configure(
        picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (320, 240)}
        )
    )
    picam2.start()
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


def SpeechRecognition():
    global play_anmi
    print("gpt speech recognition")
    client = OpenAI()
    audio_file = open("test.wav", "rb")
    transcription = client.audio.transcriptions.create(
        model="whisper-1", file=audio_file
    )
    play_anmi = False
    return transcription.text


def gpt_draw(scr):
    global quitmark
    global play_anmi
    play_anmi = True
    play_wait_anmi1 = threading.Thread(target=draw_anmi, args=(0,))
    play_wait_anmi1.start()
    client = OpenAI()
    response = client.images.generate(
        model="dall-e-3",
        prompt=scr,
        size="1792x1024",
        quality="standard",
        n=1,
    )
    image_url = response.data[0].url
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


def gpt_rec(speech_text):
    global play_anmi
    play_anmi = True
    play_wait_anmi1 = threading.Thread(target=wait_anmi, args=(0,))
    play_wait_anmi1.start()

    client = OpenAI()

    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    image_path = "/home/pi/xgoPictures/rec.jpg"

    base64_image = encode_image(image_path)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is in this image?",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
    )
    result = response.choices[0].message.content
    print(result)
    play_anmi = False
    return result
