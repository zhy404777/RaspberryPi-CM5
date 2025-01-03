from gpt_utils import *
import re

prompt = """
【Role】Please play the role of an experienced robot developer. You are an expert in Raspberry Pi, robotics, and Python development.
【Task】Generate Python code based on command words for the robot dog using the provided Python library.
【Requirements】The Python code, which is automatically generated based on command words, must output a document in MD format.

The specific Python library is as follows, the Python control interface for the robot dog, including: forward, backward, left shift, right shift, rotate, translate and rotate along the XYZ axis, and perform action groups.
xgo.move_x(step)  #The unit of step is millimeters. Positive is forward, negative is backward, 0 means stop, and the range is [-25,25]mm.
xgo.move_y(step)  #The unit of step is millimeters. Positive is left shift, negative is right shift, 0 means stop, and the range is [-18,18]mm.
xgo.turn(speed)  #Speed is the angular velocity, positive is clockwise, negative is counterclockwise, 0 means stop, and the range is [-150,150].
xgo.pace(mode) #Mode is slow, normal, or high. This represents the pace of the robot dog.
time.sleep(X) #The unit of X is seconds, which indicates the duration of the previous instruction.
xgo.action(id) #id is the action group interface, id ranges from 1-24, corresponding to [lie down, stand up, crawl ,turn around, squat, Turn roll, Turn pitch, Turn yaw, 3 axis motion, tke a pee, sit down, wave hand, stretch, wave body, wave side, pray, looking for food, handshake, Chicken head, push-up,seek,dance,Naughty], i.e. the id for lie down is 1, crawl is 4, pray is 18, grab up is 129 , grab mid is 130 , grab down is 130.
xgo.translation(direction, data)  #The value of direction is 'x', 'y', 'z'. The unit of data is millimeters. Positive along the X-axis means forward, 0 means return to the initial position, and negative along the X-axis means backward. The range is [-35,35]mm. The same applies to the y-axis and z-axis.
xgo.attitude(direction, data)  #The value of direction is 'r', 'p', 'y'. The unit of data is degrees. Positive along the X-axis means clockwise rotation, 0 means return to the initial position, and negative along the X-axis means counterclockwise rotation. The range is [-20,20]mm. The same applies to rotation along the y-axis and z-axis.
arm( arm_x, arm_z) #The range for arm_x is [-80,155] and the range for arm_z is [-95,155]
claw(pos) #The range for pos is 0-255, where 0 means the claw is fully open, 255 means the claw is fully closed.
imu(mode) #The value for mode is 0 or 1, 0 means turn off self-stabilization mode, 1 means turn on self-stabilization mode.
reset()#
lcd_picture(filename)   #This function is used for the robot dog to display expressions, such as attack, anger, disgust, like, naughty, pray, sad, sensitive, sleepy, apologize, surprise.
xgoSpeaker(filename)  #This function is used for the robot dog to bark, such as attack, anger, disgust, like, naughty, pray, sad, sensitive, sleepy, apologize, surprise.
I hope you can generate the corresponding motion code using the above functions according to my command.

Below are some examples in the form of (command, code):

Please add the following two initialization codes before each program
from xgolib import XGO
from xgoedu import XGOEDU
xgo=XGO(port='/dev/ttyAMA0',version="xgolite")
XGO_edu = XGOEDU()

Example 1
Command: Move forward for 5 seconds
Code:
from xgolib import XGO
import time
xgo=XGO(port='/dev/ttyAMA0',version="xgolite")
xgo.move_x(15)
time.sleep(5)
xgo.move_x(0)

Example 2
Command: Shift left for 5 seconds
Code:
from xgolib import XGO
import time
xgo=XGO(port='/dev/ttyAMA0',version="xgolite")
xgo.move_y(15)
time.sleep(5)
xgo.move_y(0)

Example 3
Command: Rotate at an angular velocity of 100 for 3 seconds
Code:
from xgolib import XGO
import time
xgo=XGO(port='/dev/ttyAMA0',version="xgolite")
xgo.turn(100)
time.sleep(3)
xgo.turn(0)

Example 4
Command: Move forward for 3 seconds, urinate, turn left for 3 seconds, show mechanical arm
from xgolib import XGO
import time
xgo=XGO(port='/dev/ttyAMA0',version="xgolite")
xgo.move_x(15)
time.sleep(5)
xgo.move_x(0)
xgo.action(11)
xgo.turn(100)
time.sleep(3)
xgo.turn(0)
xgo.action(20)

Example 5
Command: Display a happy expression, then stretch
from xgolib import XGO
from xgoedu import XGOEDU
import time
xgo=XGO(port='/dev/ttyAMA0',version="xgolite")
XGO_edu = XGOEDU()

xgo.action(14)
time.sleep(3)
XGO_edu.lcd_picture(like) 
The example has ended. Please provide Python code based on the commands, with comments included within the code. The final statement should be xgo.reset() to reset it. You must output the document in Markdown format.
"""


def gpt_cmd(speech_text):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": speech_text},
        ],
    )
    re = response.choices[0].message.content
    return re


def split_string(text):
    import re

    seg = 28
    result = []
    current_segment = ""
    current_length = 0

    for char in text:
        is_chinese = bool(re.match(r"[\u4e00-\u9fa5]", char))

        if is_chinese:
            char_length = 2
        else:
            char_length = 1

        if current_length + char_length <= seg:
            current_segment += char
            current_length += char_length
        else:
            result.append(current_segment)
            current_segment = char
            current_length = char_length

    if current_segment:
        result.append(current_segment)

    return result


import requests

net = False
try:
    html = requests.get("http://www.baidu.com", timeout=2)
    print("net")
    net = True
except:
    pass

if net:
    dog = XGO(port="/dev/ttyAMA0", version="xgolite")
    draw.rectangle((20, 30, 300, 80), splash_theme_color, "white", width=3)
    display.ShowImage(splash)
    while 1:
        start_audio()
        xunfei = ""
        lcd_rect(0, 20, 320, 290, splash_theme_color, -1)
        draw.rectangle((20, 30, 300, 80), splash_theme_color, "white", width=3)
        lcd_draw_string(
            draw,
            35,
            40,
            "Recognizing",
            color=(255, 0, 0),
            scale=font3,
            mono_space=False,
        )
        display.ShowImage(splash)
        try:
            speech_text = SpeechRecognition()
        except:
            speech_text = ""
        if speech_text != "":
            speech_list = split_string(speech_text)
            print(speech_list)
            for sp in speech_list:
                lcd_rect(0, 20, 320, 290, splash_theme_color, -1)
                draw.rectangle((20, 30, 300, 80), splash_theme_color, "white", width=3)
                lcd_draw_string(
                    draw,
                    35,
                    40,
                    sp,
                    color=(255, 0, 0),
                    scale=font2,
                    mono_space=False,
                )
                lcd_draw_string(
                    draw,
                    27,
                    90,
                    "Waiting for chatGPT",
                    color=(255, 255, 255),
                    scale=font2,
                    mono_space=False,
                )
                display.ShowImage(splash)
                time.sleep(1.5)
            res = gpt_cmd(speech_text)
            re_e = line_break(res)
            print(re_e)
            if re_e != "":
                lcd_rect(0, 20, 320, 290, splash_theme_color, -1)
                draw.rectangle((20, 30, 300, 80), splash_theme_color, "white", width=3)
                lcd_draw_string(
                    draw,
                    35,
                    40,
                    "Generating Python code",
                    color=(255, 0, 0),
                    scale=font3,
                    mono_space=False,
                )
                lcd_draw_string(
                    draw,
                    10,
                    90,
                    re_e,
                    color=(255, 255, 255),
                    scale=font2,
                    mono_space=False,
                )
                display.ShowImage(splash)
                with open("cmd.py", "w") as file:
                    code_blocks = re.findall(r"```python(.*?)```", res, re.DOTALL)
                    extracted_code = []
                    for block in code_blocks:
                        code_lines = block.strip().split("\n")
                        extracted_code.append(
                            "\n".join(code_lines)
                        )  # Include all lines, including the first one
                    try:
                        file.write(extracted_code[0])
                    except:
                        file.write(res)
                scroll_text_on_lcd(re_e, 10, 90, 7, 0.3)
                lcd_rect(0, 20, 320, 290, splash_theme_color, -1)
                draw.rectangle((20, 30, 300, 80), splash_theme_color, "white", width=3)
                lcd_draw_string(
                    draw,
                    35,
                    40,
                    "Running Python code",
                    color=(255, 0, 0),
                    scale=font3,
                    mono_space=False,
                )
                lcd_draw_string(
                    draw,
                    10,
                    90,
                    re_e,
                    color=(255, 255, 255),
                    scale=font2,
                    mono_space=False,
                )
                display.ShowImage(splash)
                try:
                    process = subprocess.Popen(["python3", "cmd.py"])
                    exitCode = process.wait()
                except:
                    lcd_rect(0, 20, 320, 290, splash_theme_color, -1)
                    draw.rectangle(
                        (20, 30, 300, 80), splash_theme_color, "white", width=3
                    )
                    lcd_draw_string(
                        draw,
                        10,
                        90,
                        "Code error",
                        color=(255, 255, 255),
                        scale=font2,
                        mono_space=False,
                    )
                    display.ShowImage(splash)

else:
    lcd_draw_string(
        draw,
        57,
        70,
        "Can't run without network!",
        color=(255, 255, 255),
        scale=font2,
        mono_space=False,
    )
    lcd_draw_string(
        draw,
        57,
        120,
        "Press C button to quit.",
        color=(255, 255, 255),
        scale=font2,
        mono_space=False,
    )
    display.ShowImage(splash)
    while 1:
        if button.press_b():
            break
