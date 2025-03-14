from demos.uiutils import *

import socket

current_selection = 1


def show_battery():
    lcd_rect(200, 0, 320, 25, color=splash_theme_color, thickness=-1)
    draw.bitmap((270, 4), bat)
    try:
        battery = dog.read_battery()
        print(battery)
        if str(battery) == "0":
            print("uart error")
            lcd_rect(200, 0, 320, 15, color=splash_theme_color, thickness=-1)
            draw.bitmap((270, 4), bat)
        else:
            if len(str(battery)) == 3:
                lcd_draw_string(
                    draw, 274, 4, str(battery), color=color_white, scale=font1
                )
            elif len(str(battery)) == 2:
                lcd_draw_string(
                    draw, 280, 4, str(battery), color=color_white, scale=font1
                )
            elif len(str(battery)) == 1:
                lcd_draw_string(
                    draw, 286, 4, str(battery), color=color_white, scale=font1
                )
            else:
                pass
    except:
        print("uart error!")


# draw methods


def main_program():
    global key_state_left, key_state_right, key_state_down, current_selection

    key_state_left = 0
    key_state_down = 0
    key_state_right = 0

    if button.press_a():
        key_state_down = 1
        key_state_left = 0
        key_state_right = 0
    elif button.press_c():
        key_state_down = 0
        key_state_left = 1
        key_state_right = 0
    elif button.press_d():
        key_state_down = 0
        key_state_left = 0
        key_state_right = 1
    elif button.press_b():
        print("b button,but nothing to quit")

    if key_state_left == 1:
        show_battery()
        if current_selection == 1:
            current_selection = 3
        else:
            current_selection -= 1

    if key_state_right == 1:
        show_battery()
        if current_selection == 3:
            current_selection = 1
        else:
            current_selection += 1

    if current_selection == 1:
        lcd_rect(0, 188, 320, 240, color=btn_unselected, thickness=-1)
        lcd_rect(0, 188, 110, 240, color=btn_selected, thickness=-1)
        lcd_draw_string(draw, 7, 195, la["MAIN"]["RC"], color=color_white, scale=font2)
        lcd_draw_string(
            draw, 112, 195, la["MAIN"]["PROGRAM"], color=color_white, scale=font2
        )
        lcd_draw_string(
            draw, 215, 195, la["MAIN"]["TRYDEMO"], color=color_white, scale=font2
        )
        draw.line((110, 188, 110, 240), fill=txt_unselected, width=1, joint=None)
        draw.line((210, 188, 210, 240), fill=txt_unselected, width=1, joint=None)
        draw.rectangle((0, 188, 320, 240), outline=txt_unselected, width=1)
    elif current_selection == 2:
        lcd_rect(0, 188, 320, 240, color=btn_unselected, thickness=-1)
        lcd_rect(110, 188, 210, 240, color=btn_selected, thickness=-1)
        lcd_draw_string(
            draw, 112, 195, la["MAIN"]["PROGRAM"], color=color_white, scale=font2
        )
        lcd_draw_string(draw, 7, 195, la["MAIN"]["RC"], color=color_white, scale=font2)
        lcd_draw_string(
            draw, 215, 195, la["MAIN"]["TRYDEMO"], color=color_white, scale=font2
        )
        draw.line((110, 188, 110, 240), fill=txt_unselected, width=1, joint=None)
        draw.line((210, 188, 210, 240), fill=txt_unselected, width=1, joint=None)
        draw.rectangle((0, 188, 320, 240), outline=txt_unselected, width=1)
    elif current_selection == 3:
        lcd_rect(0, 188, 320, 240, color=btn_unselected, thickness=-1)
        lcd_rect(210, 188, 320, 240, color=btn_selected, thickness=-1)
        lcd_draw_string(draw, 7, 195, la["MAIN"]["RC"], color=color_white, scale=font2)
        lcd_draw_string(
            draw, 112, 195, la["MAIN"]["PROGRAM"], color=color_white, scale=font2
        )
        lcd_draw_string(
            draw, 215, 195, la["MAIN"]["TRYDEMO"], color=color_white, scale=font2
        )
        draw.line((110, 188, 110, 240), fill=txt_unselected, width=1, joint=None)
        draw.line((210, 188, 210, 240), fill=txt_unselected, width=1, joint=None)
        draw.rectangle((0, 188, 320, 240), outline=txt_unselected, width=1)

    if key_state_down == 1:
        show_battery()
        if current_selection == 2:
            print("edublock")
            lcd_rect(0, 188, 160, 240, color=btn_selected, thickness=-1)
            lcd_draw_string(
                draw, 25, 195, la["MAIN"]["OPENING"], color=color_white, scale=font2
            )
            time.sleep(1)
            os.system("python3 demos/hotspot.py")
            lcd_rect(0, 188, 160, 240, color=btn_selected, thickness=-1)
            lcd_draw_string(
                draw, 25, 195, la["MAIN"]["PROGRAM"], color=color_white, scale=font2
            )

        if current_selection == 1:
            lang = language()
            os.system("python3 app/app_dogzilla_cn.py")
            # if lang=='cn':
            #     os.system('python3 app/app_dogzilla_cn.py')
            # else:
            #     os.system('python3 app/app_dogzilla.py')

        if current_selection == 3:
            lcd_rect(210, 188, 320, 240, color=btn_selected, thickness=-1)
            lcd_draw_string(
                draw, 215, 195, la["MAIN"]["OPENING"], color=color_white, scale=font2
            )
            display.ShowImage(splash)
            print("turn demos")
            os.system("python3 demo.py")

        print(str(current_selection) + " select")
    display.ShowImage(splash)


def is_connected(host="8.8.8.8", port=53, timeout=3):
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error as ex:
        print(f"网络连接错误: {ex}")
        return False


current_dir = os.path.dirname(os.path.abspath(__file__))

logo = Image.open(os.path.join(current_dir, "pics", "luwu@3x.png"))
wifiy = Image.open(os.path.join(current_dir, "pics", "wifi@2x.png"))
bat = Image.open(os.path.join(current_dir, "pics", "battery.png"))

if is_connected():
    draw.bitmap((10, 0), wifiy)
    draw.bitmap((74, 49), logo)
else:
    draw.bitmap((74, 49), logo)

if dog_type == "M":
    firmware_info = "MINI"
    version = "xgomini"
elif dog_type == "L":
    firmware_info = "LITE"
    version = "xgolite"
elif dog_type == "R":
    firmware_info = "RIDER"
    version = "xgorider"

lcd_draw_string(draw, 210, 133, firmware_info, color=color_white, scale=font1)

show_battery()
current_selection = 1

while True:
    main_program()
