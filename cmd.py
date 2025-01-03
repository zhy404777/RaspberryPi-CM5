# 导入必需的库
from xgolib import XGO
import time

xgo=XGO(port='/dev/ttyAMA0',version="xgolite")

# 前进5步
xgo.move_x(5)
time.sleep(1)  # 延时控制执行时间
xgo.move_x(0)

# 转一圈
xgo.turn(150)
time.sleep(2.4)  # 根据速度和所需旋转的角度计算转动时间
xgo.turn(0)

# 重置机器狗的状态
xgo.reset()
\