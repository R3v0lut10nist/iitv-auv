import RPi.GPIO as GPIO
from time import sleep
from motor_movement import *

GPIO.setmode(GPIO.BCM)
Motor1=02
Motor2=03
Motor1P=27
Motor1N=22
Motor2P=23
Motor2N=24

base_speed = 50

motor1_speed = 0
motor2_speed = 0
error = 0
last_error = 0
integral = 0

max_speed = 100
min_speed = -20


kp = 0
kd = 0
ki = 0

speed_error = 0
base_speed1 = base_speed
base_speed2 = base_speed

activate_pins(Motor1, Motor2, Motor1P, Motor1N, Motor2P, Motor2N)
pwm1, pwm2 = start_pwm(base_speed, Motor1, Motor2)

while 1:
    error = 0
    speed_error = kp*error + ki*integral - kd*(error-last_error)
    integral += error
    last_error = error
    motor1_speed = base_speed1 + speed_error
    motor2_speed = base_speed2 - speed_error
    if motor1_speed>0 && motor2_speed>0:
        forward(Motor1P,Motor1N,Motor2P,Motor2N)
    elif motor1_speed>0 && motor2_speed<0:
        motor2_speed = -motor2_speed
        right(Motor1P,Motor1N,Motor2P,Motor2N)
    elif motor1_speed<0 && motor2_speed>0:
        motor1_speed = -motor1_speed
        left(Motor1P,Motor1N,Motor2P,Motor2N)

    set_motor_speed(pwm1,pwm2,motor1_speed,motor2_speed)
