import RPi.GPIO as GPIO
from time import sleep
from motor_movement import *
from gateCV import *

makeConnection()
setGateParams()
    
GPIO.setmode(GPIO.BCM)
Motor1=03
Motor2=02
Motor1P=24
Motor1N=23
Motor2P=27
Motor2N=22

base_speed = 40

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
set_motor_speed(pwm1,pwm2,30,70)

while 1:
	motor1_speed, motor2_speed, speed_error = startCV()
    error = 0
    speed_error = kp*error + ki*integral - kd*(error-last_error)
    integral += error
    last_error = error
    motor1_speed = base_speed1 + speed_error
    motor2_speed = base_speed2 - speed_error
    if motor1_speed>0 && motor2_speed>0:
		forward(Motor1P,Motor1N,Motor2P,Motor2N)
		sleep(5000)
    elif motor1_speed>0 && motor2_speed<0:
        motor2_speed = -motor2_speed
		right(Motor1P,Motor1N,Motor2P,Motor2N)
		sleep(5000)
    elif motor1_speed<0 && motor2_speed>0:
        motor1_speed = -motor1_speed
		left(Motor1P,Motor1N,Motor2P,Motor2N)
		sleep(5000)
