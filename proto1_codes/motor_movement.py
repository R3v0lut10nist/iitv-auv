import RPi.GPIO as GPIO

def forward(Motor1P,Motor1N,Motor2P,Motor2N):
	GPIO.output(Motor1P,GPIO.HIGH)
	GPIO.output(Motor1N,GPIO.LOW)
	GPIO.output(Motor2P,GPIO.HIGH)
	GPIO.output(Motor2N,GPIO.LOW)

def reverse(Motor1P,Motor1N,Motor2P,Motor2N):
	GPIO.output(Motor1N,GPIO.HIGH)
	GPIO.output(Motor1P,GPIO.LOW)
	GPIO.output(Motor2N,GPIO.HIGH)
	GPIO.output(Motor2P,GPIO.LOW)

def right(Motor1P, Motor1N, Motor2P, Motor2N):
    GPIO.output(Motor1P,GPIO.HIGH)
    GPIO.output(Motor1N,GPIO.LOW)
    GPIO.output(Motor2P,GPIO.LOW)
    GPIO.output(Motor2N,GPIO.HIGH)
    
def left(Motor1P, Motor1N, Motor2P, Motor2N):
    GPIO.output(Motor2P,GPIO.HIGH)
    GPIO.output(Motor2N,GPIO.LOW)
    GPIO.output(Motor1P,GPIO.LOW)
    GPIO.output(Motor1N,GPIO.HIGH)    

def activate_pins(Motor1,Motor2,Motor1P,Motor1N,Motor2P,Motor2N):
	GPIO.setup(Motor1,GPIO.OUT)
	GPIO.setup(Motor2,GPIO.OUT)
	GPIO.setup(Motor1P,GPIO.OUT)	
	GPIO.setup(Motor2P,GPIO.OUT)
	GPIO.setup(Motor1N,GPIO.OUT)
	GPIO.setup(Motor2N,GPIO.OUT)

def start_pwm(base_speed,Motor1,Motor2):
	GPIO.output(Motor1,GPIO.HIGH)
	GPIO.output(Motor2,GPIO.HIGH)
	pwm1=GPIO.PWM(Motor1,100)
	pwm2=GPIO.PWM(Motor2,100)
	pwm1.start(base_speed)
	pwm2.start(base_speed)
	return	[pwm1,pwm2]

def set_motor_speed(pwm1,pwm2,duty_cycle1, duty_cycle2):
    pwm1.ChangeDutyCycle(duty_cycle1)
    pwm2.ChangeDutyCycle(duty_cycle2)

def stop(Motor1P,Motor1N,Motor2P,Motor2N):
    GPIO.output(Motor1P,GPIO.HIGH)
    GPIO.output(Motor1N,GPIO.HIGH)
    GPIO.output(Motor2P, GPIO.HIGH)
    GPIO.output(Motor2N, GPIO.HIGH)

def clean(pwm1,pwm2):
    pwm1.stop()
    pwm2.stop()
    GPIO.cleanup()