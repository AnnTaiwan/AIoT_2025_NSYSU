import RPi.GPIO as GPIO
import time

def Setup(GPIOnum, OUT_IN):
    GPIO.setmode(GPIO.BCM) # Board od BCM
    if OUT_IN == "OUT":
        GPIO.setup(GPIOnum, GPIO.OUT)
    else: 
        GPIO.setup(GPIOnum, GPIO.IN)

def TurnOnLED(GPIOnum):
    GPIO.output(GPIOnum, True)

def TurnOffLED(GPIOnum):
    GPIO.output(GPIOnum, False)
    
def GetGPIOStatus(GPIOnum):
    GPIO_State = GPIO.input(GPIOnum)
    return GPIO_State

if __name__ == "__main__":
    try:
        Setup(2, "IN")
        print("The status of the GPIO{0} is {1}".format(2, GetGPIOStatus(2)))
        Setup(2, "OUT")
        while True:
            TurnOnLED(2)
            time.sleep(1)
            TurnOffLED(2)
            time.sleep(1)
    except KeyboardInterrupt: # allow press ctrl+c to interrupt the process
        GPIO.cleanup() # clear pin info
