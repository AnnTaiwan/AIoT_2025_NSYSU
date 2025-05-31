import time
import RPi.GPIO as GPIO
import LED

MAG_PIN = 21
LED_PIN = 2
if __name__ == "__main__":
    # set up pin type
    GPIO.setmode(GPIO.BCM)
    LED.Setup(LED_PIN, "OUT")
    GPIO.setup(MAG_PIN, GPIO.IN) # magnet switch

    try:
        while True:
            # check the magnet switch state
            a = GPIO.input(MAG_PIN)
            print(a)
            if not a: # Low, when open the door, magnet leave the switch
                LED.TurnOnLED(LED_PIN)
                print("ON")
            else: # high, when close the door, magnet is close to the switch
                LED.TurnOffLED(LED_PIN)
                print("OFF")
    
            time.sleep(1)
    
    except KeyboardInterrupt:
        GPIO.cleanup()
