import paho.mqtt.client as mqtt
from time import time, sleep
import RPi.GPIO as GPIO
from dotenv import load_dotenv
import os
BUZZER_PIN = 20
load_dotenv()

MQTT_USER = os.getenv("MQTT_USER")
MQTT_PASS = os.getenv("MQTT_PASS")
MQTT_TOPIC_NAME = os.getenv("MQTT_TOPIC_NAME")

# Function to turn the buzzer ON (LOW level trigger)
def buzzer_on():
    GPIO.output(BUZZER_PIN, GPIO.LOW)  # LOW triggers the buzzer
    print("Buzzer ON")

# Function to turn the buzzer OFF
def buzzer_off():
    GPIO.output(BUZZER_PIN, GPIO.HIGH)  # HIGH turns off the buzzer
    print("Buzzer OFF")

# Function to make the buzzer beep with a delay
def buzzer_beep(beep_time=0.5, interval=0.5, count=3):
    """
    Makes the buzzer beep multiple times.

    :param beep_time: Duration of each beep (seconds)
    :param interval: Time between beeps (seconds)
    :param count: Number of beeps
    """
    for _ in range(count):
        buzzer_on()
        sleep(beep_time)  # Keep buzzer ON
        buzzer_off()
        sleep(interval)   # Wait before next beep

# Function to play a PWM tone
def play_tone(frequency=1000, duration=1):
    """
    Play a tone using PWM.

    :param frequency: Frequency of the tone in Hz
    :param duration: Duration to play the tone (seconds)
    """
    pwm = GPIO.PWM(BUZZER_PIN, frequency)  # Set PWM frequency
    pwm.start(50)  # Start PWM with 50% duty cycle
    sleep(duration)  # Play for the specified duration
    pwm.stop()  # Stop PWM
    print(f"Playing tone at {frequency} Hz for {duration} sec")

# MQTT server, buzzer is client
def on_connect(client, userdata, flags, reasonCode, properties):
    print("Connected with reasonCode:", reasonCode)
    print("Properties:", properties)
    print("Subscribed to:", MQTT_TOPIC_NAME)
    client.subscribe(MQTT_TOPIC_NAME)
# when getting the msg, execute it
def on_message(client, userdata, message):
    msg = message.payload.decode()
    print(f"Get msg: {msg}")
    if msg == "on":
        play_tone()
    elif msg == "off":
        pass
    else:
        print(f"Not valid message : {msg}")

if __name__ == "__main__":
    try:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(BUZZER_PIN, GPIO.OUT) # buzzer
        # MQTT connect
        client = mqtt.Client(
            client_id="", 
            protocol=mqtt.MQTTv5  #  MQTTv311 or MQTTv5
        )
        client.username_pw_set(MQTT_USER, MQTT_PASS) # use one usr to login
        client.on_connect = on_connect
        client.on_message = on_message

        try:
            client.connect("localhost", 1883)
        except Exception as e:
            print(f"Failed to connect to MQTT Broker: {e}")
            GPIO.cleanup()
            exit(1)
            
        client.loop_forever()
    except KeyboardInterrupt: # allow press ctrl+c to interrupt the process
        GPIO.cleanup() # clear pin info