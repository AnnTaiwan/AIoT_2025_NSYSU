import mysql.connector
from datetime import datetime
import paho.mqtt.client as mqtt

import Adafruit_DHT  # Library for DHT sensors (DHT11, DHT22, AM2302)
from time import time, sleep
import RPi.GPIO as GPIO
from dotenv import load_dotenv
import os
import LED
import random


# init some variables
cnx = 0
cur = 0
DOOR_OPEN_TH = 60 # duration threshold for turn on the buzzer, and send warning
# Define the sensor type and GPIO pin
sensor = Adafruit_DHT.DHT11  # Using the DHT11 temperature and humidity sensor
DHT11_PIN = 14  # GPIO pin number where the sensor is connected
MAG_PIN = 21
LED_PIN = 2
# db data
load_dotenv()

host = os.getenv("MYSQL_HOST")
port = int(os.getenv("MYSQL_PORT"))
user = os.getenv("MYSQL_USER")
password = os.getenv("MYSQL_PASS")
DB_NAME = os.getenv("MYSQL_DB")
TABLE_NAME = os.getenv("TABLE_NAME")

MQTT_USER = os.getenv("MQTT_USER")
MQTT_PASS = os.getenv("MQTT_PASS")
MQTT_TOPIC_NAME = os.getenv("MQTT_TOPIC_NAME")

def get_temp_humi(sensor, pin = DHT11_PIN):
    humidity, temperature = Adafruit_DHT.read_retry(sensor, pin)
    # Check if valid readings were obtained
    if humidity is not None and temperature is not None:
        # Print temperature and humidity readings
        print("Temp={0}°C Humidity={1}%".format(temperature, humidity))
        return humidity, temperature
    else:
        # Print error message if reading fails
        print("Failed to get reading. Try again!")
        return 0, 0


def insert_data(id_name, device, time_data, temp, hum, buz = False, Door = False, Duration = 0, table_name=TABLE_NAME):
    # SQL query to insert data
    sql = f"""
        INSERT INTO {table_name} (ID, Device, Date, Temperature, Humid,	Buzzer, Door, Duration)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    # Data to insert
    data = (id_name, device, time_data, temp, hum, buz, Door, Duration) 

    # Execute the SQL query
    cur.execute(sql, data)

    # Commit changes
    cnx.commit()
    print("Data \033[94minserted\033[0m successfully:", data)
    
def delete_data_by_ID(id_name="a001"):
    # SQL query to delete data
    sql = f"DELETE FROM {TABLE_NAME} WHERE ID = %s"
    
    # Data to insert
    data = (id_name,) 

    # Execute the SQL query
    cur.execute(sql, data)

    # Commit changes
    cnx.commit()
    print(f"Data \033[94mdeleted\033[0m successfully, ID: {data[0]}")

def select_all(table_name=TABLE_NAME):
    # SQL query to delete data
    # select all except the frame(too big)
    sql = "SELECT ID, Device, Date, Temperature, Humid,	Buzzer, Door, Duration FROM " + table_name
    
    # Execute the SQL query
    cur.execute(sql)
    print("List all data in table now:")
    # SELECT  no need to commit update to db
    rows = cur.fetchall()
    for row in rows: # each line is a tuple and one data
        print(row)
    return rows
def select_ID(table_name=TABLE_NAME): # get all the ID
    # SQL query to delete data
    sql = "SELECT ID FROM " + table_name
    
    # Execute the SQL query
    cur.execute(sql)
    print("List all IDs in table now:")
    # SELECT  no need to commit update to db
    rows = cur.fetchall()
    return rows
        
def init_db_table(db_name=DB_NAME, table_name=TABLE_NAME):
    # connect db
    # Connect to MySQL server on your laptop
    global cnx
    cnx = mysql.connector.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=DB_NAME
    )
     # Create a cursor object
    global cur
    cur = cnx.cursor()
    results = select_all(table_name)
    print("PRE", len(results))
    ids = select_ID(table_name)
    # remove the duplicate ids
    ids = list(set(ids))    
    for id_name in ids:   
        # clear the db's all data
        delete_data_by_ID(id_name=id_name[0])
    
    results = select_all(table_name)
    print("POST ", len(results))
    if len(results) != 0: # check if it is empty
        print("ERROR: The db is not empty!")
        return False
    else:
        print(f"✅ Initialize the {table_name} table successfully!")
        return True
    
if __name__ == "__main__":
    try:
        status = init_db_table(DB_NAME, TABLE_NAME)
        if status == False:
            exit(1)
        # set up pin
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(DHT11_PIN, GPIO.IN)
        LED.Setup(LED_PIN, "OUT")
        GPIO.setup(MAG_PIN, GPIO.IN) # magnet switch

        # MQTT: buzzer is subscriber, sensor_db_main is publisher
        # MQTT connect 
        TopicServerIP = "localhost"
        TopicServerPort = 1883

        mqttc = mqtt.Client(client_id="", protocol=mqtt.MQTTv5)
        mqttc.username_pw_set(username=MQTT_USER, password=MQTT_PASS)
        mqttc.connect(TopicServerIP, TopicServerPort)

        # set up basic info for INSERT by SQL
        id_name = "a001"
        device = "pi"

        # door open period of time
        door_open_start = None
        door_open_duration = 0.0
        # start write data into db
        while(True): # measure the temperature and humidity 
            # Read humidity and temperature from the DHT11 sensor
            humidity, temperature = get_temp_humi(sensor, pin=DHT11_PIN)
            # check door status
            door = GPIO.input(MAG_PIN) # 1 or 0
            if door: # HIGH, when open the door, magnet leave the switch
                LED.TurnOnLED(LED_PIN)
                print("\033[93mDoor is open.\033[0m")
                if door_open_start is None:
                    door_open_start = time()
        
                door_open_duration = time() - door_open_start
                print(f"\033[96mDoor was open for {door_open_duration:.2f} seconds.\033[0m")

            else: # LOW, when close the door, magnet is close to the switch
                LED.TurnOffLED(LED_PIN)
                print("\033[95mDoor is closed.\033[0m")
                if door_open_start is not None:
                    # initialize them
                    door_open_duration = 0.0
                    door_open_start = None
            # door is open too long, and use the buzzer to send the warning
            if door_open_duration > DOOR_OPEN_TH:
                # buzzer_beep(beep_time=0.2, interval=0.2, count=10)
                mqttc.publish(MQTT_TOPIC_NAME, "on") # publish event to buzzer subscriber
                print("\033[91mDoor is open for too long.\033[0m")
                buzz = True
            else:
                mqttc.publish(MQTT_TOPIC_NAME, "off") # publish event to buzzer subscriber
                buzz = False
            # Get the current time in %Y-%m-%d %H:%M:%S format
            data_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            insert_data(id_name, device, data_time, temperature, humidity, buz=buzz, Door=(door==1), Duration=door_open_duration)
            print("-------------------------------------------------------------------------------------------")
            sleep(1)
            
    except mysql.connector.Error as err:
        print(f"❌ MySQL Error: {err}")
        
    except KeyboardInterrupt: # allow press ctrl+c to interrupt the process
        mqttc.publish(MQTT_TOPIC_NAME, "off")
        GPIO.cleanup() # clear pin info
        print("Finish uploading the data into DB")
        
    finally:
        # Close cursor and connection
        if 'cur' in locals():
            cur.close()
        if 'cnx' in locals() and cnx.is_connected():
            cnx.close()