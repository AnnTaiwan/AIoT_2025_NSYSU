import mysql.connector
from dotenv import load_dotenv
import os
from flask import Flask, render_template, request, jsonify
import random

# for web
app = Flask(__name__)
# get the accessinfo for accessing my DB
load_dotenv()

host = os.getenv("MYSQL_HOST")
port = int(os.getenv("MYSQL_PORT"))
user = os.getenv("MYSQL_USER")
password = os.getenv("MYSQL_PASS")
DB_NAME = os.getenv("MYSQL_DB")
TABLE_NAME = os.getenv("TABLE_NAME")

from datetime import datetime, timedelta

def generate_random_sensor_data(n=10):
    data = []
    for _ in range(n):
        # id, device 都用固定值對應a001的格式
        id_val = 'a001'
        device_val = 'pi'
        # 隨機時間
        time_val = (datetime.now() - timedelta(minutes=random.randint(0, 10000))).strftime('%Y-%m-%d %H:%M:%S')
        temp_val = round(random.uniform(15, 30), 2)
        humid_val = round(random.uniform(30, 80), 2)
        weight_val = round(random.uniform(0, 10), 2)
        buzzer_val = random.choice([False, True])
        door_val = random.choice([False, True])
        duration_val = random.randint(0, 3600)

        # 對應row格式：ID, Device, Date, Temperature, Humid, Weight, Buzzer, Door, Duration
        row = (id_val, device_val, time_val, temp_val, humid_val, weight_val, buzzer_val, door_val, duration_val)
        data.append(row)
    return data

def get_connection():
    cnx = mysql.connector.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=DB_NAME
    )
    return cnx

def insert_data(id_name, device, time_data, frame:str, table_name=TABLE_NAME):
    # SQL query to insert data
    sql = f"""
        INSERT INTO {table_name} (ID, Device, Date, Frame)
        VALUES (%s, %s, %s, %s)
    """
    
    # Data to insert
    data = (id_name, device, time_data, frame) 
    cnx = get_connection()
    cur = cnx.cursor()

    # Execute the SQL query
    cur.execute(sql, data)

    # Commit changes
    cnx.commit()
    print("Data \033[94minserted\033[0m successfully:", data)
    
def delete_data_by_ID(id_name="a002"):
    # SQL query to delete data
    sql = f"DELETE FROM {TABLE_NAME} WHERE ID = %s"
    
    # Data to insert
    data = (id_name,) 
    cnx = get_connection()
    cur = cnx.cursor()

    # Execute the SQL query
    cur.execute(sql, data)

    # Commit changes
    cnx.commit()
    print(f"Data \033[94mdeleted\033[0m successfully, ID: {data[0]}")

def select_all(table_name=TABLE_NAME):
    # SQL query to delete data
    # select all except the frame(too big)
    sql = "SELECT ID, Device, Date, Temperature, Humid,	Buzzer, Door, Duration, Frame FROM " + table_name
    cnx = get_connection()
    cur = cnx.cursor()
    # Execute the SQL query
    cur.execute(sql)
    print("List all data in table now:")
    # SELECT  no need to commit update to db
    rows = cur.fetchall()
    # for row in rows: # each line is a tuple and one data
    #     print(row)
    return rows
def select_recent_sensor(table_name=TABLE_NAME):
    sql = f"""
        SELECT ID, Device, Date, Temperature, Humid, Buzzer, Door, Duration
        FROM {table_name}
        WHERE ID = 'a001'
        ORDER BY Date DESC
        LIMIT 50
    """
    cnx = get_connection()
    cur = cnx.cursor()

    cur.execute(sql)
    rows = cur.fetchall()
    return rows

def select_recent_frame(table_name=TABLE_NAME):
    sql = f"""
        SELECT ID, Device, Date, Frame, Action
        FROM {table_name}
        WHERE ID = 'a002'
        ORDER BY Date DESC
    """
    cnx = get_connection()
    cur = cnx.cursor()

    cur.execute(sql)
    rows = cur.fetchall()
    return rows
def select_log(table_name=TABLE_NAME): # get log info
    sql = f"""
        SELECT Log
        FROM {table_name}
        WHERE ID = 'a003'
    """
    cnx = get_connection()
    cur = cnx.cursor()

    cur.execute(sql)
    rows = cur.fetchall()
    return rows
def select_ID(table_name=TABLE_NAME): # get all the ID
    # SQL query to delete data
    sql = "SELECT ID FROM " + table_name
    cnx = get_connection()
    cur = cnx.cursor()
    
    # Execute the SQL query
    cur.execute(sql)
    print("List all IDs in table now:")
    # SELECT  no need to commit update to db
    rows = cur.fetchall()
    return rows

def connect_db_table(table_name=TABLE_NAME):
    try:
        cnx = get_connection()
        if cnx.is_connected():
            print(f"\033[92m✅ Connected to MySQL database {DB_NAME}\033[0m")
        else:
            print(f"\033[91m❌ Failed to connect to MySQL database {DB_NAME}\033[0m")
            return False

        # Test a query
        results = select_all(table_name)
        return True

    except mysql.connector.Error as err:
        print(f"\033[91mMySQL Error: {err}\033[0m")
        return False
################## web ####################################
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/show_images', methods=['POST'])
def show_images():
    rows_frame = select_recent_frame()
    dates = [row[2] for row in rows_frame]
    image_paths = [row[3] for row in rows_frame]
    actions = [bool(row[4]) for row in rows_frame]  # 確保是布林值
    return jsonify({
        'output_date': dates,
        'output_frame': image_paths,
        'output_action': actions
    })

@app.route('/show_sensors', methods=['POST'])
def show_sensors(): # return value in decreasing date
    rows_sensor = select_recent_sensor()  # 假設會回傳多筆資料，每筆是tuple/list
    # rows_sensor = generate_random_sensor_data(50)

    # 分別抽取欄位，轉換成list
    Date = [row[2] for row in rows_sensor]
    Temperature = [row[3] for row in rows_sensor]
    Humid = [row[4] for row in rows_sensor]
    Buzzer = [bool(row[5]) for row in rows_sensor]
    Door = [bool(row[6]) for row in rows_sensor]
    Duration = [row[7] for row in rows_sensor]
    tem_abnoraml = [not ((tem < 30) and (tem > 20)) for tem in Temperature]
    hum_abnoraml = [not ((hum < 60) and (hum > 40)) for hum in Humid]
    return jsonify({
        'Date': Date,
        'Temperature': Temperature,
        'Humid': Humid,
        'Buzzer': Buzzer,
        'Door': Door,
        'Duration': Duration,
        'tem_abnormal': tem_abnoraml,
        'hum_abnormal': hum_abnoraml
    })

@app.route('/update_image_status', methods=['POST'])
def update_image_status(table_name=TABLE_NAME):
    data = request.get_json()
    image_path = data['image_path']
    action = data['action']
    print(f"Get new request to update action: {image_path}, {action}")

    try:
        cnx = get_connection()
        cur = cnx.cursor()
        # SQL query to update Action value for a specific Frame
        sql = f"""
            UPDATE {table_name}
            SET Action = %s, Date = %s
            WHERE Frame = %s
        """
        data_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Data to update
        data = (action == "TAKE", data_time, image_path)

        # Execute the SQL query
        cur.execute(sql, data)

        # Commit changes
        cnx.commit()
        print(f"Action value updated for frame: {image_path} to {action == 'TAKE'}")
        result = cur.fetchone()
        return jsonify({"success": True})
    except Exception as e:
        print(e)
        return jsonify({"success": False})
    
@app.route('/show_logs', methods=['POST'])
def show_logs():
    rows_log = select_log()
    
    return jsonify({
        'output_log': rows_log
    })

if __name__ == "__main__":
    # try to connect to the db
    try:
        status = connect_db_table()
        if not status: # false, some connected errors happen
            exit(1)
        # rows = select_recent_sensor()
        # for row in rows:
        #     print(row)
        # start to run the web
        app.run(host='127.0.0.1', port=5000, debug=True)
    except KeyboardInterrupt: # allow press ctrl+c to interrupt the process
        print("Finish interacting data with DB, and quit the web server.")
    except Exception as err:
        print("Exception ERROR:", err)