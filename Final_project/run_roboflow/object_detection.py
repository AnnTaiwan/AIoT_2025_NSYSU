import mysql.connector
from datetime import datetime

from inference import get_model
import supervision as sv
import cv2
import matplotlib.pyplot  as plt

from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("ROBOFLOW_API_KEY")
print("api_key", api_key)

# load a pre-trained yolov8n model
model = get_model(model_id="fruits-by-yolo/1")
Class_Names = model.class_names
colors = {
    'Apple':       (255, 99, 132),   # 粉紅紅
    'Banana':      (255, 205, 86),   # 明亮黃
    'Grapes':      (153, 102, 255),  # 紫羅蘭
    'Kiwi':        (75, 192, 192),   # 青綠藍
    'Mango':       (255, 159, 64),   # 橘黃
    'Orange':      (255, 159, 64),   # 橘黃 (跟芒果同色可視情況調)
    'Pineapple':   (255, 205, 86),   # 明亮黃
    'Sugerapple':  (54, 162, 235),   # 清爽藍
    'Watermelon':  (255, 99, 132)    # 粉紅紅 (跟蘋果同色)
}
# DB const variable
TABLE_NAME = "smart_refrigerator"
DB_NAME = "anndb"
####################################################
def infer_frame(frame):
    # run inference on the image
    results = model.infer(frame)[0]

    # load results into supervision Detections API
    detections = sv.Detections.from_inference(results)

    # create annotators
    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # annotate the image with inference results
    annotated_image = bounding_box_annotator.annotate(
        scene=frame, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections)

    bbox_coords = detections.xyxy       # shape: (N, 4) each row: [x_min, y_min, x_max, y_max]
    class_ids = detections.class_id     # shape: (N,)
    confidences = detections.confidence # shape: (N,)

    return annotated_image, bbox_coords, class_ids, confidences



def insert_data(id_name, device, time_data, frame:str, table_name=TABLE_NAME):
    # SQL query to insert data
    sql = f"""
        INSERT INTO {table_name} (ID, Device, Date, Frame)
        VALUES (%s, %s, %s, %s)
    """
    
    # Data to insert
    data = (id_name, device, time_data, frame) 

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

    # Execute the SQL query
    cur.execute(sql, data)

    # Commit changes
    cnx.commit()
    print(f"Data \033[94mdeleted\033[0m successfully, ID: {data[0]}")

def select_all(table_name=TABLE_NAME):
    # SQL query to delete data
    # select all except the frame(too big)
    sql = "SELECT ID, Device, Date, Temperature, Humid, Weight,	Buzzer, Door, Duration FROM " + table_name
    
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
        host="127.0.0.1",  # Your laptop's IP address
        port=3306,           # MySQL default port
        user="usr002",       # Your MySQL username
        password="aiot0000", # Your MySQL password
        database=db_name  # Target database
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
    
def object_detect(id_name, device, source):
    cap = cv2.VideoCapture(source)
    HEIGHT = 360
    WIDTH = 640
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    DETECT = False
    STATUS = None
    pre_xmin = WIDTH

    count = 0
    SHOT_OR_NOT = False
    Fruits = list()
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot receive frame")
            break
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        frame_height, frame_width = frame.shape[:2]
        MIDDLE_X = frame_width // 2

        annotated_frame, bboxes, class_ids, confidences = infer_frame(frame)
        for i, box in enumerate(bboxes):
            x_min, y_min, x_max, y_max = box
            if x_min < MIDDLE_X < x_max:
                DETECT = True
                # print("Object crosses the middle line!", box)
                mid = (x_max + x_min) / 2
                if not SHOT_OR_NOT and abs(mid - MIDDLE_X) <= 3:
                    # only crop the object
                    cropped_object = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
                    filepath = "screenshots/" + Class_Names[class_ids[i]] + "_" + str(count) + ".jpg"
                    cv2.imwrite(filepath, cropped_object)
                    # insert path into db
                    # Get the current time in %Y-%m-%d %H:%M:%S format
                    data_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    insert_data(id_name, device, data_time, filepath)
                    # print the info
                    print(f"{count}: {data_time} Take the screenshot at {box}, and saved in {filepath}")
                    # record the fruits
                    Fruits.append(Class_Names[class_ids[i]] + "_" + str(count))
                    count += 1
                    # make sure next time, don't take same object's screenshot again
                    SHOT_OR_NOT = True
            else:
                DETECT = False
                SHOT_OR_NOT = False    
            if DETECT:
                if x_min < pre_xmin: # move to left, put the fruits
                    STATUS = "PUT"
                    pre_xmin = x_min # update the previous position
                elif x_min > pre_xmin: # move to right, take the fruits
                    STATUS = "TAKE"
                    pre_xmin = x_min # update the previous position
                else: # not move
                    STATUS = None
        # print status
        font = cv2.FONT_HERSHEY_SIMPLEX
        start_x, start_y = 500, 30  # 起始座標
        line_spacing = 30  # 每行間距

        # add transparent window background
        overlay = frame.copy()
        cv2.rectangle(overlay, (start_x, 10), (640, start_y + line_spacing * len(Fruits)), (0, 0, 0), -1)
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # 標題
        cv2.putText(frame, 'Records:', (start_x, 25), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # 每個水果名稱分行顯示
        for i, fruit in enumerate(Fruits):
            y = start_y + line_spacing * (i + 1)
            color = colors[fruit.split('_')[0]]
            cv2.putText(frame, f'{fruit} {STATUS}', (start_x, y), font, 0.5, color, 1, cv2.LINE_AA)
        

        cv2.line(annotated_frame, (MIDDLE_X, 0), (MIDDLE_X, frame_height), (255, 0, 0), 3)
        cv2.imshow('DetectFruits', annotated_frame)

        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()                           # 所有作業都完成後，釋放資源
    cv2.destroyAllWindows()                 # 結束所有視窗   

if __name__ == "__main__":
    try:
        status = init_db_table(DB_NAME, TABLE_NAME)
        # set up basic info
        id_name = "a002"
        device = "mypc"
        # object_detect(id_name, device, "videos/apple_flow.mp4") # main program to detect objects and write it into db
        # object_detect(id_name, device, 0) # main program to detect objects and write it into db
        url = "http://172.20.10.3:4747/video"
        object_detect(id_name, device, url) # main program to detect objects and write it into db
    except mysql.connector.Error as err:
        print(f"\033[91mMySQL Error: {err}\033[0m")
    except KeyboardInterrupt: # allow press ctrl+c to interrupt the process
        print("Finish uploading the data into DB")
    finally:
        # Close cursor and connection
        if 'cur' in locals():
            cur.close()
        if 'cnx' in locals() and cnx.is_connected():
            cnx.close()
