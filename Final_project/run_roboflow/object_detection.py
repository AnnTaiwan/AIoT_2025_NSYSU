import mysql.connector
from datetime import datetime
import numpy as np

from inference import get_model
import supervision as sv
import cv2
import matplotlib.pyplot  as plt

from dotenv import load_dotenv
import os
from time import sleep

import re

# get the accessinfo for accessing my DB
load_dotenv()

api_key = os.getenv("ROBOFLOW_API_KEY")
host = os.getenv("MYSQL_HOST")
port = int(os.getenv("MYSQL_PORT"))
user = os.getenv("MYSQL_USER")
password = os.getenv("MYSQL_PASS")
DB_NAME = os.getenv("MYSQL_DB")

# load a pre-trained yolov8n model
model = get_model(model_id="fruits-by-yolo/1")
Class_Names = model.class_names
# bbox color
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
TABLE_NAME = os.getenv("TABLE_NAME")
# log's column device and id for inserting data
log_id_name = "a003"
log_device = "mypc"
############# DB ################
def get_connection():
    cnx = mysql.connector.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=DB_NAME
    )
    return cnx

def insert_data(id_name, device, time_data, frame:str, action:bool, table_name=TABLE_NAME):
    cnx = get_connection()
    cur = cnx.cursor()
    # SQL query to insert data
    sql = f"""
        INSERT INTO {table_name} (ID, Device, Date, Frame, Action)
        VALUES (%s, %s, %s, %s, %s)
    """
    
    # Data to insert
    data = (id_name, device, time_data, frame, action) 

    # Execute the SQL query
    cur.execute(sql, data)

    # Commit changes
    cnx.commit()
    print("Data \033[94minserted\033[0m successfully:", data)

def insert_log(id_name, device, time_data, log:str, table_name=TABLE_NAME):
    cnx = get_connection()
    cur = cnx.cursor()
    # SQL query to insert data
    sql = f"""
        INSERT INTO {table_name} (ID, Device, Date, Log)
        VALUES (%s, %s, %s, %s)
    """
    
    # Data to insert
    data = (id_name, device, time_data, log) 

    # Execute the SQL query
    cur.execute(sql, data)

    # Commit changes
    cnx.commit()
    print("Data \033[94minserted\033[0m successfully:", data)

def update_action_by_frame(frame_path: str, new_action: bool, table_name=TABLE_NAME):
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
    data = (new_action, data_time, frame_path)

    # Execute the SQL query
    cur.execute(sql, data)

    # Commit changes
    cnx.commit()
    print(f"Action value updated for frame: {frame_path} to {new_action}")
    result = cur.fetchone()
    # print("After update:", result)

def check_action_by_frame(frame_path:str, table_name=TABLE_NAME):
    cnx = get_connection()
    cur = cnx.cursor()
    sql = f"""
        SELECT Action FROM {table_name}
        WHERE Frame = %s
    """
    # Data to update
    data = (frame_path,)# must add coma, make it be tuple
 
    # Execute the SQL query
    cur.execute(sql, data)

    rows = cur.fetchall()
    # Check if any rows were found
    if rows:
        # Return the first Action value
        return rows[0][0]
    else:
        # If no record found, return None or you can raise an exception
        print(f"[WARN] No record found for frame: {frame_path}")
        return None
def delete_data_by_ID(id_name="a002"):
    cnx = get_connection()
    cur = cnx.cursor()
    # SQL query to delete data
    sql = f"DELETE FROM {TABLE_NAME} WHERE ID = %s"
    
    # Data to insert
    data = (id_name,) 

    # Execute the SQL query
    cur.execute(sql, data)

    # Commit changes
    cnx.commit()
    print(f"Data \033[94mdeleted\033[0m successfully, ID: {data[0]}")

def select_door(table_name=TABLE_NAME):
    cnx = get_connection()
    cur = cnx.cursor()
    # SQL query to delete data
    # select all except the frame(too big)
    sql = f"""
        SELECT Door
        FROM {table_name}
        WHERE ID = 'a001'
        ORDER BY Date DESC
        LIMIT 1
    """
    
    # Execute the SQL query
    cur.execute(sql)
    # SELECT  no need to commit update to db
    rows = cur.fetchall()
    return rows

def select_frame_action(table_name=TABLE_NAME):
    cnx = get_connection()
    cur = cnx.cursor()
    # SQL query to delete data
    # select all except the frame(too big)
    sql = f"""
        SELECT Frame, Action
        FROM {table_name}
        WHERE ID = 'a002'
    """
    
    # Execute the SQL query
    cur.execute(sql)
    # SELECT  no need to commit update to db
    rows = cur.fetchall()
    return rows


def select_all(table_name=TABLE_NAME):
    cnx = get_connection()
    cur = cnx.cursor()
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
    cnx = get_connection()
    cur = cnx.cursor()
    # SQL query to delete data
    sql = "SELECT ID FROM " + table_name
    
    # Execute the SQL query
    cur.execute(sql)
    print("List all IDs in table now:")
    # SELECT  no need to commit update to db
    rows = cur.fetchall()
    return rows
        
def init_db_table(table_name=TABLE_NAME):
    # connect db
    # Connect to MySQL server on your laptop
    cnx = mysql.connector.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=DB_NAME
    )
     # Create a cursor object
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
############### Object detection ##################################
'''
def see_if_match(src_dir: str, new_image, orb, bf):
    print("-"*80)
    print("[START] One object is taken out. Start to see if it matches with the db data...")
    print(f"\n[INFO] Checking directory: {src_dir}")

    files = os.listdir(src_dir)
    print(f"[INFO] Total files found: {len(files)}")

    if len(files) == 0:
        print("[INFO] Directory is empty, skip matching.")
        return None

    sim = dict()  # Dictionary to store similarity scores

    # Compute keypoints and descriptors for new image
    kp2, des2 = orb.detectAndCompute(new_image, None)
    if des2 is None:
        print("[WARN] No descriptors found in new image. Aborting match.")
        return None
    else:
        print(f"[INFO] New image keypoints: {len(kp2)}")

    for file in files:
        file_path = os.path.join(src_dir, file)
        print(f"\n[INFO] Processing file: {file_path}")
        action = check_action_by_frame(file_path)
        if action is not None:
            print(f"[REMIND] This file's currently action is {'TAKE' if action else 'PUT'}.")
            if action: # this object is already taken, so skip it
                print("[REMIND-2] This object is already taken, so skip it.")
                continue
        else:
            print("[WARN] This filepath isn't exist in DB, so skip it.")
            continue
        # Read old image
        old_image = cv2.imread(file_path)
        if old_image is None:
            print(f"[WARN] Failed to read image: {file_path}. Skipping.")
            continue

        # Compute keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(old_image, None)
        if des1 is None:
            print(f"[WARN] No descriptors found in {file}. Skipping.")
            sim[file_path] = 0
            continue
        else:
            print(f"[INFO] Old image keypoints: {len(kp1)}")

        # Match descriptors
        matches = bf.match(des1, des2)
        print(f"[INFO] Total matches between {file} and new image: {len(matches)}")

        if len(matches) == 0:
            print(f"[WARN] No matches for {file}.")
            sim[file_path] = 0
            continue

        # Good matches (distance threshold)
        good_matches = [m for m in matches if m.distance < 50]
        print(f"[INFO] Good matches (<50 distance): {len(good_matches)}")

        # Similarity ratio
        similarity = len(good_matches) / len(matches)
        sim[file_path] = similarity

        print(f"[RESULT] {file} similarity ratio: {similarity:.2f}")

    # Find the file with the highest similarity over threshold
    best_score = -1
    match_file = None
    for path, score in sim.items():
        print(f"[INFO] Candidate: {path}, similarity: {score:.2f}")
        if score > best_score and score >= 0.5:
            best_score = score
            match_file = path

    if match_file:
        print(f"\n[FINAL] Best match: {match_file} with similarity {best_score:.2f}")
    else:
        print("\n[FINAL] No match found over threshold.")

    return match_file
'''
def see_if_match_type(object_label): # when taking out a fruit from refrigerator, to match with the db and update the action(PUT, TAKE)
    print("-" * 80)
    print(f"[START] One {object_label} is taken out.")
    print("[INFO] Start to see if it matches with the db data...")
    
    match_files = None
    rows = select_frame_action()  # return [(frame, action), ...]
    print(f"[INFO] Fetched {len(rows)} rows from database.")

    for idx, row in enumerate(rows):
        frame, action = row
        print(f"[CHECK] Row {idx+1}: frame = {frame}, action = {'TAKE' if action else 'PUT'}")

        if not action:
            if re.search(object_label, frame):
                print(f"  [SUCCESS] Match found: {frame}")
                match_files = frame
                return match_files
        else:
            print(f"  [SKIP] {frame} is already taken out.")

    print("[END] No match found.")
    return match_files


def init_cap(source): # init the camera
    cap = cv2.VideoCapture(source)
    HEIGHT = 360
    WIDTH = 640
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)   
    if not cap.isOpened():
        print("Cannot open camera")
        exit(1)
    return cap

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

def object_detect(cap, id_name, device):
    print("[!!] Start to do object detection...")
    HEIGHT = 360
    WIDTH = 640
    DETECT = False
    STATUS = None
    pre_xmin = WIDTH
    

    SHOT_OR_NOT = False
    Fruits = list() # detected fruits
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    ''' Not used!
    # for matching
    orb = cv2.ORB_create()  # Create ORB detector
    # Create Brute-Force matcher with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    '''
    # start to detect...
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
            # decide direction
            if x_min > pre_xmin: # move to right, put the fruits
                STATUS = "PUT"
                pre_xmin = x_min # update the previous position
            elif x_min < pre_xmin: # move to left, take the fruits
                STATUS = "TAKE"
                pre_xmin = x_min # update the previous position
            else: # not move
                STATUS = None

            if x_min < MIDDLE_X < x_max:
                DETECT = True
                # print("Object crosses the middle line!", box)
                mid = (x_max + x_min) / 2
                if not SHOT_OR_NOT and abs(mid - MIDDLE_X) < 8:
                    # only crop the object
                    cropped_object = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
                    # Get the current time in %Y-%m-%d %H:%M:%S format
                    data_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_data = f"{data_time} => {STATUS}: "
                    # if action is "TAKE", use SIFT to test if this object is same as one object in the db already, if so, change the action status in db
                    if STATUS == "TAKE":
                        '''
                        match_file = see_if_match("static/screenshots/", cropped_object, orb, bf)
                        '''
                        match_file = see_if_match_type(Class_Names[class_ids[i]])
                        if match_file == None: # didn't match, skip this detection
                            log_data += f"This object({Class_Names[class_ids[i]]}) didn't match any object in DB."
                            insert_log(log_id_name, log_device, data_time, log_data)
                            continue
                        else:
                            update_action_by_frame(match_file, STATUS == "TAKE") # change action(PUT) into (TAKE)
                            log_data += f"One object({match_file.split('/')[-1][:-4]}) is taken out."
                            insert_log(log_id_name, log_device, data_time, log_data)
                            SHOT_OR_NOT = True
                            continue
                    # save the screenshots
                    filepath = "static/screenshots/" + Class_Names[class_ids[i]] + "_" + str(count[Class_Names[class_ids[i]]]) + ".jpg"
                    cv2.imwrite(filepath, cropped_object)
                    # insert log
                    log_data += f"One object({filepath.split('/')[-1][:-4]}) is put."
                    insert_log(log_id_name, log_device, data_time, log_data)

                    # insert path into db
                    insert_data(id_name, device, data_time, filepath, STATUS == "TAKE")
                    # print the info
                    print(f"{sum(count.values())}: {data_time} Take the screenshot at {box}, and saved in {filepath}")
                    # record the fruits
                    Fruits.append(Class_Names[class_ids[i]] + "_" + str(count[Class_Names[class_ids[i]]]))
                    count[Class_Names[class_ids[i]]] += 1
                    # make sure next time, don't take same object's screenshot again
                    SHOT_OR_NOT = True
            else:
                DETECT = False
                SHOT_OR_NOT = False    

            
        # print status
        font = cv2.FONT_HERSHEY_SIMPLEX
        start_x, start_y = 500, 30  # 起始座標
        line_spacing = 20  # 每行間距

        # add transparent window background
        overlay = frame.copy()
        cv2.rectangle(overlay, (start_x, 10), (640, start_y + line_spacing * (len(Fruits) + 1)), (0, 0, 0), -1)
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # 標題
        cv2.putText(frame, 'Records:', (start_x, 25), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # 每個水果名稱分行顯示
        for i, fruit in enumerate(Fruits):
            y = start_y + line_spacing * (i + 1)
            color = colors[fruit.split('_')[0]]
            cv2.putText(frame, f'{fruit}', (start_x, y), font, 0.5, color, 1, cv2.LINE_AA)
        

        cv2.line(annotated_frame, (MIDDLE_X, 0), (MIDDLE_X, frame_height), (255, 0, 0), 3)
        cv2.imshow('DetectFruits', annotated_frame)

        if cv2.waitKey(1) == ord('q'):
            break
        
        doors = select_door()
        if len(doors) > 0 and doors[0][0] != True: # door is close
            break
def clear_directory(dir_path): # clear all screenshots
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

if __name__ == "__main__":
    cap = None
    try:
        status = init_db_table(TABLE_NAME)
        # clear previous screenshots
        clear_directory("static/screenshots/")
        # dict to record type of screenshots
        count = { cls: 0 for cls in Class_Names}
        # set up basic info
        id_name = "a002"
        device = "mypc"
        # choose source
        # cap = init_cap(0)
        # cap = init_cap(1)
        url = os.getenv("DROID_CAM")
        cap = init_cap(url)

        # see if door is open
        while True:
            doors = select_door()
            print("Door status:", "Open" if doors[0][0] else "Close")
            if len(doors) > 0 and doors[0][0] == True:
                object_detect(cap, id_name, device) # main program to detect objects and write it into db
            else: # just play every frame
                sleep(1)
    except mysql.connector.Error as err:
        print(f"\033[91mMySQL Error: {err}\033[0m")
    except KeyboardInterrupt: # allow press ctrl+c to interrupt the process
        cap.release()                           # 所有作業都完成後，釋放資源
        cv2.destroyAllWindows()                 # 結束所有視窗   
        print("Finish uploading the data into DB")
    except Exception as err:
        print(f"\033[91mError: {err}\033[0m")
    finally:
        cap.release()                           # 所有作業都完成後，釋放資源
        cv2.destroyAllWindows()                 # 結束所有視窗   
