from ultralytics import YOLO
import cv2
import csv
import random

# --- Random Character Generation ---
def generate_random_char():
    random_number = random.randint(0, 99)
    return str(random_number).zfill(2)

'''
#Keep this line for reference.
This line is for set the confidence threshold for the 'person' class to 0.8.
'''
# confidence_threshold = 0.8  # Set custom threshold for the 'person' class

# load yolov8 model
model = YOLO('yolov9s.pt')

# --- Naming Setup ---
random_char = generate_random_char() # Generate random 2-digit character
# csv_file_path = 'centroid_data.csv'
csv_file_path = f'centroid_data_{random_char}.csv'
with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['x', 'y'])  # Write the header row
    
video_file_path = f'video_{random_char}.mp4'  # Naming the video file
# -----


'''
#Keep this line for reference.
Detect objects in a video and save the centroid coordinates of the detected objects to a CSV file.
'''
# # --- Naming Setup ---
# # base_file_name = 'centroid_data'  
# random_char = generate_random_char() # Generate random 2-digit character
# csv_file_path = f'centroid_data_{random_char}.csv'  # Naming the CSV file
# video_file_path = f'video_{random_char}.avi'  # Naming the video file
# # ---

#result = model.predict(source="./PGM-normal_cewek_sore_2.mp4", save=True, save_txt=True, classes=[0], stream=True)

# load video
video_path = 'D:\\@Dev\\finalPrj\\objectDetection1\\dataset\\480%Ks2-Abnormal Scene3.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties for output
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print(f"Video Width: {width}, Video Height: {height}")

# Create video writer object
fourcc = cv2.VideoWriter_fourcc(*'avc1')
out = cv2.VideoWriter(video_file_path, fourcc, fps, (width, height))

# Main processing loop
while True:
    ret, frame = cap.read()

    # Write frame to video
    if not ret:
        break  # End of video

    '''
    # Keep this line for reference.
    Resize frame optional, or i guess no need tho.
    '''
    # Resize frame (optional)
    # frame = cv2.resize(frame, (960, 540))

    # Object detection
    results = model.predict(frame, conf=0.8, classes=[0], save=True, save_txt=True)

    # --- Save Centroid to CSV ---
    for r in results:  
        for i, box in enumerate(r.boxes.xyxy):
            centroid_y = (box[1] + box[3]) / 2  # Use YOLO's y1 and y2 
            centroid_x = (box[0] + box[2]) / 2  # Use YOLO's x1 and x2

            # Draw bounding box and add label + confidence score
            label = f'{r.names[int(r.boxes.cls[i])]} {r.boxes.conf[i].item():.2f}'
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            '''
            # Keep this line for reference.
            This line is for adding label and confidence score to the bounding box.But i think line above is enough.
            '''
            # cv2.putText(frame, label, (int(box[0]), int(box[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

            # Print and/or draw on the frame
            print(f"Centroid Coordinates: ({centroid_x}, {centroid_y})\n")
            print(box, "\n")  # Print YOLO bounding box coordinates
            cv2.circle(frame, (int(centroid_x), int(centroid_y)), radius=2, color=(0, 0, 255), thickness=3) # Draw centroid
            
            # Save to CSV
            with open(csv_file_path, 'a', newline='') as csvfile:  # 'a' for append
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([centroid_x, centroid_y])
            
            # with open(csv_file_path, 'w', newline='') as csvfile:
            #     csv_writer = csv.writer(csvfile)
            #     csv_writer.writerow(['x', 'y'])  # Header
            
            # Save to CSV
            # with open(csv_file_path, 'a', newline='') as csvfile: 
            #    csv_writer = csv.writer(csvfile)

    # Write frame to video
    for result in results:
        frame_ = result.plot()  # Get frame with bounding boxes
        out.write(frame_)
        cv2.imshow('OBJ Detection', frame_)  # Display frame

    # Exit condition
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()  # Release the video writer
cv2.destroyAllWindows()