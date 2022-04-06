from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
import cv2
import os
import json
import random
from matplotlib.image import imread

def open_waste_slot():

    """
        open the machine so that
        a user can enter the machine
    :return:
    """

    send_command_to_machine("open_waste_slot")
    return True


def close_waste_slot():
    """
    close the waste box for user safety
    :return:
    """

    send_command_to_machine("close_waste_slot")
    return True


def process_waste(waste_type):

    """
    move the good slot and shredd the waste
    :return:
    """

    move_container(waste_type)
    was_sucessful = shred_waste()

    return was_sucessful


def move_container(waste_type):

    BOTTLE_BOX = 0
    GLASS_BOX = 1
    command_name = "move_container"

    if waste_type == "bottle":
        send_command_to_machine(command_name, BOTTLE_BOX)
    elif waste_type == "glass":
        send_command_to_machine(command_name, GLASS_BOX)

    return True


def send_command_to_machine(command_name, value=None):

    """
    simulate command sending to raspberry pi
    do nothing to work even if the machine is not connected

    :param command_name:
    :param value:
    :return:
    """
    return True



def shred_waste():

    send_command_to_machine("shred_waste")

    return True


def take_trash_picture():

    """
        function simulating the picture taking
        inside the machine. 

        Call this function to ask the machine to 
        take picture of the trash

        return : np array of the picture
    """

    send_command_to_machine("take_picture")

    paths = os.listdir('camera')
    path = random.choice(paths)

    return imread(os.path.join("./camera", path))


def classify(img):

    # Read the key from an external (not tracked) resource
    with open('src/settings/appsettings.json') as json_file:
        json_data = json.load(json_file)
        prediction_key = json_data['prediction_key']

    ENDPOINT = "https://cgcustomvision-prediction.cognitiveservices.azure.com/"
    project_id = "ef16960a-94fe-468f-a42b-3e8ea8fc1483"
    publish_iteration_name = "Iteration1"

    prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
    predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)

    # with open(os.path.join (os.path.dirname(__file__), img_path), "rb") as image_contents:
    # with open(img, "rb") as image_contents:
    results = predictor.classify_image(
        project_id, publish_iteration_name, img)
    result = {
        "tag": "",
        "prob": 0
        }

    for prediction in results.predictions:
        if prediction.probability > result["prob"]:
            result["tag"] = prediction.tag_name
            result["prob"] = prediction.probability
    result["prob"] *= 100
    print(result)

    return result

def take_picture():

    cam = cv2.VideoCapture(0)

    cv2.namedWindow("test")

    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()

    cv2.destroyAllWindows()
