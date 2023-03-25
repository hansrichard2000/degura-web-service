import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
import core.utils as utils
from core.config import cfg
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import time
from flask import Flask, request, Response, jsonify, send_from_directory, abort, render_template
import os
import json
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image

framework = 'tf'
weights_path = './checkpoints/custom-416'
size = 416
tiny = False
model = 'yolov4'
output_path = './static/detections/'
iou = 0.45
score = 0.25

class Flag:
    tiny = tiny
    model = model

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
FLAGS = Flag
STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
input_size = size

# load model
if framework == 'tflite':
    interpreter = tf.lite.Interpreter(model_path=weights_path)
else:
    saved_model_loaded = tf.saved_model.load(weights_path, tags=[tag_constants.SERVING])

# Initialize Flask application
app = Flask(__name__)
print("loaded")

# Home Routing
@app.route('/')
def home():
    return render_template('./index.html')

# API that returns JSON with classes found in images
@app.route('/detections/by-image-files', methods=['POST'])
def get_detections_by_image_files():
    images = request.files.getlist("images")
    image_path_list = []
    for image in images:
        image_name = image.filename
        image_path_list.append("./temp/" + image_name)
        image.save(os.path.join(os.getcwd(), "temp/", image_name))

    # create list of final response
    response = []

    # loop through images in list and run Yolov4 model on each
    for count, image_path in enumerate(image_path_list):
        # create list of responses for current image
        responses = []
        try:
            original_image = cv2.imread(image_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            image_data = cv2.resize(original_image, (input_size, input_size))
            image_data = image_data / 255.
        except cv2.error:
            # remove temporary images
            for name in image_path_list:
                os.remove(name)
            abort(404, "it is not an image file or image file is an unsupported format. try jpg or png")
        except Exception as e:
            # remove temporary images
            for name in image_path_list:
                os.remove(name)
            print(e.__class__)
            print(e)
            abort(500)
        
        #Input processed image file for detection
        images_data = []
        for i in range(1):
            images_data.append(image_data)
        images_data = np.asarray(images_data).astype(np.float32)

        if framework == 'tflite':
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            print(input_details)
            print(output_details)
            interpreter.set_tensor(input_details[0]['index'], images_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            if model == 'yolov3' and tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            t1 = time.time()
            infer = saved_model_loaded.signatures['serving_default']
            batch_data = tf.constant(images_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]
            t2 = time.time()
            print('time: {}'.format(t2 - t1))

        t1 = time.time()
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=iou,
            score_threshold=score
        )
        t2 = time.time()
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)
        print('time: {}'.format(t2 - t1))
        for i in range(valid_detections[0]):
            print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                        np.array(scores[0][i]),
                                        np.array(boxes[0][i])))
            responses.append({
                "class": class_names[int(classes[0][i])],
                "confidence": float("{0:.2f}".format(np.array(scores[0][i]) * 100)),
                "box": np.array(boxes[0][i]).tolist()
            })
        response.append({
            "image": image_path_list[count][7:],
            "detections": responses
        })
        
        # Produce bounding box coordinate
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        # allowed_classes = list(class_names.values())

        # custom allowed classes (uncomment line below to allow detections for only people)
        allowed_classes = ['kepala']

        # draw bounding box
        image = utils.draw_bbox(original_image, pred_bbox, allowed_classes=allowed_classes)

        image = Image.fromarray(image.astype(np.uint8))

        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        cv2.imwrite(output_path + 'detection' + str(count) + '.png', image)

    # remove temporary images
    for name in image_path_list:
        os.remove(name)
    try:
        return Response(response=json.dumps({"response": response}), mimetype="application/json")
    except FileNotFoundError:
        abort(404)
    
# API that returns image with detections on it
@app.route('/image/by-image-file', methods=['POST'])
def get_image_by_image_file():
    image = request.files["images"]
    image_filename = image.filename
    image_path = "./temp/" + image.filename
    image.save(os.path.join(os.getcwd(), image_path[2:]))

    try:
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        image_data = cv2.resize(original_image, (input_size, input_size))
        image_data = image_data / 255.
    except cv2.error:
        # remove temporary image
        os.remove(image_path)
        abort(404, "it is not an image file or image file is an unsupported format. try jpg or png")
    except Exception as e:
        # remove temporary image
        os.remove(image_path)
        print(e.__class__)
        print(e)
        abort(500)

    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)

    if framework == 'tflite':
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
        interpreter.set_tensor(input_details[0]['index'], images_data)
        interpreter.invoke()
        pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
        if model == 'yolov3' and tiny == True:
            boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                            input_shape=tf.constant([input_size, input_size]))
        else:
            boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                            input_shape=tf.constant([input_size, input_size]))
    else:
        t1 = time.time()
        infer = saved_model_loaded.signatures['serving_default']
        batch_data = tf.constant(images_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]
        t2 = time.time()
        print('time: {}'.format(t2 - t1))

    t1 = time.time()
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=iou,
        score_threshold=score
    )
    t2 = time.time()
    class_names = utils.read_class_names(cfg.YOLO.CLASSES)
    print('time: {}'.format(t2 - t1))
    for i in range(valid_detections[0]):
        print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                    np.array(scores[0][i]),
                                    np.array(boxes[0][i])))

    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

    # read in all class names from config
    class_names = utils.read_class_names(cfg.YOLO.CLASSES)

    # by default allow all classes in .names file
    # allowed_classes = list(class_names.values())

    # custom allowed classes (uncomment line below to allow detections for only people)
    allowed_classes = ['kepala']

    image = utils.draw_bbox(original_image, pred_bbox, allowed_classes=allowed_classes)

    image = Image.fromarray(image.astype(np.uint8))

    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    # Download file detected.png and save it to output folder
    cv2.imwrite(output_path + image_filename[0:len(image_filename) - 4] + '.png', image)
    # cv2.imwrite(output_path + 'detection' + '.png', image)

    # prepare image for response
    _, img_encoded = cv2.imencode('.png', image)
    response = img_encoded.tobytes()
    print(type(response))

    # remove temporary image
    os.remove(image_path)
    # print(f"{image.filename}XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

    try:
        return Response(response=response, status=200, mimetype='image/png')
    except FileNotFoundError:
        abort(404)

@app.route('/classifier')
def index_view():
    return render_template('index.html')
#Allow files with extension png, jpg and jpeg
ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT
           
# Function to load and prepare the image in right shape
def read_image(filename):
    img = load_img(filename, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x
    
@app.route('/classifier/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename): #Checking file format
            filename = file.filename
            file_path = os.path.join('static/images', filename)
            file.save(file_path)
            img = read_image(file_path) #prepressing method
            class_prediction=model.predict(img) 
            classes_x=np.argmax(class_prediction,axis=1)
            if classes_x == 0:
              fruit = "Apple"
            elif classes_x == 1:
              fruit = "Banana"
            else:
              fruit = "Orange"
            #'fruit' , 'prob' . 'user_image' these names we have seen in predict.html.
            return render_template('predict.html', fruit = fruit,prob=class_prediction, user_image = file_path)
        else:
            return "Unable to read the file. Please check file extension"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050)