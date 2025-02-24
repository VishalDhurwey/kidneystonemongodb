from flask import Flask, request, jsonify, send_file
import cv2
import torch
import os
import uuid
import numpy as np
from flask_cors import CORS
from ultralytics import YOLO
from pymongo import MongoClient
from bson.objectid import ObjectId
from werkzeug.utils import secure_filename
import gc
import google.generativeai as genai

# MongoDB Connection
MONGO_URI = os.environ.get("MONGO_URI")  # Get the connection string from Render environment variables
client = MongoClient(MONGO_URI)
db = client["your_database_name"]  # Replace with your actual database name
uploads_collection = db["uploads"]
results_collection = db["results"]

# Constants
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
DPI = 300  # Define the DPI of the image for mm conversion

# Load YOLO Model
MODEL_PATH = 'last (1).pt'
try:
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

# Flask Setup
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
LABELS_FOLDER = 'labels'

for folder in [UPLOAD_FOLDER, RESULT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_old_files(directory, max_files=100):
    """Remove old files if directory has more than max_files"""
    files = os.listdir(directory)
    if len(files) > max_files:
        for file in sorted(files, key=lambda x: os.path.getctime(os.path.join(directory, x)))[:-max_files]:
            try:
                os.remove(os.path.join(directory, file))
            except Exception as e:
                print(f"Error cleaning up file {file}: {e}")

def pixels_to_mm(pixels, dpi):
    return round((pixels / dpi) * 25.4, 2)  # Convert pixels to mm

def predict_image(image_path=None, image=None, conf=0.3, dpi=300):
    """
    Run inference on an image and return stone locations and processed image
    Args:
        image_path: Path to image file (optional if image is provided)
        image: OpenCV image array (optional if image_path is provided)
        conf: Confidence threshold for detections
        dpi: DPI of the image for size calculations
    Returns:
        Tuple of (processed_image, detected_stones)
    """
    try:
        if image is None and image_path is not None:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"❌ Image not found at {image_path}")
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("❌ Failed to load the image. Check the file format and path.")
        elif image is None:
            raise ValueError("❌ Either image_path or image must be provided")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_height, image_width = image.shape[:2]
        
        # Run inference
        results = model(image)
        
        detected_stones = []
        
        # Process detections
        for result in results:
            for box in result.boxes:
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                conf_score = float(box.conf[0])
                
                if conf_score > conf:
                    # Calculate center of the bounding box
                    center_x = (x_min + x_max) / 2
                    center_y = (y_min + y_max) / 2
                    
                    # Calculate size in millimeters
                    width_mm = pixels_to_mm(x_max - x_min, dpi)
                    height_mm = pixels_to_mm(y_max - y_min, dpi)
                    size_mm = round((width_mm + height_mm) / 2, 2)  # Average of width and height
                    
                    # Determine the relative location
                    if center_x < image_width / 2 and center_y < image_height / 2:
                        location = "Top-left"
                    elif center_x >= image_width / 2 and center_y < image_height / 2:
                        location = "Top-right"
                    elif center_x < image_width / 2 and center_y >= image_height / 2:
                        location = "Bottom-left"
                    else:
                        location = "Bottom-right"

                    # Save detected stone info
                    detected_stones.append({
                        "confidence": float(conf_score),
                        "coordinates": [int(x_min), int(y_min), int(x_max), int(y_max)],
                        "center": [float(center_x), float(center_y)],
                        "location": location,
                        "size_mm": size_mm,
                        "dimensions_mm": {
                            "width": round(width_mm, 2),
                            "height": round(height_mm, 2)
                        }
                    })
                    
                    # Draw bounding box
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                    stone_number = len(detected_stones)
                    label = f"Stone {stone_number}: {location} ({size_mm}mm)"
                    cv2.putText(image, label, (x_min, y_min - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Print detected stones
        if detected_stones:
            print(f"✅ Number of Kidney Stones Detected: {len(detected_stones)}")
            for i, stone in enumerate(detected_stones, 1):
                print(f"  {i}. Location: {stone['location']}, Size: {stone['size_mm']}mm, "
                      f"Coordinates: {stone['coordinates']}")
        else:
            print("❌ No kidney stones detected above the confidence threshold.")

        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return image, detected_stones

    except Exception as e:
        print(f"❌ Error in predict_image: {str(e)}")
        raise

# Set your Gemini API Key
genai.configure(api_key="AIzaSyAx9pUTUhIRVWQjIMOsR6_oxl8vBkXLXOg")

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message')

    # Construct a better prompt for shorter, polite responses
    prompt = f"""
You are a polite kidney health assistant.
Answer the user's question briefly in 1 short and clear sentence.
Keep the tone friendly and supportive.

User: {user_message}
Assistant:
"""


    # Call Gemini API (assuming you're using Google Generative AI API)
    import google.generativeai as genai

    genai.configure(api_key="AIzaSyAx9pUTUhIRVWQjIMOsR6_oxl8vBkXLXOg")
    model = genai.GenerativeModel('gemini-pro')

    response = model.generate_content(prompt)
    answer = response.text.strip()

    return jsonify({"response": answer})


@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file uploaded'}), 400

        image_file = request.files['image']

        if image_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if not allowed_file(image_file.filename):
            return jsonify({'error': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

        # Secure the filename and generate a unique name
        image_filename = f"{uuid.uuid4().hex}_{secure_filename(image_file.filename)}"

        # Read image file as binary
        image_binary = image_file.read()

        # Store uploaded image in MongoDB
        upload_result = uploads_collection.insert_one({
            "filename": secure_filename(image_filename),  # Add secure_filename here
            "image_data": image_binary
        })

        # Run detection (existing code)
        pred_img, detections = predict_image(image=cv2.imdecode(np.frombuffer(image_binary, np.uint8), cv2.IMREAD_COLOR), dpi=DPI)

        # Encode processed image to binary
        _, pred_img_encoded = cv2.imencode('.jpg', cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR))

        # Store results in MongoDB
        result_id = results_collection.insert_one({
            "upload_id": upload_result.inserted_id,  # Link to the uploaded image
            "filename": f"pred_{image_filename}",
            "predicted_image_data": pred_img_encoded.tobytes(),
            "detections": detections
        }).inserted_id

        return jsonify({
            'status': 'success',
            'result_id': str(result_id),  # Send result ID to frontend
            'detections': detections,
            'message': f'Successfully processed image with {len(detections)} detections'
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500
# @app.route('/download/<filename>', methods=['GET'])
# def download_file(filename):
#     try:
#         return send_file(
#             os.path.join(RESULT_FOLDER, secure_filename(filename)), 
#             as_attachment=True
#         )
#     except Exception as e:
#         return jsonify({'error': f'File not found: {str(e)}'}), 404

@app.route('/result/<result_id>', methods=['GET'])
def get_result(result_id):
    try:
        result = results_collection.find_one({"_id": ObjectId(result_id)})
        if result:
            # Option 1: Serve image data directly (adjust Content-Type as needed)
            # return result["predicted_image_data"], 200, {'Content-Type': 'image/jpeg'}

            # Option 2: Return image as base64 for frontend to display
            from base64 import b64encode
            image_base64 = b64encode(result["predicted_image_data"]).decode('utf-8')
            return jsonify({'image_data': image_base64, 'detections': result['detections']})

        else:
            return jsonify({'error': 'Result not found'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

    # port = int(os.environ.get('PORT', 5000))
    # app.run(host='0.0.0.0', port=port)
