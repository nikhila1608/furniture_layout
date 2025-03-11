from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import numpy as np
from keras.models import load_model
from keras.losses import MeanSquaredError
from keras.saving import register_keras_serializable
from keras.utils import custom_object_scope

# Register MSE function for compatibility
@register_keras_serializable()
def mse(y_true, y_pred):
    return MeanSquaredError()(y_true, y_pred)

# Load the model with custom object scope
MODEL_PATH = "C://Users//GVS Nikhila//OneDrive//Desktop//assignment1//furniture_layout//layout//furniture_layout_model.h5"

with custom_object_scope({'mse': mse}):
    model = load_model(MODEL_PATH)

# View to handle API requests
@csrf_exempt
def predict_layout(request):
    if request.method == 'POST':
        try:
            # Parse JSON input
            data = json.loads(request.body)
            
            # Extract room dimensions
            room = data.get("room", [])
            if len(room) != 2:
                return JsonResponse({"error": "Room dimensions should have two values (length and width)"}, status=400)

            room_length, room_width = room

            # Extract furniture list and ensure it's formatted correctly
            furniture = data.get("furniture", [])
            if not furniture:
                return JsonResponse({"error": "Furniture list cannot be empty"}, status=400)
            
            furniture_features = []
            for item in furniture:
                w = item.get("w", 0)
                h = item.get("h", 0)
                x = item.get("x", 0)
                y = item.get("y", 0)
                
                # Create a feature array for each furniture item (width, height, x, y)
                furniture_features.extend([w, h, x, y])

            # Combine room dimensions and furniture features into one list
            input_features = [room_length, room_width] + furniture_features
            
            # Reshape the data as needed for the model input (1D array into a 2D array with 1 row)
            input_features = np.array(input_features).reshape(1, -1)

            # Make prediction (assume the model is already loaded)
            prediction = model.predict(input_features)
            
            # Assuming the prediction is an array of positions, extract them
            predicted_positions = prediction.tolist()

            # Construct the response data with the predicted positions
            response_data = {
                "predicted_positions": predicted_positions
            }

            return JsonResponse(response_data, status=200)

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=400)