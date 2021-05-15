from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from prediction.predict import predict

# Create your views here.
test = {
    "gender": "Male",
    "age": 67,
    "hypertension": 0,
    "heart_disease": 0,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "smoking_status": "formerly smoked"
}

[['Male', 67.0, 0, 1, 'Yes', 'Private',
  'Urban', 228.69, 36.6, 'formerly smoked']]


@api_view(['GET', 'POST'])
def api_add(request):
    sum = 0
    response_dict = {}
    if request.method == 'GET':
        # Do nothing
        pass
    elif request.method == 'POST':
        # Add the numbers
        x = []
        data = request.data
        for key in data:
            x.append(data[key])
        x = [x]
        response_dict = {"predicted": predict(x)}
        print(response_dict)
    return Response(response_dict, status=status.HTTP_201_CREATED)
