"""
Copyright (C) Microsoft Corporation. All rights reserved.​
 ​
Microsoft Corporation (“Microsoft”) grants you a nonexclusive, perpetual,
royalty-free right to use, copy, and modify the software code provided by us
("Software Code"). You may not sublicense the Software Code or any use of it
(except to your affiliates and to vendors to perform work on your behalf)
through distribution, network access, service agreement, lease, rental, or
otherwise. This license does not purport to express any claim of ownership over
data you may have shared with Microsoft in the creation of the Software Code.
Unless applicable law gives you more rights, Microsoft reserves all other
rights not expressly granted herein, whether by implication, estoppel or
otherwise. ​
 ​
THE SOFTWARE CODE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
MICROSOFT OR ITS LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THE SOFTWARE CODE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
import numpy
import joblib
import os
from azureml.core.model import Model
from inference_schema.schema_decorators \
    import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type \
    import NumpyParameterType
import keras
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd


def init():
    # load the model from file into a global object
    global model
    #data = [[2,230,2]]
    #scalar = MinMaxScaler(feature_range=(-1, 1))
    #scalar = scalar.fit(data)
    #data = scalar.transform(data)
    #print("value of data after transform")
    #print(data)
    #data = data[:, 1:]
    #print("Value of data after reshaping")
    #print(data)
    #data = data.reshape(data.shape[0], 1, data.shape[1])
    #print("Value of data after reshaping")
    #print(data)

    # we assume that we have just one model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder
    # (./azureml-models/$MODEL_NAME/$VERSION)
    model_path = Model.get_model_path(
        os.getenv("AZUREML_MODEL_DIR").split('/')[-2])

    #model = joblib.load(model_path)
    model = keras.models.load_model(model_path)
    print("Model is loaded")

input_sample = numpy.array([[30,178,1690,5]])
output_sample = numpy.array([4])
#input_sample = [[30,178,1690]]
#output_sample = [4]


# Inference_schema generates a schema for your web service
# It then creates an OpenAPI (Swagger) specification for the web service
# at http://<scoring_base_url>/swagger.json
@input_schema('data', NumpyParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
#@input_schema('data', input_sample)
#@output_schema(output_sample)
def run(data, request_headers):
    import json
    scalar = MinMaxScaler(feature_range=(-1, 1))
    scalar = scalar.fit(data)
    data = scalar.transform(data)
    data = data[:, 1:]
    data = data.reshape(data.shape[0], 1, data.shape[1])
    #print(test)
    #print(test.shape)
    #result = model.predict(data)

    y_pred=model.predict(data)
    y_pred = y_pred.reshape(y_pred.shape[0], 1, 1)
    pred_test_set = []
    for index in range(0, len(y_pred)):
        pred_test_set.append(np.concatenate([y_pred[index], data[index]], axis=1))

    # reshape pred_test_set
    pred_test_set = np.array(pred_test_set)
    pred_test_set = pred_test_set.reshape(pred_test_set.shape[0], pred_test_set.shape[2])

    # inverse transform
    pred_test_set_inverted = scalar.inverse_transform(pred_test_set)

    print(pred_test_set_inverted)
    print("predicted value:")
    result = int(y_pred + pred_test_set_inverted[0][3])


    print("value of result is: ")
    print(result)

    # Demonstrate how we can log custom data into the Application Insights
    # traces collection.
    # The 'X-Ms-Request-id' value is generated internally and can be used to
    # correlate a log entry with the Application Insights requests collection.
    # The HTTP 'traceparent' header may be set by the caller to implement
    # distributed tracing (per the W3C Trace Context proposed specification)
    # and can be used to correlate the request to external systems.
    print(('{{"RequestId":"{0}", '
           '"TraceParent":"{1}", '
           '"NumberOfPredictions":{2}}}'
           ).format(
               request_headers.get("X-Ms-Request-Id", ""),
               request_headers.get("Traceparent", ""),
               result
               #len(result)
    ))

    #return {"result": result.tolist()}
    return  {"result": result}

if __name__ == "__main__":
    init()
    test =numpy.array([[0,0,24,5]])
    prediction = run(test, {})  
    print("Test result: ", prediction)
