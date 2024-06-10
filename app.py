import traceback

from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

from australiamedicarecodes import provide_australian_medicare_codes

app = Flask(__name__)
CORS(app, support_credentials=True)
# app.secret_key = ""
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True  # 2 space indentation
app.config['JSON_SORT_KEYS'] = False  # avoids jsonify to sort the keys in alphabetical manner


@app.route('/')
@cross_origin(supports_credentials=True)
def home_page():
    result = [
        {
            'Created By': 'Global Medics Australia Pty Ltd : https://globalmedics.ai/',
            'description': 'Australian Medicare Codes',
        }
    ]
    return jsonify(result)


# predict will provide the response for the function created in Model Prediction.py
@app.route('/australianmedicarecodes', methods=['POST', 'GET'])
@cross_origin(supports_credentials=True)
def summarise_conversation():
    if request.method == 'POST':
        try:
            data = request.json
            if data is not None:
                response = jsonify(provide_australian_medicare_codes(data))
                return response
            else:
                return "Data is not defined."
        except:
            return jsonify({'trace': traceback.format_exc()})


if __name__ == '__main__':
    app.run(debug=False, port=5008)
