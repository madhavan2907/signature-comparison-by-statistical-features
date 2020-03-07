from flask import Flask, request
import sign_match
import json
import os

app = Flask(__name__)

approot=os.getcwd()+'/'

@app.route('/api/signsts', methods = ['GET'])
def getHealthstatus():
    return "service is executing"

def save_file(file):
    filename = file.filename
    filepath = approot + "/input/" + filename + '.jpg'
    file.save(filepath)
    print("File Saved --> " + filename)
    return filepath

#working with single file
@app.route('/ai/sign', methods=['POST'])
def signpoc():
    req = request.json;
    file_path1 = req['filePath1']
    file_path2 = req['filePath2']
    print(req)
    response_dict = {}
    try:
        percent_match = sign_match.compare_api(file_path1, file_path2)
        response_dict['status']=1
        response_dict['response']= [percent_match,approot+"input/1_1.png",approot+"input/1_2.png",approot+"input/1_3.png",
                     approot+"input/2_1.png",approot+"input/2_2.png",approot+"input/2_3.png"]
    except ValueError as ve1:
        print(ve1)
        response_dict['status']=0
        response_dict['response']= ["Error in comparision!"]
    return json.dumps(response_dict)
	

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5010,debug=True)
