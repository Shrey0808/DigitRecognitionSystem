# from prediction import preprocess_and_extract_digits
# import matplotlib.pyplot as plt

# image_path = 'static\img1.jpg' 
# predictions, digit = preprocess_and_extract_digits(image_path)
# print("Predictions:", predictions)




from flask import Flask ,render_template, url_for, request , jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# POST
# @app.route('/predict',methods = ['GET','POST'])
# def predict():
#     if request.method =='POST':
#         n = int(request.form['nitrogen'])
#         p = int(request.form['phosphorus'])
#         k = int(request.form['potassium'])
#         t = float(request.form['temperature'])
#         h = float(request.form['humidity'])
#         ph = float(request.form['ph'])
#         r = float(request.form['rainfall'])
#         user_data = [[n , p , k , t, h, ph , r ]]
#         trans_data = scaler.transform(user_data)
#         prediction = kmeans.predict(trans_data)[0]
#         print(prediction)
#         dt = dict(df[df['cluster_12'] == prediction]['label'].value_counts())
#         ls = []
#         for k,v in dt.items():
#             if v>=70:
#                 ls.append(k)
#         print(ls)
#         return jsonify(ls)



if __name__ =="__main__":
    app.run(debug = True)