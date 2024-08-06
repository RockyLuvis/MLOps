from flask import Flask, render_template, jsonify, request

from src.pipeline.prediction_pipeline import PredictPipeline, CustomData

app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template("index.html")

@app.route('/submit', methods=["GET", "POST"])
def predict_datapoints():
    if request.method=="GET":
        return render_template("form.html")
    else:
        #Initialize custome data object
        data = CustomData(
            carat=float(request.form.get("carat")),
            depth = float (request.form.get("depth")),
            table = float (request.form.get("table")),
            x = float (request.form.get("x")),
            y = float(request.form.get("y")),
            z = float (request.form.get("z")),
            cut = request.form.get("cut"),
            color = request.form.get("color"),
            clarity = request.form.get("clarity"),
        )
        Converted_df_data = data.get_data_as_dataframe()

        print("Converted_df_data:",Converted_df_data)

        #now call Prediction pipeline

        predict_pipeline = PredictPipeline()

        pred = predict_pipeline.predict(Converted_df_data)

        result = round (pred[0],2)

        return render_template("result.html",result=result)

   
if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8000)
