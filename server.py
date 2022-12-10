from typing import List, Tuple
from flask import Flask, jsonify, request, render_template, flash
from config import ApplicationConfig

app = Flask(__name__)
app.config.from_object(ApplicationConfig)


@app.route("/", methods =  ["GET", "POST"])
def mainpage():

    if request.method == "POST":
        link = request.form["link"]
        if not link:
            flash("Link cannot be empty")
        else:
            return render_template("homepage.html", link=link)
    else:
        return render_template('homepage.html')



if __name__ == '__main__':
    app.run(debug=True, threaded=True)
