from flask import Flask,render_template,request
import houseprice as hp

app=Flask(__name__)

@app.route('/house',methods=['POST'])
def show():
    if request.method=="POST":
        area=request.form['area']
        bhk=request.form['bedroom']
        bath=request.form['bathroom']
        location=request.form['location']
        print(area,location,bath,bhk)
        hppr=hp.house_price(location,area,bath,bhk)
        print(hppr)
    return render_template("housepredictor.html")

@app.route('/',methods=['GET'])
def show1():
    return render_template("housepredictor.html")

if __name__ == '__main__':
    app.run()