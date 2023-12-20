from flask import Flask, render_template, request, redirect
from src.inference import get_prediction
from src.visualization import plot

app = Flask(__name__)


@app.route("/")
def landing():
    return render_template(
        "landing.html",
    )


from pandas import DataFrame

df = DataFrame()
df["1"] = [1, 2, 3]
df["2"] = [1, 2, 3]
df["image_url"] = ["", "", ""]
df["color"] = ["red", "blue", "green"]


@app.route("/", methods=["POST"])
def show_result():
    smiles = request.form["smiles"]
    result_bbb, result_cyp, result_sol, result_clr, drugLikeness = get_prediction(
        smiles
    )

    script, div = plot(df)

    return render_template(
        "result.html",
        smiles=smiles,
        result_bbb=f"{(result_bbb * 100):.1f}%",
        result_cyp=f"{(result_cyp * 100):.1f}%",
        result_sol=f"{result_sol:.4f}",
        result_clr=f"{result_clr:.4f}",
        drug_likeness=f"{drugLikeness}%",
        script=script,
        div=div,
    )


if __name__ == "__main__":
    app.run()
