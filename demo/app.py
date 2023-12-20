from flask import Flask, render_template, request, send_from_directory
from src.inference import get_prediction
from src.visualization import plot
from src.model_config import get_tsne_result

app = Flask(__name__)


@app.route("/")
def landing():
    return render_template(
        "landing.html",
    )


@app.route("/", methods=["POST"])
def show_result():
    smiles = request.form["smiles"]
    result_bbb, result_cyp, result_sol, result_clr, drugLikeness = get_prediction(
        smiles
    )

    tsne_result = get_tsne_result()
    script, div = plot(tsne_result)

    return render_template(
        "result.html",
        smiles=smiles,
        result_bbb=f"{(result_bbb * 100):.1f}%",
        result_cyp=f"{(result_cyp * 100):.1f}%",
        result_sol=f"{result_sol:.4f}",
        result_clr=f"{result_clr:.4f}",
        drug_likeness=f"{drugLikeness:.2f}%",
        script=script,
        div=div,
    )


@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory("images", filename)


if __name__ == "__main__":
    app.run()
