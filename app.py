from flask import Flask, request, render_template, render_template_string, session, redirect, url_for
import requests
from requests.exceptions import RequestException
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from string import printable
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing import sequence
from keras.models import load_model
import warnings
from urllib3.exceptions import InsecureRequestWarning
from functools import wraps
import os
from authlib.integrations.flask_client import OAuth
from flask import jsonify

from extractor import Extractor


app = Flask(__name__)
app.secret_key = os.environ.get("APP_SECRET_KEY", "change-me-in-production")

oauth = OAuth(app)
google = oauth.register(
    "google",
    client_id=os.environ.get("GOOGLE_CLIENT_ID"),
    client_secret=os.environ.get("GOOGLE_CLIENT_SECRET"),
    access_token_url="https://oauth2.googleapis.com/token",
    authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
    api_base_url="https://www.googleapis.com/oauth2/v2/",
    client_kwargs={"scope": "openid email profile"},
)


# Load training data and model once at startup
legitimate_train = pd.read_csv("features/legitimate_train.csv")
phish_train = pd.read_csv("features/phish_train.csv")
BASE_DF = pd.concat([legitimate_train, phish_train], axis=0).reset_index(drop=True)

MODEL = load_model("models/model_C.h5")

# disable insecure HTTPS warnings (demo only)
warnings.simplefilter("ignore", InsecureRequestWarning)


def create_scaler(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_scale = ["html_length", "n_hyperlinks", "n_script_tag", "n_link_tag", "n_comment_tag"]
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[[c + "_std" for c in cols_to_scale]] = scaler.fit_transform(df_scaled[cols_to_scale].astype(float))
    df_scaled = df_scaled.drop(columns=cols_to_scale)
    return df_scaled


def create_X_1(temp_X_1: pd.DataFrame):
    url_int_tokens = [[printable.index(x) + 1 for x in url if x in printable] for url in temp_X_1.url]
    max_len = 150
    X_new_1 = sequence.pad_sequences(url_int_tokens, maxlen=max_len)
    return X_new_1


def create_X_2(temp_X_2: pd.DataFrame):
    x = temp_X_2.drop(columns=["url"]).values.astype(float)
    X_new_2 = x.reshape(x.shape[0], x.shape[1], 1)
    return X_new_2


def build_feature_row(url: str, html: str) -> pd.DataFrame:
    hostname = urlparse(url).hostname
    soup = BeautifulSoup(html, "html.parser")

    extractor = Extractor()

    general_features = extractor.general_f(soup, hostname)
    a_tag_features = extractor.a_tag(soup, hostname)
    form_tag_features = extractor.form_tag(soup, hostname)
    ext_resource_features = extractor.ext_resource(soup, hostname)
    favicon_feature = extractor.favicon_feature(soup, url, hostname)

    row = {
        "url": url,
        "n_hyperlinks": a_tag_features["n_hyperlinks"],
        "null_p_ratio": a_tag_features["nullpointers_ratio"],
        "external_l_ratio": a_tag_features["external_ratio"],
        "p_data_forms": form_tag_features["form_input_b"],
        "html_length": general_features["html_length"],
        "n_script_tag": general_features["n_script_tag"],
        "n_link_tag": general_features["n_link_tag"],
        "n_comment_tag": general_features["n_comment_tag"],
        "ext_res_ratio": ext_resource_features,
        "favicon_used": favicon_feature,
        "int_form_act_ratio": form_tag_features["int_form_act_ratio"],
        "abn_form_act_ratio": form_tag_features["abn_form_act_ratio"],
        "ext_form_act_ratio": form_tag_features["ext_form_act_ratio"],
        "title_tag": general_features["title_tag"],
        "title_url_brand": general_features["title_url_brand"],
        # dummy label so the column exists; not used for prediction
        "result_flag": 0,
    }

    return pd.DataFrame([row])


# REMOVED INLINE TEMPLATES

def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login", next=request.path))
        return f(*args, **kwargs)

    return wrapper


@app.route("/", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        if username == "admin" and password == "password123":
            session["user"] = username
            next_url = request.args.get("next") or url_for("detect")
            return redirect(next_url)
        else:
            error = "Invalid username or password."

    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


@app.route("/auth/google")
def auth_google():
    redirect_uri = url_for("auth_google_callback", _external=True)
    return google.authorize_redirect(redirect_uri)


@app.route("/auth/google/callback")
def auth_google_callback():
    token = google.authorize_access_token()
    resp = google.get("userinfo")
    user_info = resp.json()
    # use email as session identifier
    session["user"] = user_info.get("email") or "google_user"
    return redirect(url_for("detect"))


@app.route("/auth/facebook")
def auth_facebook():
    # Demo-only: in a real app, redirect to Facebook's OAuth2 flow
    session["user"] = "facebook_user"
    return redirect(url_for("detect"))


@app.route("/auth/apple")
def auth_apple():
    # Demo-only: in a real app, redirect to Apple's OAuth2 flow
    session["user"] = "apple_user"
    return redirect(url_for("detect"))


@app.route("/detect", methods=["GET", "POST"])
@login_required
def detect():
    # Detect route now simply renders the SPA and passes the logged-in user via template var
    return render_template("login.html", auto_login=True, email=session.get("user"))

@app.route("/api/scan", methods=["POST"])
def api_scan():
    data = request.get_json()
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "Please enter a URL."}), 400
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    try:
        try:
            # Short timeout so offline demo URLs fail fast and fallback
            resp = requests.get(
                url,
                timeout=4,
                headers={"User-Agent": "Mozilla/5.0"},
                verify=False,
            )
            resp.raise_for_status()
            html = resp.text
        except RequestException:
            html = ""  # Fallback to lexical-only features if website is unreachable

        sample_df = build_feature_row(url, html)
        df_all = pd.concat([BASE_DF, sample_df], axis=0).reset_index(drop=True)
        df_scaled = create_scaler(df_all)

        X = df_scaled.drop(columns=["result_flag"])
        sample_scaled = X.tail(1)

        x1 = create_X_1(sample_scaled)
        x2 = create_X_2(sample_scaled)

        proba = float(MODEL.predict([x1, x2])[0][0])
        verdict = "phishing" if proba >= 0.5 else "safe"
        confidence = int(proba * 100) if proba >= 0.5 else int((1.0 - proba) * 100)
        
        return jsonify({
            "verdict": verdict,
            "confidence": confidence
        })
    except Exception as exc:
        return jsonify({"error": f"Unexpected error processing URL: {exc}"}), 500


if __name__ == "__main__":
    app.run(debug=True)

