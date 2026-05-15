"""
app.py
======
Flask application factory + entry point.
"""

import os
import glob
import shutil
from flask import Flask, render_template


def create_app():
    app = Flask(
        __name__,
        template_folder=os.path.join("base", "templates"),
        static_folder=os.path.join("base", "static"),
    )

    # ── Load config
    from config.config import (
        SECRET_KEY, DEBUG, UPLOAD_FOLDER,
        MAX_CONTENT_LENGTH, SAVE_COCO_DEFAULT,
    )
    app.secret_key          = SECRET_KEY
    app.config["DEBUG"]     = DEBUG
    app.config["UPLOAD_FOLDER"]      = UPLOAD_FOLDER
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
    app.config["SAVE_COCO_DEFAULT"]  = SAVE_COCO_DEFAULT

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # ── Register blueprints
    from base.com.controller.car_analysis_controller import bp as car_bp
    app.register_blueprint(car_bp)

    # ── Root
    @app.route("/")
    def home():
        return render_template("car_analysis/index.html")

    @app.errorhandler(413)
    def too_large(e):
        from flask import jsonify
        return jsonify({"error": "File too large. Max 20 MB allowed."}), 413

    return app


if __name__ == "__main__":
    from config.config import HOST, PORT, DEBUG
    app = create_app()
    app.run(host=HOST, port=PORT, debug=DEBUG)
