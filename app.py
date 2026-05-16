"""
app.py
======
Flask application factory + entry point.
"""

import os
import glob
import shutil
from flask import Flask, render_template


def _clean_uploads(upload_folder: str) -> None:
    """Delete all files inside the uploads folder on startup."""
    if not os.path.isdir(upload_folder):
        return
    for entry in os.scandir(upload_folder):
        try:
            if entry.is_file() or entry.is_symlink():
                os.remove(entry.path)
            elif entry.is_dir():
                shutil.rmtree(entry.path)
        except Exception as exc:
            print(f"[startup] Could not remove {entry.path}: {exc}")


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
    _clean_uploads(UPLOAD_FOLDER)

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
