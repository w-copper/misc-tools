import flask
import os
from flask import Flask, request, jsonify

app = flask.Flask(__name__)

RET = 0


@app.route("/api/user/login", methods=["POST", "GET"])
def login():
    return jsonify({"code": 0, "data": {"token": "musk"}})


@app.route("/api/drone/live/lists", methods=["POST", "GET"])
def drone_live_lists():
    L = [
        {
            "PublishUrl": "rtmp://127.0.0.1:1935/videotest",
            "liveUrl": "rtmp://127.0.0.1:1935/videotest",
        },
    ]
    # L = []

    return jsonify({"code": 0, "data": L})
