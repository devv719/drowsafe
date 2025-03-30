from flask import Flask, jsonify, render_template
import subprocess

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")  # Serve index.html from the templates folder

@app.route('/start-demo', methods=['GET'])
def start_demo():
    try:
        # Run the drowsiness detection script
        subprocess.Popen(["python", "drowsiness_detection.py"])  # Ensure the script is in the same folder
        return jsonify({"status": "success", "message": "Demo started"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
@app.route('/dashboard', methods=['GET'])
def dashboard():
    return render_template("dashboard.html")
if __name__ == '__main__':
    app.run(debug=True)
