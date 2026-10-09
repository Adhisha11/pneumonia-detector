# Pneumonia Detection AI

A web app that classifies a chest X-ray image into one of four classes using a trained deep learning model:

- Normal
- Bacterial Pneumonia
- Viral Pneumonia
- COVID-19 Pneumonia

A user logs in, uploads an X-ray, and the Flask backend runs it through a TensorFlow/Keras model and returns the predicted class with class probabilities. The frontend shows the result with a short advisory message.

> **Medical disclaimer.** This is an educational project. It is not a medical device and must not be used to diagnose, treat or rule out any condition. Always consult a qualified healthcare professional.

---

## Features

- Login and registration pages
- Drag-and-drop or click-to-upload for JPG and PNG X-rays
- Prediction from a trained Keras model (`.h5`), served by a Flask REST API
- Probability for every class, plus the top prediction
- Colour-coded result page with a message for each class
- Responsive Tailwind CSS interface with a landing page
- A reports page (static design mock-up, see Limitations)

---

## Tech stack

| Part | Technology |
|---|---|
| Backend | Python, Flask, Flask-CORS |
| Model | TensorFlow / Keras (`.h5` file) |
| Image handling | Pillow, NumPy |
| Frontend | HTML, Tailwind CSS (CDN), vanilla JavaScript |

---

## How it works

```
Login  ->  Upload X-ray  ->  POST /predict  ->  Result page
                               |
                    resize to 150 x 150, RGB
                    scale pixels to 0-1
                    model.predict()
                    return class + probabilities
```

The upload page sends the image to `http://127.0.0.1:5000/predict`. The JSON response is stored in the browser's `localStorage`, and the result page reads it from there.

**Class order.** The model outputs probabilities in the order the classes were used in training. The backend maps them as:

```
0 -> COVID-19 Pneumonia
1 -> Normal
2 -> Bacterial Pneumonia
3 -> Viral Pneumonia
```

If you retrain the model, change this list in `app.py` to match your training class order.

---

## Project structure

```
project/
├── backend/
│   ├── app.py                      Flask API
│   └── my_chest_xray_model.h5      trained model (must be here)
├── frontend/
│   ├── upload.html                 upload page
│   ├── dashboard.html              result page
│   └── ...                         landing, login and reports pages
│   └── pneumo.jpg.jpg              placeholder X-ray image
└── vercel.json                     deployment config
```

Adjust the names to match your repository. The landing and login pages are in the `frontend` folder as well.

---

## Getting started

You need Python 3.9 to 3.11 (check which versions your TensorFlow release supports).

### 1. Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux

pip install flask flask-cors tensorflow numpy pillow
python app.py
```

Make sure `my_chest_xray_model.h5` is in the `backend` folder. The API starts at `http://127.0.0.1:5000`. Opening that address in a browser shows "Pneumonia Detection API is running!".

### 2. Frontend

The frontend is plain HTML, so no build step is needed. Open the landing or login page in a browser. For best results, serve the folder with a simple server:

```bash
cd frontend
python -m http.server 8080
```

Then open `http://localhost:8080`. Keep the backend running at the same time.

### 3. Use the app

1. Register an account, then log in.
2. Upload a chest X-ray (JPG or PNG).
3. Click **Analyze** and read the result.

---

## API

### `GET /`

Health check. Returns a text message.

### `POST /predict`

Send the image as multipart form data with the field name `file`.

```bash
curl -X POST -F "file=@xray.jpg" http://127.0.0.1:5000/predict
```

Example response:

```json
{
  "predicted_class": "Bacterial Pneumonia",
  "confidence": 9.4321,
  "probabilities": {
    "COVID-19 Pneumonia": 0.01,
    "Normal": 0.02,
    "Bacterial Pneumonia": 0.94,
    "Viral Pneumonia": 0.03
  }
}
```

Errors return `400` (no file) or `500` (processing failure) with an `error` message.

---

## Known issues and limitations

- **Confidence value.** `app.py` multiplies the top probability by 10, so the `confidence` field is on a 0 to 10 scale. To get a percentage, multiply by 100 instead. The frontend does not currently show this value.
- **Duplicate class list.** `CLASS_NAMES` is defined twice in `app.py`. The one inside `predict()` is the one used. Remove the unused one at the top of the file to avoid confusion.
- **Demo-only login.** Accounts are stored in the browser's `localStorage` with plain-text passwords, and nothing is checked on the server. Do not use real passwords or patient data. A real deployment needs server-side authentication and hashed passwords.
- **Reports page is a mock-up.** It shows sample patients and dates, and it is not connected to the backend or to real results.
- **Result image.** The backend does not return the uploaded image, so the result page shows the placeholder X-ray unless an `image` field is added to the response.
- **Unused message.** The result page includes a "Fungal Pneumonia" message, but the model has no such class.
- **Backend address is hard-coded** to `http://127.0.0.1:5000` in the upload page. Change it when you deploy the backend elsewhere.
- **Model quality.** Accuracy depends on the training data and has not been clinically validated. Do not rely on it for real decisions.
- **Deployment.** TensorFlow and the model file are large. Serverless platforms such as Vercel have size limits that this app may exceed. A host that runs a normal Python server (Render, Railway, a VM) is usually a better fit for the backend.


