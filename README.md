---
title: Phishing Detection Web App
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 5000
pinned: false
---

# Phishing Detection Application
Detecting phishing attacks using a combined model of LSTM and CNN (TensorFlow / Keras).

## Features
- Hybrid CNN + LSTM Neural Network architecture trained on 40,000 URLs.
- Real-time DOM HTML & lexical feature extraction.
- OAuth2 Google login integration.
- Flask REST API endpoint (`/api/scan`).
