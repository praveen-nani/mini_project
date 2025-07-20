# Live Expression Simulation from Static Images 🎭🎥

A real-time facial expression simulation system using Generative Adversarial Networks (GANs), enabling dynamic expression transfer from live webcam input onto static face images. This project integrates deep learning with an intuitive Tkinter-based UI, OpenCV-powered live video processing, and automated watermarking for content authenticity.

---

## 📌 Key Features

* **Live Expression Transfer:** Uses a webcam feed to map real-time facial expressions onto static images.
* **Deep Learning Powered:** Built using Generative Adversarial Networks (GANs) for dynamic expression generation.
* **Intuitive GUI:** Desktop application with a Tkinter interface for easy control and interaction.
* **Real-Time Recording:** Supports live video recording of generated outputs.
* **Watermarking:** Automatically applies your app’s logo as a watermark for security and authenticity.
* **Privacy-Supportive:** Embeds watermark during recording to discourage misuse.

---

## 📂 Project Overview

* **Frontend:**

  * `TkinterApp.py`: Main desktop GUI application.
  * Handles image loading, webcam control, real-time generation, video recording, and watermarking.

* **Backend Models:**

  * `modules/`: Contains core deep learning modules:

    * `generator.py`: Image generation using GANs.
    * `dense_motion.py`: Motion estimation.
    * `keypoint_detector.py`: Facial keypoint detection.
    * `discriminator.py`, `model.py`, `util.py`: Supporting model files.

* **Synchronized BatchNorm:**

  * `sync_batchnorm/`: Custom batch normalization implementation, adapted for multi-device compatibility (application runs on single GPU as tested).

* **Assets:**

  * `buttons/`: UI button images.
  * `media/`: Static images for expression animation.

---

## 🚀 How to Run

1. **Clone the Repository:**

```bash
git clone https://github.com/praveen-nani/mini_project.git
cd mini_project
```

2. **Prepare a Static Image:**

* Place your static face image inside the `media/` folder.

3. **Install Dependencies:**

```bash
pip install torch torchvision opencv-python numpy pillow
```

4. **Run the Application:**

```bash
python TkinterApp.py
```

5. **Using the Application:**

* Use GUI buttons to:

  * Start/stop live webcam feed.
  * Load a static image.
  * Watch live expressions animate the static image.
  * Record video (with watermark embedded).

---

## 💻 Technologies Used

* **Artificial Intelligence & Deep Learning:** PyTorch-based GAN models.
* **Computer Vision:** OpenCV for live webcam streaming and frame handling.
* **User Interface:** Python Tkinter for GUI development.
* **Watermarking:** PIL and OpenCV for applying watermark/logo during video recording.

---

## 🎥 Demo
![FakeFace_20220427T104212](https://github.com/user-attachments/assets/539b0287-7796-45da-9d93-e062bed5d730)


---

## ⚙️ System Requirements

* Python 3.7 or higher
* Compatible with Windows, Linux
* NVIDIA GPU recommended for better performance (runs on CPU too)

---

## 📄 License

This project currently does not use a license.

---

## 🤝 Contribution

Feel free to fork and contribute via pull requests.

---
