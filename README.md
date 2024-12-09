# 🖼️ Ancient Vision  
**Transform your images into timeless ancient art!**  
A collaborative group project for image conversion using advanced AI techniques to recreate ancient art styles.

---

## ✨ Features

- 🎨 Convert modern images into **ancient art styles** (e.g., Renaissance, Baroque, Egyptian).  
- 📤 **Image Upload:** Upload in JPEG formats.  
- ⚙️ **Style Selection:** Choose and apply ancient styles to your images.  
- 🔧 **Customization Controls:** Adjust style intensity for desired effects.  
- 🖼️ **Preview Function:** Real-time preview of converted images before final download.  
- 📊 **High-Resolution Outputs:** Get high-quality image results suitable for printing or sharing.

---

## 📚 Tech Stack

### Frontend:
- **React.js** (UI Development)  
- **Axios** (API Integration)  
- **React Router** (Navigation)

### Backend:
- **Flask** (API Development)  
- **Flask-CORS** (Cross-Origin Resource Sharing)  
- **pymongo** (MongoDB Integration)  
- **python-dotenv** (Environment Management)

### Database:
- **MongoDB** (Image metadata and storage)

### Machine Learning:
- **Neural Style Transfer (NST)**  
- **Generative Adversarial Networks (GANs)**

---

## 🔰 Getting Started  

Follow these steps to set up and run the project locally.  

### Prerequisites
Ensure the following software is installed on your system:  
- **Python 3.13.1** or later  
- **Node.js** (v18+) and **npm**  
- **MongoDB** (Local or Atlas)  
- **Git**

---

### 🖥️ Frontend Setup  

1. **Fork the repository by clicking on `Fork` option on top right of the main repository.**
2. **Clone the Repository:**  
   ```bash
   git clone https://github.com/<your-username>/Ancient-Vision.git
   cd Ancient-Vision/frontend
   ```
3. **Install Dependencies:**
    ```bash
    npm install
    ```
4. **Run the Development Server:**
    ```bash
    npm start
    ```
5. **Open http://localhost:3000 in your browser to view the app.**

---

### ⚙️ Backend Setup

1. **Navigate to the Backend Directory:**  
   ```bash
   cd Ancient-Vision/backend
   ```
2. **Set Up a Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4. **Configure Environment Variables:**
    Create a .env file with the following variables:
    ```bash
    MONGO_URI=mongodb://localhost:27017/ancient_vision_db
    SECRET_KEY=supersecretkey
    ```
5. **Run the Backend Server:**
    ```bash
    export FLASK_APP=app
    flask run
    ```
6. **The backend will run at http://127.0.0.1:5000.**

---
## Contributors
- [Vinay Surwase](https://github.com/VinaySurwase)
- [Taanvi Khevaria](https://github.com/taanvi2205)
- [Meenakshi Shibu](https://github.com/meenakshishibu16)
- [Om Parate](https://github.com/omparate7)
- [Abhinav Harsh](https://github.com/Abhinav-creator45)

---

## License
This project is for academic purposes only and is restricted to contributors.
Not intended for public use or distribution.

---

For details on how to contribute, see [CONTRIBUTING.md](https://github.com/VinaySurwase/Ancient-Vision/blob/main/CONTRIBUTING.md).