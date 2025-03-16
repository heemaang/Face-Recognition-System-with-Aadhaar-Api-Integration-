# 📸 Attendance Management System with Face Recognition

## 🏠 Home Page

### 🗂️ Table of Contents
- Introduction
- Features
- User Management
- Attendance Management
- Technologies Used
- How It Works
- Screenshots
- Installation and Setup
- Future Enhancements
- Developer Info

## 🚀 Introduction
This project is built to explore API integrations with Aadhaar authentication for secure and efficient attendance tracking. It leverages facial recognition technology and aims to provide a seamless way to manage attendance using AI-driven automation.

Developed by: **Heemaang Saxena**

### ✨ Features
#### 👤 User Management
- **Add User:**
  - Register users with their name and ID.
  - System captures 10 images for facial recognition.
  - Initially, users are unregistered.
- **Register User:**
  - Admin assigns sections to users for registration.
- **Unregister/Remove User:**
  - Admin can remove users from the system.

#### ✅ Attendance Management
- **Mark Attendance:**
  - Uses face recognition for secure attendance logging.
- **View Attendance:**
  - Admin can filter attendance by date or user ID.

### 🛠️ Technologies Used
#### Programming Languages & Libraries
- **Python**
- **Flask** - Backend web framework
- **OpenCV (cv2)** - For image processing & face recognition
- **NumPy** - For numerical computations
- **Pandas** - For data handling
- **Joblib** - Model serialization
- **CSV** - Data storage

#### Machine Learning
- **scikit-learn** - K-Nearest Neighbors (KNN) algorithm for face recognition

### ⚙️ How It Works
1. **Add User:** Enter details and capture images.
2. **Register User:** Assign a section to enable attendance tracking.
3. **Mark Attendance:** The system recognizes faces and logs attendance.
4. **View Attendance:** Admin filters attendance records.

### 📸 Screenshots
(Add relevant screenshots here)

### 📹 Watch the Full Setup Video
👉 [[Video Link Here](https://drive.google.com/file/d/1tlKJuaeHhrAreIZyNgXcZBLSFp0eQUrH/view)] 

### 🛠️ Installation and Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-link
   cd attendance-management-system
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Flask application:
   ```bash
   flask run
   ```
5. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

### 🚀 Future Enhancements
- Aadhaar-based biometric authentication
- Mobile app integration
- AI-driven analytics for attendance trends
- Multi-factor authentication

### 👨‍💻 Developer Info
**Heemaang Saxena**
- **GitHub:** [https://github.com/heemaang](https://github.com/heemaang)
- **Email:** heemaang.saxena18@gmail.com

### 📢 Contact
For contributions or enhancements, feel free to reach out!

