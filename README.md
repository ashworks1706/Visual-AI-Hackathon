###  E-Scooter Theft Detection and Prevention System

#### **Overview**
This project addresses the increasing issue of e-scooter thefts in outdoor parking lots, specifically at Arizona State University (ASU). Our solution leverages advanced computer vision techniques, GPS tracking, and user authentication to provide a scalable and efficient theft prevention system. By integrating YOLO (You Only Look Once) object detection models with robust user authentication and alert mechanisms, we aim to create a secure and user-friendly platform for e-scooter owners.

---

#### **Key Features**
1. **User Authentication with Clerk Auth**:
   - We implement a secure login system using Clerk Auth to authenticate users.
   - User information, including images, is stored securely in our database for future reference.
   - This ensures that only authorized users can access the e-scooters.

2. **YOLO Model for Face and Scooter Detection**:
   - Our system uses a YOLO model trained on 5000 images from the Roboflow API to detect e-scooters in real-time.
   - Additionally, the YOLO model is trained to recognize user faces, enabling us to match users with their scooters.

3. **E-Scooter Dataset**:
   - The dataset consists of 5000 annotated images of e-scooters, collected and curated through Roboflow.
   - These images include various lighting conditions, angles, and environments to ensure robustness.

4. **Real-Time Detection**:
   - The trained YOLO model detects both the scooter and the user in the frame.
   - It matches the detected face with the registered user data in the database.

5. **GPS Tracking**:
   - The system integrates GPS tracking to monitor the location of each scooter.
   - If a user's GPS location indicates they are away from their scooter but someone else is detected attempting to use it, an alert is triggered.

6. **Automated Alerts**:
   - If unauthorized access is detected, the system sends an email alert to the registered user using Email.js.
   - The email includes details such as the time of detection and an image of the unauthorized individual.

---

#### **Technical Workflow**
1. **User Registration**:
   - Users log in through our platform using Clerk Auth.
   - Their details, including facial images, are stored securely in our database.

2. **Model Training**:
   - A YOLOv8 model is trained on two datasets:
     - A dataset of e-scooters (5000 images) from Roboflow.
     - A dataset of user facial images captured during registration.
   - The training process ensures high accuracy in detecting both scooters and faces in various conditions.

3. **Real-Time Monitoring**:
   - The system continuously monitors parking lots using overhead cameras.
   - It detects scooters and users in real-time and matches them against the database.

4. **Alert Mechanism**:
   - If an unauthorized individual attempts to access a scooter, the system cross-checks GPS data and user proximity.
   - If a mismatch is detected, an email alert is sent instantly to the registered user.

---

#### **Scalability**
Our solution is designed to be scalable across multiple locations and use cases:
- **University Campuses**: Ideal for universities like ASU with large outdoor parking lots prone to thefts.
- **Public Parking Lots**: Can be deployed in public spaces where e-scooters are frequently used.
- **Corporate Campuses**: Useful for companies offering shared mobility solutions for employees.
- **Smart Cities**: Integrates seamlessly into smart city infrastructure for enhanced mobility security.

---

#### **Advantages**
1. **High Accuracy**:
   - By training on a large dataset of 5000 images, our YOLO model achieves high accuracy in detecting scooters and faces.

2. **Real-Time Monitoring**:
   - The system operates in real-time, ensuring immediate detection of suspicious activity.

3. **User-Friendly Interface**:
   - Secure login and registration ensure ease of use for end-users while maintaining robust security.

4. **Proactive Theft Prevention**:
   - Alerts are sent before theft occurs by leveraging GPS data and facial recognition.

5. **Cost-Effective Deployment**:
   - The system uses existing camera infrastructure and cloud-based processing for scalability without significant hardware costs.

---

#### **Future Enhancements**
1. **Integration with Law Enforcement**:
   - Provide law enforcement agencies with real-time access to theft alerts for quicker response times.

2. **Enhanced AI Models**:
   - Incorporate additional models for detecting suspicious behavior or tampering attempts near scooters.

3. **Mobile App Integration**:
   - Develop a mobile app for users to receive alerts, track their scooters, and view real-time camera feeds.

4. **Multi-Camera Support**:
   - Expand support for multiple cameras across larger areas with centralized monitoring.

5. **Battery Monitoring**:
   - Integrate battery status monitoring for e-scooters to provide additional insights to users.

---

#### **Conclusion**
Our E-Scooter Theft Detection System combines cutting-edge AI technology with practical features like GPS tracking and automated alerts to address a pressing issue faced by universities like ASU. With its scalability, accuracy, and ease of deployment, this solution has the potential to revolutionize e-scooter security across various sectors.

-