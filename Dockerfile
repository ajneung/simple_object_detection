# ใช้ Python 3.10 เป็น Base Image
FROM python:3.10

# ตั้งค่า Working Directory
WORKDIR /app

# คัดลอกไฟล์ทั้งหมดไปยัง Container
COPY . /app  

# อัปเดต pip และติดตั้ง dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt  

# ติดตั้งไลบรารีเพิ่มเติมที่ต้องใช้
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# ตั้งค่า Environment Variables เพื่อลด Warning ของ TensorFlow
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV CUDA_VISIBLE_DEVICES=0  

# กำหนดให้ TensorFlow ใช้ GPU ตัวหลัก

# เปิดพอร์ต Flask
EXPOSE 5005

# ใช้ Gunicorn แทน Flask Dev Server เพื่อลดการรีโหลด
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5005", "app:app"]
