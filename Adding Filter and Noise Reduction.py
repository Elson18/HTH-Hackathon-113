def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    blurred = cv2.GaussianBlur(enhanced_bgr, (5, 5), 0)
    final_frame = cv2.bilateralFilter(blurred, 9, 75, 75)

    return final_frame
