import cv2


class VideoFaceDetector:
    """Detect faces in video."""

    cascade = cv2.CascadeClassifier('source/faces.xml')

    def __init__(self, input_video=None):
        self.video = cv2.VideoCapture(input_video)
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.output = cv2.VideoWriter(
            'outputs/output.avi',
            cv2.VideoWriter.fourcc(*'DIVX'),
            self.fps,
            (self.width, self.height)
        )

    def detect_faces_rects(self, mode='r'):
        """Detect faces and mark in by rectangles."""

        while True:
            frame_exists, frame = self.video.read()
            if frame_exists:
                if mode == 'r':
                    frame = self.draw_rectangles(frame)
                elif mode == 'b':
                    frame = self.blur_faces(frame)
                self.output.write(frame)
            else:
                break

        self.output.release()

    def draw_rectangles(self, frame):
        faces = self.cascade.detectMultiScale(frame, 1.1, 8)
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 4)

        return frame

    def blur_faces(self, frame):
        faces = self.cascade.detectMultiScale(frame, 1.1, 8)
        for x, y, w, h in faces:
            ROI = frame[y:y + h, x:x + w]
            blur = cv2.GaussianBlur(ROI, (51, 51), 0)
            frame[y:y + h, x:x + w] = blur

        return frame


if __name__ == '__main__':
    dtr = VideoFaceDetector('source/video.mp4')
    dtr.detect_faces_rects('b')
