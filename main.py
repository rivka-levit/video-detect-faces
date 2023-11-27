import cv2


class VideoFaceDetector:
    """Detect faces in video."""

    cascade = cv2.CascadeClassifier('source/faces.xml')

    def __init__(self, input_video=None):
        self.video = cv2.VideoCapture(input_video)
        self.cat = cv2.imread('source/cat.jpg')
        self.cat_w = self.cat.shape[1]
        self.cat_h = self.cat.shape[0]
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.output = cv2.VideoWriter(
            'outputs/output.avi',
            cv2.VideoWriter.fourcc(*'DIVX'),
            self.fps,
            (self.width, self.height)
        )

    def detect_faces(self, mode='r') -> None:
        """
        Detect faces and mark in by rectangles.
        :param mode: "r" - rectangles, "b" - blur, "c" - cats
        :return: None
        """

        while True:
            frame_exists, frame = self.video.read()

            if frame_exists:
                if mode == 'r':
                    frame = self.draw_rectangles(frame)
                elif mode == 'b':
                    frame = self.blur_faces(frame)
                elif mode == 'c':
                    frame = self.draw_cats(frame)
                else:
                    raise AttributeError('Invalid mode! Allowed modes: "r", '
                                         '"b", "c", by default it is "r".')

                self.output.write(frame)

            else:
                break

        self.output.release()

    def draw_cats(self, frame):
        """Detect faces and draw cats instead of them."""

        faces = self.cascade.detectMultiScale(frame, 1.7, 8)

        try:
            for x, y, w, h in faces:
                frame[y:y + self.cat_h, x:x + self.cat_w] = self.cat
        except ValueError:
            pass

        return frame

    def draw_rectangles(self, frame):
        """Detect faces and draw rectangles around them."""

        faces = self.cascade.detectMultiScale(frame, 1.7, 8)

        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 4)

        return frame

    def blur_faces(self, frame):
        """Detect faces and draw blur area instead of them."""

        faces = self.cascade.detectMultiScale(frame, 1.3, 10)

        for x, y, w, h in faces:
            face_place = frame[y:y + h, x:x + w]
            blur = cv2.GaussianBlur(face_place, (101, 101), 0)
            frame[y:y + h, x:x + w] = blur

        return frame


if __name__ == '__main__':
    dtr = VideoFaceDetector('source/smile.mp4')
    dtr.detect_faces('c')
