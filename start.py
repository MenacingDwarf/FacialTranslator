from FacialTranslator import FacialTranslator
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--input', default='0',
                        help='Video source. Can be an integer for webcam or a string for a video file.')
    parser.add_argument('--ip', default='127.0.0.1',
                        help='IP address of the Unreal LiveLink server.')
    parser.add_argument('--port', default=11111,
                        help='Port of the Unreal LiveLink server.')
    parser.add_argument('--hide_image', action='store_true',
                        help='Hide the image window.')
    parser.add_argument('--show_debug', action='store_true',
                        help='Show debug window.')
    args = parser.parse_args()

    print("Starting FacialTranslator")
    mediapipe_face = FacialTranslator(args.input, args.ip, args.port, args.hide_image, args.show_debug)
    mediapipe_face.start()