import numpy as np
import pandas as pd
from PIL import Image
import os
import cv2

folder = "data/widerface/manual"
files = os.listdir(folder)

def manual_labeling():
    data = []
    total_files = len(files)

    print("Instructions:")
    print("For Male: 'm' = yes, 'f' = no")
    print("For Eyeglasses: 'y' = yes, 'n' = no")
    print("'r' = redo current image, 'c' = confirm and move to next, 'q' = quit early\n")

    for i, f in enumerate(files, start=1):
        img_path = os.path.join(folder, f)
        img = Image.open(img_path)
        img = np.array(img)  # shape (H,W,3), RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.namedWindow("Image", cv2.WINDOW_GUI_NORMAL)

        while True:  # loop until user confirms
            male, eyeglasses = None, None

            # Select Male first
            while male is None:
                cv2.imshow("Image", img)
                print(f"\nFile {i}/{total_files}: {f}")
                print("For Male: 'y' = yes, 'n' = no, 'q' = quit")
                key = cv2.waitKey(0) & 0xFF
                if key == ord('y'):
                    male = 1
                elif key == ord('n'):
                    male = 0
                elif key == ord('q'):
                    cv2.destroyAllWindows()
                    df = pd.DataFrame(data, columns=["filename", "eyeglasses", "male"])
                    df.to_csv(os.path.join(folder, "widerface_faces_labels.csv"), index=False)
                    return df

            # Select Eyeglasses second
            while eyeglasses is None:
                cv2.imshow("Image", img)
                print("For Eyeglasses: 'y' = yes, 'n' = no, 'q' = quit")
                key = cv2.waitKey(0) & 0xFF
                if key == ord('y'):
                    eyeglasses = 1
                elif key == ord('n'):
                    eyeglasses = 0
                elif key == ord('q'):
                    cv2.destroyAllWindows()
                    df = pd.DataFrame(data, columns=["filename", "eyeglasses", "male"])
                    df.to_csv(os.path.join(folder, "widerface_faces_labels.csv"), index=False)
                    return df

            # Show choices and allow redo
            print(f"Your choices: Male={male}, Eyeglasses={eyeglasses}")
            print("Press 'c' to confirm, 'r' to redo, 'q' to quit")
            key = cv2.waitKey(0) & 0xFF
            if key == ord('c'):
                data.append([f, eyeglasses, male])
                print(f"Saved: {f} -> Male={male}, Eyeglasses={eyeglasses}\n")
                break
            elif key == ord('r'):
                print("Redoing this image...\n")
                continue
            elif key == ord('q'):
                cv2.destroyAllWindows()
                df = pd.DataFrame(data, columns=["filename", "eyeglasses", "male"])
                df.to_csv(os.path.join(folder, "widerface_faces_labels.csv"), index=False)
                return df

    cv2.destroyAllWindows()
    df = pd.DataFrame(data, columns=["filename", "eyeglasses", "male"])
    df.to_csv(os.path.join(folder, "widerface_faces_labels.csv"), index=False)
    print(f"\nSaved {len(data)} labels to widerface_faces_labels.csv")
    return df

if __name__ == "__main__":
    manual_labeling()
