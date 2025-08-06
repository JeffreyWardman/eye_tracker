import bz2
import os
import urllib.request


def download_shape_predictor(save_folder: str = "assets") -> None:
    # Download and extract shape_predictor if not present
    predictor_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    predictor_bz2 = os.path.join(save_folder, "shape_predictor_68_face_landmarks.dat.bz2")
    predictor_dat = os.path.join(save_folder, "shape_predictor_68_face_landmarks.dat")

    if not os.path.exists(predictor_dat):
        os.makedirs(save_folder, exist_ok=True)
        print("Downloading shape_predictor_68_face_landmarks.dat.bz2...")
        urllib.request.urlretrieve(predictor_url, predictor_bz2)  # noqa: S310

        print("Extracting shape_predictor_68_face_landmarks.dat...")
        with bz2.open(predictor_bz2, "rb") as f_in, open(predictor_dat, "wb") as f_out:
            f_out.write(f_in.read())

        # Clean up the compressed file
        os.remove(predictor_bz2)
