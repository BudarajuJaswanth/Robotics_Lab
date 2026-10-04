"""
Utility script to generate printable ArUco markers PNG images using OpenCV.
"""

import os
import argparse
import cv2

def main():
    parser = argparse.ArgumentParser(description="Generate ArUco Marker Image PNG")
    parser.add_argument("--id", type=int, default=23, help="ArUco Marker ID (default: 23)")
    parser.add_argument("--size", type=int, default=400, help="Image size in pixels (default: 400)")
    parser.add_argument("--dict", type=str, default="DICT_6X6_250", help="ArUco Dictionary (default: DICT_6X6_250)")
    parser.add_argument("--output", type=str, default="data/aruco_marker_23.png", help="Output PNG path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    dict_id = getattr(cv2.aruco, args.dict, cv2.aruco.DICT_6X6_250)
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        marker_img = cv2.aruco.generateImageMarker(dictionary, args.id, args.size)
    else:
        dictionary = cv2.aruco.Dictionary_get(dict_id)
        marker_img = cv2.aruco.drawMarker(dictionary, args.id, args.size)

    cv2.imwrite(args.output, marker_img)
    print(f"✅ Generated ArUco Marker (Dict: {args.dict}, ID: {args.id}) -> {args.output}")

if __name__ == "__main__":
    main()
