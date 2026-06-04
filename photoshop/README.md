# Image Alignment Script for Photoshop

This script automatically aligns two photos of the same subject taken from slightly different angles, zoom levels, or lighting conditions. It uses Photoshop's built-in Auto-Align Layers feature.

## Requirements

- Adobe Photoshop (any recent version)

## Running the Script

1. In Photoshop, go to **File → Scripts → Browse...** and select the `align_images.jsx` file.  (In the future this could be streamlined by coping the scritp to the right folder that will enable it from Photoshop's native File -> Scripts menu.)  

2. A dialog will ask you to select the **Reference image** — this is the image that stays fixed. The other image will be warped to match it.

3. A second dialog will ask you to select the **image to align**.

4. The script will run automatically. When it finishes you will see two layers in the Layers panel:
   - **Reference** — the original reference image (bottom layer)
   - **To Align** — the aligned image at 75% opacity (top layer)

## After the Script Runs

- The **To Align** layer is set to 75% opacity so both images are visible simultaneously as an overlay.
- You can adjust the opacity using the **Opacity slider** in the Layers panel to compare the two images.
- Go to **File → Save As** to save the result.

## Notes

- The script does not modify or overwrite your original image files.
- For best results, the two images should be of the same subject with only minor differences in angle, zoom, or lighting.
- Processing time depends on image size — large files may take a minute or two.
