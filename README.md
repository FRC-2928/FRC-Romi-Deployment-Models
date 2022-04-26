# FRC-OAK-Deployment-Models
This repository stores deep learning models for deployment on a Raspberry Pi with an attached [Luxonis OAK](https://shop.luxonis.com/products/1098obcenclosure) camera.  The OAK camera has an imbedded TPU the runs on the [Intel OpenVINO Toolkit](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html). Before deploying these models the Raspberry Pi must be installed with the [WPILibPi](https://github.com/wpilibsuite/WPILibPi/releases) Romi image.  A full description on how these models are created and deployed can be found on the [Team 2928 FRC Training](https://2928-frc-programmer-training.readthedocs.io/en/latest/MachineLearning/MLIndex/) site. The main files are included in this repository are as follows:

- `spacial_tiny_yolo_wpi.py`  This is the default script that runs inference on a YoloV3 model and outputs detected objects with a label, bounding boxes and their X, Y, Z coordinates from the camera.  The script will display its output in a Web browser at `10.0.0.2:8091` and also places all of the data into the *WPILib* Network Tables.

- `tiny_yolo_wpi.py`  This is a lighter version of the above script that only collects the label and bounding box data.

- `yolo-v3-tiny-tf_openvino_2021.4_6shave.blob` This is a model file that contains the weights and directed graph to detect 80 common objects.  The blob file format is designed to run specifically on an *OpenVINO* device.

- `custom.blob` This model has been trained on the Rapid-React balls from the 2022 FIRST Competition.


