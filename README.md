# FRC-Romi-Deployment-Models
This repository stores *Deep Learning* models and scripts that can be deployed on the Raspberry Pi **Romi** image. There are two deployment options.  One uses an attached [Luxonis OAK](https://shop.luxonis.com/products/1098obcenclosure) camera.  The OAK camera has an imbedded TPU the runs on the [Intel OpenVINO Toolkit](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html). The OAK camera's TPU runs the inference model, taking the processing off of the Raspberry Pi.  

The second option uses a Raspberry Pi/USB camera.  In this configuration the model inference is done on the Raspberry Pi.  Since the Pi doesn't have a GPU the detection frames per second FPS will be low.  This can be spead up by using a [Coral USB Accelerator](https://coral.ai/products/accelerator).  However, currently these are difficult to aquire.


Before deploying these models the Raspberry Pi must be installed with the [WPILibPi](https://github.com/wpilibsuite/WPILibPi/releases) Romi image.  A full description on how these models are created and deployed can be found on the [Team 2928 FRC Training](https://2928-frc-programmer-training.readthedocs.io/en/latest/MachineLearning/MLIndex/) site. The main files are included in this repository are as follows:

- `oak_yolo_spacial.py`  This is the default script that runs inference on a Yolo model and outputs detected objects with a label, bounding boxes and their X, Y, Z coordinates from the camera.  The script will display its output in a Web browser at `10.0.0.2:8091` and also places all of the data into the *WPILib* Network Tables.

- `oak_yolo.py`  This is a lighter version of the above script that only collects the label and bounding box data.

- `oak_yolo_spacial_gui.py`  This can be used on a laptop or other device that has a gui desktop.

- `tflite_yolo.py` This script is used to run inference with a Raspberry Pi or USB camera.  It uses the tflite deployment format.  The script displays its output in a Web browser at `10.0.0.2:8091` and also places data into the *WPILib* Network Tables.

- `rapid-react.blob` This model has been trained on the Rapid-React balls from the 2022 FIRST Competition. The blob file format is designed to run specifically on an *OpenVINO* device.

- `rapid-react-config.json` This is the configuration file needed to load the rapid-react model.  It includes the labels for the objects of interest. 

- `rapid-react.tflite` This model has been trained on the Rapid-React balls from the 2022 FIRST Competition. The `tflite` file format runs natively on the Raspberry Pi.


