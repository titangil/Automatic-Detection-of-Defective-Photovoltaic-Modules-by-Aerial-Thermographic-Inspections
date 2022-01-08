# Automatic Detection of Defective Photovoltaic Modules by Aerial Thermographic Inspections


1)	Introduction

   &emsp;Solar energy is being used more, but not that they only have advantages. The period of use of the solar panel must always be checked for defects, whether it is before distribution, transportation, installation, or after use. This has made many companies outsource to inspect PV modules at residences, industrial factories, and solar farms. 

   &emsp;In this project, the idea is to equip a drone with an external RGB camera and thermal camera module that angle perpendicular to the ground, a microprocessor for image processing and drone handling, and other power sources such as battery and power regulator. The drone will be programmed to be able to automatically navigate above solar field and record individual PV panel as an image and store it for post-processing afterward. The post-process operation begins after the drone landed; we will take the data into image classification through machine learning. By categorizing solar PV defects into three classifications, namely, hotspots, bypass circuits, and junction boxes. The output will be information containing irregularity data and the position of each PV panel.

  &emsp;This project can help reduce time and increase the frequency of the inspection.

2)	Equipment

&emsp;&emsp;&emsp;&emsp;&emsp;![image](https://user-images.githubusercontent.com/86349365/148644938-7b7d1b03-9abb-40e1-9fed-3e07e98f7866.png)

3)	Methodology

Our process will be divided into two parts, midflight operation, and post-processing. Midflight operation will cover navigation and image segmentation of photovoltaic panels while post-processing will process the segmented thermal image from the midflight operation.

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![image](https://user-images.githubusercontent.com/86349365/148644983-6d2508b7-ab68-438f-875a-fd17bf80f8e6.png)

&emsp;3.1	  Midflight Operation

&emsp;Midflight operation covers guided navigation above the solar array by utilizing image processing.
We have done a simulated model in ROS environment and Gazebo simulation, written in Python language. For communication, we used MavLink protocol on Ardupilot firmware.

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![image](https://user-images.githubusercontent.com/86349365/148645011-61da30be-32e2-4373-958f-d6705be6fbcc.png)

&emsp;3.2	 Post Processing

&emsp;As the drone aviate, it would capture thermal images along the solar array. We can use image processing and machine learning to segment and classify defect of each panel. 

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![image](https://user-images.githubusercontent.com/86349365/148645092-2e4cac9a-f7a3-46d0-b247-b37248c3f632.png)

