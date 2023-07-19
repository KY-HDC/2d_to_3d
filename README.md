# Transition From 2D GANs to 3D GANs
---------
In this work, we conducted two re-implementations: one for Wasserstein Generative Adversarial Network (WGAN) using conv2d layers, and another for 3D-WGAN employing conv3d layers. 
Notice that this work is intended to provide a quick overview of 2D-GAN and 3D-GAN, not to implement state-of-the-art models or similar advanced approaches.

---------
Let's get started
---------

**2D-GAN** models are effective at generating fake images, but they are not as proficient as 3D-GANs in handling the sequential nature of CT scans. 
While 2D-GANs are capable of producing synthetic images, the power of 3D-GANs lies in their ability to preserve spatial relationships within volumetric CT data, resulting in more accurate and realistic image synthesis.
    
| CT-Fake-Image-Samples from 2DGAN |
| ------|

<p align="center">
    <img src="https://github.com/Harry-KIT/2d_to_3d/blob/main/assets/video_2d.gif?raw=true" width="240">
</p>

---------

 A single patient's CT scan consists of approximately 64 slices, where each slice represents a cross-sectional image of the body taken at different positions. 
 These slices collectively form a volumetric data set, providing a comprehensive 3D view of the patient's anatomy, which is crucial for precise medical diagnosis and treatment planning.

<p align="center">
    
<img src="https://github.com/Harry-KIT/2d_to_3d/blob/main/assets/ct_scans_for_one_patient.png?raw=true" width="440" height="211">

</p>
 
---------
**3D-GANs** are equipped with 3D convolutional layers, enabling them to handle and generate synthetic images as a sequence of CT scans properly. 
This capability is crucial, considering that each patient's CT scan typically consists of a sequence of slices taken at different positions, representing a three-dimensional volume.

<p align="center">
    
<img src="https://github.com/Harry-KIT/2d_to_3d/blob/main/assets/Figure_2.png?raw=true" width="440" height="200">

</p>

---------

By leveraging conv3d layers, 3D-GANs can capture the spatial and temporal relationships between adjacent slices in a CT scan, resulting in more accurate and realistic synthetic CT volumes. 
This is particularly valuable in medical imaging, where a comprehensive understanding of the 3D structure is essential for precise diagnosis, treatment planning, and analysis.

<p align="center">
    
<img src="https://github.com/Harry-KIT/2d_to_3d/blob/main/assets/1st_32.jpg?raw=true" width="960" height="200">

</p>
<p align="center">
    
<img src="https://github.com/Harry-KIT/2d_to_3d/blob/main/assets/2nd_32.jpg?raw=true" width="960" height="211">

</p>

In summary, 3D-GANs are well-suited for generating synthetic CT scans due to their ability to process and maintain the spatial continuity of volumetric data, making them a valuable tool 
for medical image synthesis and enhancing various tasks in the field of radiology and patient care.

| CT-Fake-Image-Samples from 3DGAN |
| ------|

<p align="center">
    <img src="https://github.com/Harry-KIT/2d_to_3d/blob/main/assets/video_3d.gif?raw=true" width="240">
</p>

---------

**References**
---------
* https://github.com/eriklindernoren/PyTorch-GAN
* https://github.com/cyclomon/3dbraingen
* https://github.com/hasibzunair/3D-image-classification-tutorial
* [DATASET](https://github.com/hasibzunair/3D-image-classification-tutorial/releases/tag/v0.2)
