# 2D-GAN에서 3D-GAN으로 전환

---------
**2D-WGAN** 은 생성된 적대적 신경망(GAN) 프레임워크의 변형으로, 생성된 데이터 분포와 실제 데이터 분포 사이의 불일치 척도로 Wasserstein 거리를 도입합니다. 
WGAN은 교육 중 모드 붕괴 및 불안정성과 같은 기존 GAN의 일부 제한 사항을 해결하는 것을 목표로 합니다.
    
| 2DGAN의 CT-가짜 이미지-샘플 |
| ------|

<p align="center">
    <img src="https://github.com/Harry-KIT/2d_to_3d/blob/main/assets/video_2d.gif?raw=true" width="240">
</p>
 
 
---------
**3D-WGAN** (3D Wasserstein Generative Adversarial Network)은 볼륨 이미지 또는 포인트 클라우드와 같은 3차원(3D) 데이터를 생성하기 위해 특별히 설계된 WGAN의 변형입니다. 
WGAN의 원칙을 3D 데이터 생성 영역으로 확장합니다. 3D-WGAN은 3차원 의료영상, 컴퓨터 그래픽, 가상현실 등 다양한 분야에 활용되고 있다.

<p align="center">
    
| 3DGAN의 CT-가짜 이미지-샘플 |
| ------|

</p>

<p align="center">
    <img src="https://github.com/Harry-KIT/2d_to_3d/blob/main/assets/video_3d.gif?raw=true" width="240">
</p>

---------

**References**
---------
* https://github.com/eriklindernoren/PyTorch-GAN
* https://github.com/cyclomon/3dbraingen
* https://github.com/hasibzunair/3D-image-classification-tutorial
* [데이터 세트](https://github.com/hasibzunair/3D-image-classification-tutorial/releases/tag/v0.2)
