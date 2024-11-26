#Abstract

Recent advancements in deep learning have significantly propelled object detec-tion techniques, yet remote sensing image object detection remains challenging due to complex backgrounds, small-scale and densely arranged targets, and high
sample similarity. To address these issues, we propose an enhanced multi-scale attention mechanism tailored for robust remote sensing image object detection. Firstly, a fine-grained context feature extraction module is introduced, leverag-
ing context modeling to augment the capture of remote context information. Secondly, a novel Multi-Scale Channels and Spatial Attention (MSCPA) mod-ule is designed, capable of capturing long-range spatial interactions with precise
location information, facilitating accurate target localization. This module serves as a plug-and-play component, adaptable to any convolutional neural network for natural image object detection, and readily applicable to remote sensingimages. Lastly, a feature fusion structure guided by spatial context refinement
and perception is constructed, associating target features with global semantic information for more precise target recognition. Experiments are conducted onthree public remote sensing datasets: VisDrone2019, TinyPerson, and NWPU VHR-10.The results show that compared to the baseline model, the proposed
method improvesthe mAP@.5 detection accuracy on the three datasets by 10.3%, 4%, and 4.7%, respectively. Additionally, for the same datasets, the mAP@.95 detection accuracy improves by 6.3%, 1.8%, and 7.8%, respectively.

#Dataset
The three datasets used in this research can be downloaded from 
