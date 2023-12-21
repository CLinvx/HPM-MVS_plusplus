# HPM-MVS++
* an enhanced version of HPM-MVS (HPM-MVS with Prior Consistency and Mandatory Consistency)
* the initial open source version for HPM-MVS++, maybe the code should be further optimized

## Dependencies
The code has been tested on Windows 10 with RTX 3070.<br />
 [cmake](https://cmake.org/)<br />
 [CUDA](https://developer.nvidia.com/cuda-toolkit) >= 6.0<br />
 [OpenCV](https://opencv.org/) >=2.4

## Useage
* Compile
```
mkdir build
cd build
cmake ..
make
```
* Test 
``` 
Use script colmap2mvsnet_acm.py to convert COLMAP SfM result to MVS input   
Run ./HPM-MVS_plusplus $data_folder true/flase(semantic segmentation masks for filtering sky area) to get reconstruction results 
```

## Citation
If you find our work useful in your research, please consider citing:
```
@InProceedings{Ren_2023_ICCV,
    author    = {Ren, Chunlin and Xu, Qingshan and Zhang, Shikun and Yang, Jiaqi},
    title     = {Hierarchical Prior Mining for Non-local Multi-View Stereo},
    booktitle = {Proc. IEEE/CVF International Conference on Computer Vision},
    month     = {October},
    year      = {2023},
    pages     = {3611-3620}
}
```

## Acknowledgemets
This code largely benefits from the following repositories: [ACMH](https://github.com/GhiXu/ACMH), [ACMP](https://github.com/GhiXu/ACMP), [ACMMP](https://github.com/GhiXu/ACMMP). Thanks to their authors for opening source of their excellent works.
