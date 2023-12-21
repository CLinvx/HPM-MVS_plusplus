#ifndef _HPM_H_
#define _HPM_H_

#include "main.h"

int readDepthDmb(const std::string file_path, cv::Mat_<float> &depth);
int readNormalDmb(const std::string file_path, cv::Mat_<cv::Vec3f> &normal);
int writeDepthDmb(const std::string file_path, const cv::Mat_<float> depth);
int writeNormalDmb(const std::string file_path, const cv::Mat_<cv::Vec3f> normal);

Camera ReadCamera(const std::string &cam_path);
void  RescaleImageAndCamera(cv::Mat_<cv::Vec3b> &src, cv::Mat_<cv::Vec3b> &dst, cv::Mat_<float> &depth, Camera &camera);
void RescaleMask(cv::Mat_<cv::Vec3b>& src, cv::Mat_<cv::Vec3b>& dst, cv::Mat_<float>& depth);
float3 Get3DPointonWorld(const int x, const int y, const float depth, const Camera camera);
void ProjectonCamera(const float3 PointX, const Camera camera, float2 &point, float &depth);
float GetAngle(const cv::Vec3f &v1, const cv::Vec3f &v2);
void StoreColorPlyFileBinaryPointCloud (const std::string &plyFilePath, const std::vector<PointList> &pc);
void ExportPointCloud(const std::string& plyFilePath, const std::vector<PointList>& pc);
void RunJBU(const cv::Mat_<float>  &scaled_image_float, const cv::Mat_<float> &src_depthmap, const std::string &dense_folder , const Problem &problem);

#define CUDA_SAFE_CALL(error) CudaSafeCall(error, __FILE__, __LINE__)
#define CUDA_CHECK_ERROR() CudaCheckError(__FILE__, __LINE__)

void CudaSafeCall(const cudaError_t error, const std::string& file, const int line);
void CudaCheckError(const char* file, const int line);

struct cudaTextureObjects {
    cudaTextureObject_t images[MAX_IMAGES];
};

struct PatchMatchParams {
    int max_iterations = 3;
    int patch_size = 11;
    int num_images = 5;
    int max_image_size=3200;
    int radius_increment = 2;
    float sigma_spatial = 5.0f;
    float sigma_color = 3.0f;
    int top_k = 4;
    float baseline = 0.54f;
    float depth_min = 0.0f;
    float depth_max = 1.0f;
    float disparity_min = 0.0f;
    float disparity_max = 1.0f;

    float scaled_cols;
    float scaled_rows;

    bool geom_consistency = false;
    bool prior_consistency = false;
    bool multi_geometry = false;
    bool hierarchy = false;
    bool upsample = false;
    bool mand_consistency = false;
};

class HPM {
public:
    HPM();
    ~HPM();

    void InuputInitialization(const std::string &dense_folder, const std::vector<Problem> &problem, const int idx);
    void Colmap2MVS(const std::string &dense_folder, std::vector<Problem> &problems);
    void CudaSpaceInitialization(const std::string &dense_folder, const Problem &problem);
    void RunPatchMatch();
    void SetGeomConsistencyParams(bool multi_geometry);
    void SetPlanarPriorParams();
    void SetHierarchyParams();
    void SetMandConsistencyParams(bool flag);

    int GetReferenceImageWidth();
    int GetReferenceImageHeight();
    cv::Mat GetReferenceImage();
    float4 GetPlaneHypothesis(const int index);
    float GetCost(const int index);
    float GetTexture(const int index);
    void GetSupportPoints(std::vector<cv::Point>& support2DPoints);
    void GetSupportPoints_Double_Check(std::vector<cv::Point>& support2DPoints, const cv::Mat_<float>& costs, const cv::Mat_<float>& mand_consistency, float hpm_factor);
    void GetSupportPoints_Simple_Check(std::vector<cv::Point>& support2DPoints, const cv::Mat_<float>& costs, const cv::Mat_<float>& mand_consistency, float hpm_factor);
    void GetSupportPoints_Classify_Check(std::vector<cv::Point>& support2DPoints, const cv::Mat_<float>& costs, const cv::Mat_<float>& mand_consistency, const cv::Mat_<float>& texture, float hpm_factor);
    std::vector<Triangle> DelaunayTriangulation(const cv::Rect boundRC, const std::vector<cv::Point>& points);
    float4 GetPriorPlaneParams(const Triangle triangle, const cv::Mat_<float> depths);
    float4 GetPriorPlaneParams_factor(const Triangle triangle, const cv::Mat_<float> depths, float factor);
    float GetDepthFromPlaneParam(const float4 plane_hypothesis, const int x, const int y);
    float GetMinDepth();
    float GetMaxDepth();
    void CudaPlanarPriorInitialization(const std::vector<float4> &PlaneParams, const cv::Mat_<float> &masks);
    void CudaHypothesesReload(cv::Mat_ <float>depths, cv::Mat_<float>costs, cv::Mat_<cv::Vec3f>normals);
    void CudaConfidenceInitialization(const std::string& dense_folder, const std::vector<Problem>& problems, const int idx);
    void CudaCannyInitialization(const cv::Mat_<int>& Canny);

    void CudaPlanarPriorRelease();
    void CudaSpaceRelease(bool geom_consistency);
    void ReleaseProblemHostMemory();

    float4 TransformNormal(float4 plane_hypothesis);
    float GetDepthFromPlaneParam_factor(const float4 plane_hypothesis, const int x, const int y, float factor);
    void JointBilateralUpsampling_prior(const cv::Mat_<float>& scaled_image_float, const cv::Mat_<float>& src_depthmap, cv::Mat_<float>& upsample_depthmap, const cv::Mat_<cv::Vec3f>& src_normal, cv::Mat_<cv::Vec3f>& upsample_normal);
    float4 TransformNormal2RefCam(float4 plane_hypothesis);
    float GetDistance2Origin(const int2 p, const float depth, const float4 normal);
    void ReloadPlanarPriorInitialization(const cv::Mat_<float>& masks, float4* prior_plane_parameters);

    void TextureInformationInitialization();

private:
    int num_images;
    std::vector<cv::Mat> images;
    std::vector<cv::Mat> depths;
    std::vector<Camera> cameras;
    cudaTextureObjects texture_objects_host;
    cudaTextureObjects texture_depths_host;
    float4 *plane_hypotheses_host;
    float4 *scaled_plane_hypotheses_host;
    float *costs_host;
    float *pre_costs_host;
    float4 *prior_planes_host;
    unsigned int *plane_masks_host;
    PatchMatchParams params;
    float* confidences_host;
    float* texture_host;


    Camera *cameras_cuda;
    cudaArray *cuArray[MAX_IMAGES];
    cudaArray *cuDepthArray[MAX_IMAGES];
    cudaTextureObjects *texture_objects_cuda;
    cudaTextureObjects *texture_depths_cuda;
    float4 *plane_hypotheses_cuda;
    float4 *scaled_plane_hypotheses_cuda;
    float *costs_cuda;
    float *pre_costs_cuda;
    curandState *rand_states_cuda;
    unsigned int *selected_views_cuda;
    float* depths_cuda;
    float4* prior_planes_cuda;
    unsigned int* plane_masks_cuda;
    float* confidences_cuda;
    unsigned int* Canny_cuda;
    float* texture_cuda;
};

struct TexObj {
    cudaTextureObject_t imgs[MAX_IMAGES];
};

struct JBUParameters {
    int height;
    int width;
    int s_height;
    int s_width;
    int Imagescale;
};

struct JBUTexObj {
    cudaTextureObject_t imgs[JBU_NUM];
};

class JBU {
public:
    JBU();
    ~JBU();

    // Host Parameters
    float *depth_h;
    JBUTexObj jt_h;
    JBUParameters jp_h;

    // Device Parameters
    float *depth_d;
    cudaArray *cuArray[JBU_NUM]; // The first for reference image, and the second for stereo depth image
    JBUTexObj *jt_d;
    JBUParameters *jp_d;

    void InitializeParameters(int n);
    void CudaRun();
};

class JBU_prior {
public:
    JBU_prior();
    ~JBU_prior();

    // Host Parameters
    float* depth_h;
    float4* normal_h;
    JBUTexObj jt_h;
    JBUParameters jp_h;
    float4* normal_origin_host;

    // Device Parameters
    float* depth_d;
    cudaArray* cuArray[JBU_NUM]; // The first for reference image, and the second for stereo depth image
    JBUTexObj* jt_d;
    JBUParameters* jp_d;
    float4* normal_d;
    float4* normal_origin_cuda;

    void InitializeParameters_prior(int n, int origin_n);
    void CudaRun_prior();
    void ReleaseJBUCudaMemory_prior();
    void ReleaseJBUHostMemory_prior();
};


#endif // _ACMMP_H_
