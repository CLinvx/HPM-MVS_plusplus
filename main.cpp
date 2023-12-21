#include "main.h"
#include "HPM.h"

void GenerateSampleList(const std::string& dense_folder, std::vector<Problem>& problems)
{
	std::string cluster_list_path = dense_folder + std::string("/pair.txt");

	problems.clear();

	std::ifstream file(cluster_list_path);

	int num_images;
	file >> num_images;

	for (int i = 0; i < num_images; ++i) {
		Problem problem;
		problem.src_image_ids.clear();
		file >> problem.ref_image_id;

		int num_src_images;
		file >> num_src_images;
		for (int j = 0; j < num_src_images; ++j) {
			int id;
			float score;
			file >> id >> score;
			if (score <= 0.0f) {
				continue;
			}
			problem.src_image_ids.push_back(id);
		}
		problems.push_back(problem);
	}
}

int ComputeMultiScaleSettings(const std::string& dense_folder, std::vector<Problem>& problems)
{
	int max_num_downscale = -1;
	int size_bound = 1000;
	PatchMatchParams pmp;
	std::string image_folder = dense_folder + std::string("/images");

	size_t num_images = problems.size();

	for (size_t i = 0; i < num_images; ++i) {
		std::stringstream image_path;
		image_path << image_folder << "/" << std::setw(8) << std::setfill('0') << problems[i].ref_image_id << ".jpg";
		cv::Mat_<uint8_t> image_uint = cv::imread(image_path.str(), cv::IMREAD_GRAYSCALE);

		int rows = image_uint.rows;
		int cols = image_uint.cols;
		int max_size = std::max(rows, cols);
		if (max_size > pmp.max_image_size) {
			max_size = pmp.max_image_size;
		}
		problems[i].max_image_size = max_size;

		int k = 0;
		while (max_size > size_bound) {
			max_size /= 2;
			k++;
		}

		if (k > max_num_downscale) {
			max_num_downscale = k;
		}

		problems[i].num_downscale = k;
	}

	return max_num_downscale;
}

void ProcessProblem(const std::string& dense_folder, const std::vector<Problem>& problems, const int idx, bool geom_consistency, bool prior_consistency, bool hierarchy, bool mand_consistency, int image_scale, bool multi_geometrty = false, int hpm_scale_distance = 0)
{
	const Problem problem = problems[idx];
	std::cout << "Processing image " << std::setw(8) << std::setfill('0') << problem.ref_image_id << "..." << std::endl;
	cudaSetDevice(0);
	std::stringstream result_path;
	result_path << dense_folder << "/HPM_MVS_plusplus" << "/2333_" << std::setw(8) << std::setfill('0') << problem.ref_image_id;
	std::string result_folder = result_path.str();
	mkdir(result_folder.c_str());

	HPM hpm;
	if (geom_consistency) {
		hpm.SetGeomConsistencyParams(multi_geometrty);
	}
	if (hierarchy) {
		hpm.SetHierarchyParams();
	}

	hpm.InuputInitialization(dense_folder, problems, idx);
	hpm.CudaSpaceInitialization(dense_folder, problem);

	const int width = hpm.GetReferenceImageWidth();
	const int height = hpm.GetReferenceImageHeight();

	cv::Mat_<float> depths = cv::Mat::zeros(height, width, CV_32FC1);
	cv::Mat_<cv::Vec3f> normals = cv::Mat::zeros(height, width, CV_32FC3);
	cv::Mat_<float> costs = cv::Mat::zeros(height, width, CV_32FC1);
	cv::Mat_<float>texture = cv::Mat::zeros(height, width, CV_32FC1);

	hpm.SetMandConsistencyParams(mand_consistency);

	std::stringstream canny_image_path;
	cv::Mat Image_grey;
	cv::Mat Canny_edge;
	std::string image_folder = dense_folder + std::string("/images");
	canny_image_path << image_folder << "/" << std::setw(8) << std::setfill('0') << problem.ref_image_id << ".jpg";
	cv::resize(cv::imread(canny_image_path.str(), 1), Image_grey, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
	cv::Canny(Image_grey, Canny_edge, 50, 150);
	std::cout << "Get Canny egdes down!" << std::endl;
	hpm.CudaCannyInitialization(Canny_edge);
	Image_grey.release();
	Canny_edge.release();

	if (mand_consistency || prior_consistency) {
		hpm.CudaConfidenceInitialization(dense_folder, problems, idx);
	}
	if (!prior_consistency && !mand_consistency) {
		hpm.TextureInformationInitialization();
	}

	if (!prior_consistency) {
		if (mand_consistency) {
			std::cout << "Run Mandatory Consistency ..." << std::endl;
		}
		else if (geom_consistency) {
			std::cout << "Run Geometric Consistency ..." << std::endl;
		}
		else {
			std::cout << "Run Photometric Consistency ..." << std::endl;
		}
		hpm.RunPatchMatch();
		for (int col = 0; col < width; ++col) {
			for (int row = 0; row < height; ++row) {
				int center = row * width + col;
				float4 plane_hypothesis = hpm.GetPlaneHypothesis(center);
				depths(row, col) = plane_hypothesis.w;
				normals(row, col) = cv::Vec3f(plane_hypothesis.x, plane_hypothesis.y, plane_hypothesis.z);
				costs(row, col) = hpm.GetCost(center);
				if (!mand_consistency) {
					texture(row, col) = hpm.GetTexture(center);
				}
			}
		}
	}
	else if (prior_consistency) {
		std::cout << "Run Prior Consistency ..." << std::endl;
		hpm.SetPlanarPriorParams();

		std::stringstream result_path;
		result_path << dense_folder << "/HPM_MVS_plusplus" << "/2333_" << std::setw(8) << std::setfill('0') << problem.ref_image_id;
		std::string result_folder = result_path.str();

		std::string suffix = "/depths.dmb";
		if (multi_geometrty) {
			suffix = "/depths_geom.dmb";
		}
		std::string depth_path = result_folder + suffix;
		std::string normal_path = result_folder + "/normals.dmb";
		std::string conf_path = result_folder + "/confidence.dmb";
		std::string cost_path = result_folder + "/costs.dmb";

		cv::Mat_<float>confidences;

		readDepthDmb(depth_path, depths);
		readNormalDmb(normal_path, normals);
		readDepthDmb(conf_path, confidences);
		readDepthDmb(cost_path, costs);


		if (hpm_scale_distance == 0) {
			const cv::Rect imageRC(0, 0, width, height);
			std::vector<cv::Point> support2DPoints;

			std::string texture_path = result_folder + "/texture" + std::to_string(image_scale) + ".dmb";
			cv::Mat_<float>textures;
			readDepthDmb(texture_path, textures);

			hpm.GetSupportPoints_Classify_Check(support2DPoints, costs, confidences, textures, 1);
			const auto triangles = hpm.DelaunayTriangulation(imageRC, support2DPoints);

			cv::Mat refImage = hpm.GetReferenceImage().clone();
			std::vector<cv::Mat> mbgr(3);
			mbgr[0] = refImage.clone();
			mbgr[1] = refImage.clone();
			mbgr[2] = refImage.clone();
			cv::Mat srcImage;
			cv::merge(mbgr, srcImage);
			for (const auto triangle : triangles) {
				if (imageRC.contains(triangle.pt1) && imageRC.contains(triangle.pt2) && imageRC.contains(triangle.pt3)) {
					cv::line(srcImage, triangle.pt1, triangle.pt2, cv::Scalar(0, 0, 255));
					cv::line(srcImage, triangle.pt1, triangle.pt3, cv::Scalar(0, 0, 255));
					cv::line(srcImage, triangle.pt2, triangle.pt3, cv::Scalar(0, 0, 255));
				}
			}
			std::string triangulation_path = result_folder + "/triangulation0.png";
			cv::imwrite(triangulation_path, srcImage);

			refImage.release();
			mbgr.clear();
			mbgr.shrink_to_fit();
			srcImage.release();

			cv::Mat_<float> mask_tri = cv::Mat::zeros(height, width, CV_32FC1);
			std::vector<float4> planeParams_tri;
			planeParams_tri.clear();

			uint32_t idx = 0;
			for (const auto triangle : triangles) {
				if (imageRC.contains(triangle.pt1) && imageRC.contains(triangle.pt2) && imageRC.contains(triangle.pt3)) {
					float L01 = sqrt(pow(triangle.pt1.x - triangle.pt2.x, 2) + pow(triangle.pt1.y - triangle.pt2.y, 2));
					float L02 = sqrt(pow(triangle.pt1.x - triangle.pt3.x, 2) + pow(triangle.pt1.y - triangle.pt3.y, 2));
					float L12 = sqrt(pow(triangle.pt2.x - triangle.pt3.x, 2) + pow(triangle.pt2.y - triangle.pt3.y, 2));

					float max_edge_length = std::max(L01, std::max(L02, L12));
					float step = 1.0 / max_edge_length;

					for (float p = 0; p < 1.0; p += step) {
						for (float q = 0; q < 1.0 - p; q += step) {
							int x = p * triangle.pt1.x + q * triangle.pt2.x + (1.0 - p - q) * triangle.pt3.x;
							int y = p * triangle.pt1.y + q * triangle.pt2.y + (1.0 - p - q) * triangle.pt3.y;
							mask_tri(y, x) = idx + 1.0; // To distinguish from the label of non-triangulated areas
						}
					}
					float4 n4 = hpm.GetPriorPlaneParams(triangle, depths);
					planeParams_tri.push_back(n4);
					idx++;
				}
			}

			cv::Mat_<float> priordepths = cv::Mat::zeros(height, width, CV_32FC1);
			for (int i = 0; i < width; ++i) {
				for (int j = 0; j < height; ++j) {
					if (mask_tri(j, i) > 0) {
						float d = hpm.GetDepthFromPlaneParam(planeParams_tri[mask_tri(j, i) - 1], i, j);
						if (d <= hpm.GetMaxDepth() && d >= hpm.GetMinDepth()) {
							priordepths(j, i) = d;
						}
						else {
							mask_tri(j, i) = 0;
						}
					}
				}
			}
			std::string prior_path = result_folder + "/depths_prior0.dmb";
			writeDepthDmb(prior_path, priordepths);
			priordepths.release();

			hpm.CudaPlanarPriorInitialization(planeParams_tri, mask_tri);
			hpm.CudaHypothesesReload(depths, costs, normals);
			hpm.RunPatchMatch();
			textures.release();
			mask_tri.release();
			planeParams_tri.clear();
			planeParams_tri.shrink_to_fit();
			support2DPoints.clear();
			support2DPoints.shrink_to_fit();
			planeParams_tri.clear();
			planeParams_tri.shrink_to_fit();
		}
		else if (hpm_scale_distance == 1 || hpm_scale_distance == 2) {
			float hpm_factor = 1.0 / (hpm_scale_distance * 2);
			int hpm_width = std::round(width * hpm_factor);
			int hpm_height = std::round(height * hpm_factor);

			cv::Mat_<float>depths_downsample;
			cv::Mat_<float>costs_downsample;
			cv::Mat_<float>confidences_downsample;

			cv::resize(depths, depths_downsample, cv::Size(hpm_width, hpm_height), 0, 0, cv::INTER_LINEAR);
			cv::resize(costs, costs_downsample, cv::Size(hpm_width, hpm_height), 0, 0, cv::INTER_LINEAR);
			cv::resize(confidences, confidences_downsample, cv::Size(hpm_width, hpm_height), 0, 0, cv::INTER_LINEAR);



			std::string texture_path = result_folder + "/texture" + std::to_string(image_scale + hpm_scale_distance) + ".dmb";
			cv::Mat_<float>textures;
			readDepthDmb(texture_path, textures);

			const cv::Rect imageRC(0, 0, hpm_width, hpm_height);
			std::vector<cv::Point> support2DPoints;
			support2DPoints.clear();
			hpm.GetSupportPoints_Classify_Check(support2DPoints, costs_downsample, confidences_downsample, textures, hpm_factor);

			const auto triangles = hpm.DelaunayTriangulation(imageRC, support2DPoints);

			cv::Mat refImage;
			cv::resize(hpm.GetReferenceImage().clone(), refImage, cv::Size(hpm_width, hpm_height), 0, 0, cv::INTER_LINEAR);
			std::vector<cv::Mat> mbgr(3);
			mbgr[0] = refImage.clone();
			mbgr[1] = refImage.clone();
			mbgr[2] = refImage.clone();
			cv::Mat srcImage;
			cv::merge(mbgr, srcImage);
			for (const auto triangle : triangles) {
				if (imageRC.contains(triangle.pt1) && imageRC.contains(triangle.pt2) && imageRC.contains(triangle.pt3)) {
					cv::line(srcImage, triangle.pt1, triangle.pt2, cv::Scalar(0, 0, 255));
					cv::line(srcImage, triangle.pt1, triangle.pt3, cv::Scalar(0, 0, 255));
					cv::line(srcImage, triangle.pt2, triangle.pt3, cv::Scalar(0, 0, 255));
				}
			}
			std::string triangulation_path = result_folder + "/triangulation" + std::to_string(hpm_scale_distance) + ".png";
			cv::imwrite(triangulation_path, srcImage);

			std::vector<float4>planeParams_tri;
			cv::Mat_<float> mask_tri = cv::Mat::zeros(hpm_height, hpm_width, CV_32FC1);
			planeParams_tri.clear();
			uint32_t idx = 0;
			for (const auto triangle : triangles) {
				if (imageRC.contains(triangle.pt1) && imageRC.contains(triangle.pt2) && imageRC.contains(triangle.pt3)) {
					float L01 = sqrt(pow(triangle.pt1.x - triangle.pt2.x, 2) + pow(triangle.pt1.y - triangle.pt2.y, 2));
					float L02 = sqrt(pow(triangle.pt1.x - triangle.pt3.x, 2) + pow(triangle.pt1.y - triangle.pt3.y, 2));
					float L12 = sqrt(pow(triangle.pt2.x - triangle.pt3.x, 2) + pow(triangle.pt2.y - triangle.pt3.y, 2));

					float max_edge_length = std::max(L01, std::max(L02, L12));
					float step = 1.0 / max_edge_length;

					for (float p = 0; p < 1.0; p += step) {
						for (float q = 0; q < 1.0 - p; q += step) {
							int x = p * triangle.pt1.x + q * triangle.pt2.x + (1.0 - p - q) * triangle.pt3.x;
							int y = p * triangle.pt1.y + q * triangle.pt2.y + (1.0 - p - q) * triangle.pt3.y;
							mask_tri(y, x) = idx + 1.0; // To distinguish from the label of non-triangulated areas
						}
					}
					//renew the camera's parameters 
					float4 n4 = hpm.GetPriorPlaneParams_factor(triangle, depths_downsample, hpm_factor);
					planeParams_tri.push_back(n4);
					idx++;
				}
			}

			cv::Mat_<float>priordepths = cv::Mat::zeros(hpm_height, hpm_width, CV_32FC1);
			cv::Mat_<cv::Vec3f>priornormals = cv::Mat::zeros(hpm_height, hpm_width, CV_32FC3);

			for (int i = 0; i < hpm_width; ++i) {
				for (int j = 0; j < hpm_height; ++j) {
					if (mask_tri(j, i) > 0) {
						float d = hpm.GetDepthFromPlaneParam_factor(planeParams_tri[mask_tri(j, i) - 1], i, j, hpm_factor);
						if (d <= hpm.GetMaxDepth() * 1.2f && d >= hpm.GetMinDepth() * 0.6f) {
							priordepths(j, i) = d;
							float4 tmp_n4 = hpm.TransformNormal(planeParams_tri[mask_tri(j, i) - 1]);
							priornormals(j, i)[0] = tmp_n4.x;
							priornormals(j, i)[1] = tmp_n4.y;
							priornormals(j, i)[2] = tmp_n4.z;
						}
						else {
							mask_tri(j, i) = 0;
						}
					}
				}
			}

			std::string depths_prior_path = result_folder + "/depths_prior" + std::to_string(hpm_scale_distance) + ".dmb";
			writeDepthDmb(depths_prior_path, priordepths);

			std::stringstream image_path;
			image_path << dense_folder << "/images" << "/" << std::setw(8) << std::setfill('0') << problem.ref_image_id << ".jpg";
			cv::Mat_<uint8_t> image_uint;
			cv::resize(cv::imread(image_path.str(), cv::IMREAD_GRAYSCALE), image_uint, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
			cv::Mat image_float;
			image_uint.convertTo(image_float, CV_32FC1);
			cv::Mat_<float>priordepths_upsample = cv::Mat::zeros(height, width, CV_32FC1);
			cv::Mat_ < cv::Vec3f >priornormals_upsample = cv::Mat::zeros(height, width, CV_32FC3);
			std::cout << "Running JBU..." << std::endl;
			hpm.JointBilateralUpsampling_prior(image_float, priordepths, priordepths_upsample, priornormals, priornormals_upsample);

			cv::Mat_<float>mask_tri_new = cv::Mat::zeros(height, width, CV_32FC1);
			float4* prior_planeParams = new float4[height * width];
			for (int i = 0; i < width; i++) {
				for (int j = 0; j < height; j++) {
					if (priordepths_upsample(j, i) <= hpm.GetMaxDepth() && priordepths_upsample(j, i) >= hpm.GetMinDepth()) {
						mask_tri_new(j, i) = 1;
						int center = j * width + i;
						float4 tmp_reload;
						tmp_reload.x = priornormals_upsample(j, i)[0];
						tmp_reload.y = priornormals_upsample(j, i)[1];
						tmp_reload.z = priornormals_upsample(j, i)[2];
						tmp_reload.w = priordepths_upsample(j, i);
						tmp_reload = hpm.TransformNormal2RefCam(tmp_reload);
						float depth_now = tmp_reload.w;
						int2 p = make_int2(i, j);
						tmp_reload.w = hpm.GetDistance2Origin(p, depth_now, tmp_reload);
						prior_planeParams[center] = tmp_reload;
					}
					else {
						mask_tri_new(j, i) = 0;
					}
				}
			}

			depths_prior_path = result_folder + "/depths_prior" + std::to_string(hpm_scale_distance) + "_upsample.dmb";
			writeDepthDmb(depths_prior_path, priordepths_upsample);
			hpm.ReloadPlanarPriorInitialization(mask_tri_new, prior_planeParams);
			hpm.CudaHypothesesReload(depths, costs, normals);
			hpm.RunPatchMatch();

			refImage.release();
			mbgr.clear();
			mbgr.shrink_to_fit();
			srcImage.release();
			textures.release();
			support2DPoints.clear();
			support2DPoints.shrink_to_fit();
			depths_downsample.release();
			costs_downsample.release();
			confidences_downsample.release();
			priordepths.release();
			priornormals.release();
			image_uint.release();
			image_float.release();
			priordepths_upsample.release();
			priornormals_upsample.release();
			mask_tri_new.release();
			delete(prior_planeParams);
		}

		for (int col = 0; col < width; ++col) {
			for (int row = 0; row < height; ++row) {
				int center = row * width + col;
				float4 plane_hypothesis = hpm.GetPlaneHypothesis(center);
				depths(row, col) = plane_hypothesis.w;
				normals(row, col) = cv::Vec3f(plane_hypothesis.x, plane_hypothesis.y, plane_hypothesis.z);
				costs(row, col) = hpm.GetCost(center);
			}
		}

		hpm.CudaPlanarPriorRelease();
	}

	std::string suffix = "/depths.dmb";
	if (geom_consistency) {
		suffix = "/depths_geom.dmb";
	}
	std::string depth_path = result_folder + suffix;
	std::string normal_path = result_folder + "/normals.dmb";
	std::string cost_path = result_folder + "/costs.dmb";

	writeDepthDmb(depth_path, depths);
	writeNormalDmb(normal_path, normals);
	writeDepthDmb(cost_path, costs);
	if (!mand_consistency && !prior_consistency) {
		std::string texture_path = result_folder + "/texture" + std::to_string(image_scale) + ".dmb";
		writeDepthDmb(texture_path, texture);
	}
	texture.release();
	depths.release();
	normals.release();
	costs.release();
	hpm.CudaSpaceRelease(geom_consistency);
	hpm.ReleaseProblemHostMemory();
	std::cout << "Processing image " << std::setw(8) << std::setfill('0') << problem.ref_image_id << " done!" << std::endl;
}

void JointBilateralUpsampling(const std::string& dense_folder, const Problem& problem, int acmmp_size)
{
	std::stringstream result_path;
	result_path << dense_folder << "/HPM_MVS_plusplus" << "/2333_" << std::setw(8) << std::setfill('0') << problem.ref_image_id;
	std::string result_folder = result_path.str();
	std::string depth_path = result_folder + "/depths_geom.dmb";
	cv::Mat_<float> ref_depth;
	readDepthDmb(depth_path, ref_depth);

	std::string image_folder = dense_folder + std::string("/images");
	std::stringstream image_path;
	image_path << image_folder << "/" << std::setw(8) << std::setfill('0') << problem.ref_image_id << ".jpg";
	cv::Mat_<uint8_t> image_uint = cv::imread(image_path.str(), cv::IMREAD_GRAYSCALE);
	cv::Mat image_float;
	image_uint.convertTo(image_float, CV_32FC1);
	const float factor_x = static_cast<float>(acmmp_size) / image_float.cols;
	const float factor_y = static_cast<float>(acmmp_size) / image_float.rows;
	const float factor = std::min(factor_x, factor_y);

	const int new_cols = std::round(image_float.cols * factor);
	const int new_rows = std::round(image_float.rows * factor);
	cv::Mat scaled_image_float;
	cv::resize(image_float, scaled_image_float, cv::Size(new_cols, new_rows), 0, 0, cv::INTER_LINEAR);

	std::cout << "Run JBU for image " << problem.ref_image_id << ".jpg" << std::endl;
	RunJBU(scaled_image_float, ref_depth, dense_folder, problem);
}

void RunFusion_Sky_Strict(std::string& dense_folder, const std::vector<Problem>& problems, bool geom_consistency)
{
	size_t num_images = problems.size();
	std::string image_folder = dense_folder + std::string("/images");
	std::string cam_folder = dense_folder + std::string("/cams");
	std::string mask_folder = dense_folder + std::string("/masks");

	std::vector<cv::Mat> images;
	std::vector<Camera> cameras;
	std::vector<cv::Mat_<float>> depths;
	std::vector<cv::Mat_<cv::Vec3f>> normals;
	std::vector<cv::Mat> masks;
	std::vector<cv::Mat> sky_masks;
	images.clear();
	cameras.clear();
	depths.clear();
	normals.clear();
	masks.clear();
	sky_masks.clear();

	for (size_t i = 0; i < num_images; ++i) {
		std::cout << "Reading image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;

		std::stringstream image_path;
		image_path << image_folder << "/" << std::setw(8) << std::setfill('0') << problems[i].ref_image_id << ".jpg";
		cv::Mat_<cv::Vec3b> image = cv::imread(image_path.str(), cv::IMREAD_COLOR);

		std::stringstream sky_mask_path;
		sky_mask_path << mask_folder << "/" << std::setw(8) << std::setfill('0') << problems[i].ref_image_id << ".jpg";
		cv::Mat_<cv::Vec3b> sky_mask = cv::imread(sky_mask_path.str(), cv::IMREAD_COLOR);

		std::stringstream cam_path;
		cam_path << cam_folder << "/" << std::setw(8) << std::setfill('0') << problems[i].ref_image_id << "_cam.txt";
		Camera camera = ReadCamera(cam_path.str());

		std::stringstream result_path;
		result_path << dense_folder << "/HPM_MVS_plusplus" << "/2333_" << std::setw(8) << std::setfill('0') << problems[i].ref_image_id;
		std::string result_folder = result_path.str();
		std::string suffix = "/depths.dmb";
		if (geom_consistency) {
			suffix = "/depths_geom.dmb";
		}
		std::string depth_path = result_folder + suffix;
		std::string normal_path = result_folder + "/normals.dmb";
		cv::Mat_<float> depth;
		cv::Mat_<cv::Vec3f> normal;
		readDepthDmb(depth_path, depth);
		readNormalDmb(normal_path, normal);

		cv::Mat_<cv::Vec3b> scaled_image;
		RescaleImageAndCamera(image, scaled_image, depth, camera);
		cv::Mat_<cv::Vec3b>scaled_sky_mask;
		RescaleMask(sky_mask, scaled_sky_mask, depth);
		
		cv::Mat mask = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);

		images.push_back(scaled_image);
		cameras.push_back(camera);
		depths.push_back(depth);
		normals.push_back(normal);
		masks.push_back(mask);
		sky_masks.push_back(scaled_sky_mask);
		image.release();
		sky_mask.release();
		depth.release();
		normal.release();
		scaled_image.release();
		mask.release();
	}

	std::vector<PointList> PointCloud;
	PointCloud.clear();

	for (size_t i = 0; i < num_images; ++i) {
		std::cout << "Fusing image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;
		const int cols = depths[i].cols;
		const int rows = depths[i].rows;
		int num_ngb = problems[i].src_image_ids.size();
		std::vector<int2> used_list(num_ngb, make_int2(-1, -1));
		for (int r = 0; r < rows; ++r) {
			for (int c = 0; c < cols; ++c) {
				bool view_strict = false;
				if (masks[i].at<uchar>(r, c) == 1)
					continue;
				float ref_depth = depths[i].at<float>(r, c);
				cv::Vec3f ref_normal = normals[i].at<cv::Vec3f>(r, c);

				if (ref_depth <= 0.0)
					continue;

				float3 PointX = Get3DPointonWorld(c, r, ref_depth, cameras[i]);
				float3 consistent_Point = PointX;
				cv::Vec3f consistent_normal = ref_normal;
				float consistent_Color[3] = { (float)images[i].at<cv::Vec3b>(r, c)[0], (float)images[i].at<cv::Vec3b>(r, c)[1], (float)images[i].at<cv::Vec3b>(r, c)[2] };
				float segment_Color[3] = { (float)sky_masks[i].at<cv::Vec3b>(r, c)[0], (float)sky_masks[i].at<cv::Vec3b>(r, c)[1], (float)sky_masks[i].at<cv::Vec3b>(r, c)[2] };
				int num_consistent = 0;
				float dynamic_consistency = 0;

				for (int j = 0; j < num_ngb; ++j) {
					int src_id = problems[i].src_image_ids[j];
					const int src_cols = depths[src_id].cols;
					const int src_rows = depths[src_id].rows;
					float2 point;
					float proj_depth;
					ProjectonCamera(PointX, cameras[src_id], point, proj_depth);
					int src_r = int(point.y + 0.5f);
					int src_c = int(point.x + 0.5f);
					if (src_c >= 0 && src_c < src_cols && src_r >= 0 && src_r < src_rows) {
						if (masks[src_id].at<uchar>(src_r, src_c) == 1)
							continue;

						float src_depth = depths[src_id].at<float>(src_r, src_c);
						cv::Vec3f src_normal = normals[src_id].at<cv::Vec3f>(src_r, src_c);
						if (src_depth <= 0.0)
							continue;

						float3 tmp_X = Get3DPointonWorld(src_c, src_r, src_depth, cameras[src_id]);
						float2 tmp_pt;
						ProjectonCamera(tmp_X, cameras[i], tmp_pt, proj_depth);
						float reproj_error = sqrt(pow(c - tmp_pt.x, 2) + pow(r - tmp_pt.y, 2));
						float relative_depth_diff = fabs(proj_depth - ref_depth) / ref_depth;
						float angle = GetAngle(ref_normal, src_normal);

						if (reproj_error < 2.0f && relative_depth_diff < 0.01f && angle < 0.174533f) {
							used_list[j].x = src_c;
							used_list[j].y = src_r;

							float tmp_index = reproj_error + 200 * relative_depth_diff + angle * 10;
							float cons = exp(-tmp_index);
							dynamic_consistency += exp(-tmp_index);
							num_consistent++;
						}
					}
				}

				int view_num = 1;
				float factor = 0.3;


				if (num_consistent >= view_num && (dynamic_consistency > factor * num_consistent)) {
					PointList point3D;
					point3D.coord = consistent_Point;
					point3D.color = make_float3(consistent_Color[0], consistent_Color[1], consistent_Color[2]);
					
					float3 seg_Color = make_float3(segment_Color[0], segment_Color[1], segment_Color[2]);
					if ((int)seg_Color.x != 234 && (int)seg_Color.y != 235 && (int)seg_Color.z != 55) {
						PointCloud.push_back(point3D);
					}
					for (int j = 0; j < num_ngb; ++j) {
						if (used_list[j].x == -1)
							continue;
						masks[problems[i].src_image_ids[j]].at<uchar>(used_list[j].y, used_list[j].x) = 1;
					}
				}
			}
		}
	}

	std::string ply_path = dense_folder + "/HPM_MVS_plusplus/HPM_MVS_plusplus_mask.ply";
	ExportPointCloud(ply_path, PointCloud);
}

void RunFusion(std::string& dense_folder, const std::vector<Problem>& problems, bool geom_consistency)
{
	size_t num_images = problems.size();
	std::string image_folder = dense_folder + std::string("/images");
	std::string cam_folder = dense_folder + std::string("/cams");

	std::vector<cv::Mat> images;
	std::vector<Camera> cameras;
	std::vector<cv::Mat_<float>> depths;
	std::vector<cv::Mat_<cv::Vec3f>> normals;
	std::vector<cv::Mat> masks;
	images.clear();
	cameras.clear();
	depths.clear();
	normals.clear();
	masks.clear();

	for (size_t i = 0; i < num_images; ++i) {
		std::cout << "Reading image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;
		std::stringstream image_path;
		image_path << image_folder << "/" << std::setw(8) << std::setfill('0') << problems[i].ref_image_id << ".jpg";
		cv::Mat_<cv::Vec3b> image = cv::imread(image_path.str(), cv::IMREAD_COLOR);
		std::stringstream cam_path;
		cam_path << cam_folder << "/" << std::setw(8) << std::setfill('0') << problems[i].ref_image_id << "_cam.txt";
		Camera camera = ReadCamera(cam_path.str());

		std::stringstream result_path;
		result_path << dense_folder << "/HPM_MVS_plusplus" << "/2333_" << std::setw(8) << std::setfill('0') << problems[i].ref_image_id;
		std::string result_folder = result_path.str();
		std::string suffix = "/depths.dmb";
		if (geom_consistency) {
			suffix = "/depths_geom.dmb";
		}
		std::string depth_path = result_folder + suffix;
		std::string normal_path = result_folder + "/normals.dmb";
		cv::Mat_<float> depth;
		cv::Mat_<cv::Vec3f> normal;
		readDepthDmb(depth_path, depth);
		readNormalDmb(normal_path, normal);

		cv::Mat_<cv::Vec3b> scaled_image;
		RescaleImageAndCamera(image, scaled_image, depth, camera);
		images.push_back(scaled_image);
		cameras.push_back(camera);
		depths.push_back(depth);
		normals.push_back(normal);
		cv::Mat mask = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
		masks.push_back(mask);


	}

	std::vector<PointList> PointCloud;
	PointCloud.clear();

	for (size_t i = 0; i < num_images; ++i) {
		std::cout << "Fusing image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;
		const int cols = depths[i].cols;
		const int rows = depths[i].rows;
		int num_ngb = problems[i].src_image_ids.size();
		std::vector<int2> used_list(num_ngb, make_int2(-1, -1));
		for (int r = 0; r < rows; ++r) {
			for (int c = 0; c < cols; ++c) {
				if (masks[i].at<uchar>(r, c) == 1)
					continue;
				float ref_depth = depths[i].at<float>(r, c);
				cv::Vec3f ref_normal = normals[i].at<cv::Vec3f>(r, c);

				if (ref_depth <= 0.0)
					continue;

				float3 PointX = Get3DPointonWorld(c, r, ref_depth, cameras[i]);
				float3 consistent_Point = PointX;
				cv::Vec3f consistent_normal = ref_normal;
				float consistent_Color[3] = { (float)images[i].at<cv::Vec3b>(r, c)[0], (float)images[i].at<cv::Vec3b>(r, c)[1], (float)images[i].at<cv::Vec3b>(r, c)[2] };
				int num_consistent = 0;
				float dynamic_consistency = 0;

				for (int j = 0; j < num_ngb; ++j) {
					int src_id = problems[i].src_image_ids[j];
					const int src_cols = depths[src_id].cols;
					const int src_rows = depths[src_id].rows;
					float2 point;
					float proj_depth;
					ProjectonCamera(PointX, cameras[src_id], point, proj_depth);
					int src_r = int(point.y + 0.5f);
					int src_c = int(point.x + 0.5f);
					if (src_c >= 0 && src_c < src_cols && src_r >= 0 && src_r < src_rows) {
						if (masks[src_id].at<uchar>(src_r, src_c) == 1)
							continue;

						float src_depth = depths[src_id].at<float>(src_r, src_c);
						cv::Vec3f src_normal = normals[src_id].at<cv::Vec3f>(src_r, src_c);
						if (src_depth <= 0.0)
							continue;

						float3 tmp_X = Get3DPointonWorld(src_c, src_r, src_depth, cameras[src_id]);
						float2 tmp_pt;
						ProjectonCamera(tmp_X, cameras[i], tmp_pt, proj_depth);
						float reproj_error = sqrt(pow(c - tmp_pt.x, 2) + pow(r - tmp_pt.y, 2));
						float relative_depth_diff = fabs(proj_depth - ref_depth) / ref_depth;
						float angle = GetAngle(ref_normal, src_normal);

						if (reproj_error < 2.0f && relative_depth_diff < 0.01f && angle < 0.174533f) {
							used_list[j].x = src_c;
							used_list[j].y = src_r;

							float tmp_index = reproj_error + 200 * relative_depth_diff + angle * 10;
							float cons = exp(-tmp_index);
							dynamic_consistency += exp(-tmp_index);
							num_consistent++;
						}
					}
				}

				if (num_consistent >= 1 && (dynamic_consistency > 0.3 * num_consistent)) {
					PointList point3D;
					point3D.coord = consistent_Point;
					point3D.normal = make_float3(consistent_normal[0], consistent_normal[1], consistent_normal[2]);
					point3D.color = make_float3(consistent_Color[0], consistent_Color[1], consistent_Color[2]);
					PointCloud.push_back(point3D);
					for (int j = 0; j < num_ngb; ++j) {
						if (used_list[j].x == -1)
							continue;
						masks[problems[i].src_image_ids[j]].at<uchar>(used_list[j].y, used_list[j].x) = 1;
					}
				}
			}
		}
	}

	std::string ply_path = dense_folder + "/HPM_MVS_plusplus/HPM_MVS_plusplus.ply";
	ExportPointCloud(ply_path, PointCloud);
}

void ConfidenceEvaluation(std::string& dense_folder, const std::vector<Problem>& problems, bool geom_consistency) {
	size_t num_images = problems.size();
	std::string image_folder = dense_folder + std::string("/images");
	std::string cam_folder = dense_folder + std::string("/cams");

	std::vector<Camera> cameras;
	std::vector<cv::Mat_<float>> depths;
	std::vector<cv::Mat_<cv::Vec3f>> normals;
	std::vector<cv::Mat> masks;
	std::vector<cv::Mat_<float>>consistency;
	cameras.clear();
	depths.clear();
	normals.clear();
	masks.clear();
	consistency.clear();
	for (size_t i = 0; i < num_images; ++i) {
		std::cout << "Reading image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;
		std::stringstream image_path;
		image_path << image_folder << "/" << std::setw(8) << std::setfill('0') << problems[i].ref_image_id << ".jpg";
		cv::Mat_<cv::Vec3b> image = cv::imread(image_path.str(), cv::IMREAD_COLOR);
		std::stringstream cam_path;
		cam_path << cam_folder << "/" << std::setw(8) << std::setfill('0') << problems[i].ref_image_id << "_cam.txt";
		Camera camera = ReadCamera(cam_path.str());
		std::stringstream result_path;
		result_path << dense_folder << "/HPM_MVS_plusplus" << "/2333_" << std::setw(8) << std::setfill('0') << problems[i].ref_image_id;
		std::string result_folder = result_path.str();
		std::string suffix = "/depths.dmb";
		if (geom_consistency) {
			suffix = "/depths_geom.dmb";
		}
		std::string depth_path = result_folder + suffix;
		std::string normal_path = result_folder + "/normals.dmb";
		cv::Mat_<float> depth;
		cv::Mat_<cv::Vec3f> normal;
		readDepthDmb(depth_path, depth);
		readNormalDmb(normal_path, normal);

		cv::Mat_<cv::Vec3b> scaled_image;
		RescaleImageAndCamera(image, scaled_image, depth, camera);
		cameras.push_back(camera);
		depths.push_back(depth);
		normals.push_back(normal);
		cv::Mat mask = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
		masks.push_back(mask);
		cv::Mat consist = cv::Mat::zeros(depth.rows, depth.cols, CV_32FC1);
		consistency.push_back(consist);
		depth.release();
		normal.release();
		mask.release();
		consist.release();
		image.release();
		scaled_image.release();

	}
	for (size_t i = 0; i < num_images; ++i) {
		std::cout << "Hypothesis Confidence Evaluating Image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;
		const int cols = depths[i].cols;
		const int rows = depths[i].rows;
		int num_ngb = problems[i].src_image_ids.size();
		std::vector<int2> used_list(num_ngb, make_int2(-1, -1));
		for (int r = 0; r < rows; ++r) {
			for (int c = 0; c < cols; ++c) {
				if (masks[i].at<uchar>(r, c) == 1)
					continue;
				float ref_depth = depths[i].at<float>(r, c);
				cv::Vec3f ref_normal = normals[i].at<cv::Vec3f>(r, c);

				if (ref_depth <= 0.0) {
					continue;
				}
				if (ref_depth < cameras[i].depth_min || ref_depth > cameras[i].depth_max) {
					continue;
				}

				float3 PointX = Get3DPointonWorld(c, r, ref_depth, cameras[i]);
				float3 consistent_Point = PointX;
				cv::Vec3f consistent_normal = ref_normal;
				int num_consistent = 0;
				float dynamic_consistency = 0;

				for (int j = 0; j < num_ngb; ++j) {
					int src_id = problems[i].src_image_ids[j];
					const int src_cols = depths[src_id].cols;
					const int src_rows = depths[src_id].rows;
					float2 point;
					float proj_depth;
					ProjectonCamera(PointX, cameras[src_id], point, proj_depth);
					int src_r = int(point.y + 0.5f);
					int src_c = int(point.x + 0.5f);
					if (src_c >= 0 && src_c < src_cols && src_r >= 0 && src_r < src_rows) {
						if (masks[src_id].at<uchar>(src_r, src_c) == 1)
							continue;

						float src_depth = depths[src_id].at<float>(src_r, src_c);
						cv::Vec3f src_normal = normals[src_id].at<cv::Vec3f>(src_r, src_c);
						if (src_depth <= 0.0) {
							continue;
						}
						if (src_depth < cameras[i].depth_min || src_depth > cameras[i].depth_max) {
							continue;
						}

						float3 tmp_X = Get3DPointonWorld(src_c, src_r, src_depth, cameras[src_id]);
						float2 tmp_pt;
						ProjectonCamera(tmp_X, cameras[i], tmp_pt, proj_depth);
						float reproj_error = sqrt(pow(c - tmp_pt.x, 2) + pow(r - tmp_pt.y, 2));
						float relative_depth_diff = fabs(proj_depth - ref_depth) / ref_depth;
						float angle = GetAngle(ref_normal, src_normal);

						if (reproj_error < 2.0f && relative_depth_diff < 0.01f && angle < 0.174533f) {
							used_list[j].x = src_c;
							used_list[j].y = src_r;

							float tmp_index = reproj_error + 200 * relative_depth_diff + angle * 10;
							dynamic_consistency += exp(-tmp_index);
							num_consistent++;
						}
					}
				}

				if (num_consistent >= 1 && (dynamic_consistency > 0.3 * num_consistent)) {
					consistency[i](r, c) = dynamic_consistency;

					for (int j = 0; j < num_ngb; ++j) {
						if (used_list[j].x == -1)
							continue;
						masks[problems[i].src_image_ids[j]].at<uchar>(used_list[j].y, used_list[j].x) = 1;
						consistency[problems[i].src_image_ids[j]](used_list[j].y, used_list[j].x) = dynamic_consistency;
					}
				}
			}
		}
		std::stringstream result_path;
		result_path << dense_folder << "/HPM_MVS_plusplus" << "/2333_" << std::setw(8) << std::setfill('0') << problems[i].ref_image_id;
		std::string result_folder = result_path.str();
		std::string mask_path = result_folder + "/confidence.dmb";
		writeDepthDmb(mask_path, consistency[i]);
	}
	cameras.clear();
	depths.clear();
	normals.clear();
	masks.clear();
	consistency.clear();

	cameras.shrink_to_fit();
	depths.shrink_to_fit();
	normals.shrink_to_fit();
	masks.shrink_to_fit();
	consistency.shrink_to_fit();
	std::cout << "Hypotheses Confidence Evaluating Over..." << std::endl;
}

int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cout << "USAGE: HPM-MVS_plusplus dense_folder true/flase(mask defualt: flase)" << std::endl;
		return -1;
	}

	std::string dense_folder = argv[1];
	std::string mask = argv[2];
	bool mask_flag = false;
	if (mask == "true") {
		mask_flag = true;
	}

	std::vector<Problem> problems;
	GenerateSampleList(dense_folder, problems);

	std::string output_folder = dense_folder + std::string("/HPM_MVS_plusplus");
	mkdir(output_folder.c_str());

	size_t num_images = problems.size();
	std::cout << "There are " << num_images << " problems needed to be processed!" << std::endl;

	int max_num_downscale = ComputeMultiScaleSettings(dense_folder, problems);

	int flag = 0;
	int geom_iterations;
	bool geom_consistency = false;
	bool prior_consistency = false;
	bool hierarchy = false;
	bool multi_geometry = false;
	bool mand_consistency = false;
	int max_hpm_scale = max_num_downscale;
	while (max_num_downscale >= 0) {
		geom_iterations = 3;
		std::cout << "Scale: " << max_num_downscale << std::endl;

		for (size_t i = 0; i < num_images; ++i) {
			if (problems[i].num_downscale >= 0) {
				problems[i].cur_image_size = problems[i].max_image_size / pow(2, problems[i].num_downscale);
				problems[i].num_downscale--;
			}
		}

		if (flag == 0) {
			flag = 1;
			geom_consistency = false;
			prior_consistency = false;
			for (size_t i = 0; i < num_images; ++i) {
				ProcessProblem(dense_folder, problems, i, geom_consistency, prior_consistency, hierarchy, mand_consistency, max_num_downscale);
			}
			prior_consistency = true;
			geom_consistency = false;
			for (int hpm_scale = max_hpm_scale; hpm_scale >= max_num_downscale; hpm_scale--) {
				ConfidenceEvaluation(dense_folder, problems, geom_consistency);
				for (size_t i = 0; i < num_images; ++i) {
					std::cout << "HPM Scale: " << hpm_scale << std::endl;
					int hpm_scale_distance = hpm_scale - problems[i].num_downscale - 1;
					ProcessProblem(dense_folder, problems, i, geom_consistency, prior_consistency, hierarchy, mand_consistency, max_num_downscale, multi_geometry, hpm_scale_distance);
				}
			}

			geom_consistency = false;
			prior_consistency = false;
			for (int geom_iter = 0; geom_iter < geom_iterations; ++geom_iter) {
				if (geom_iter == 0) {
					multi_geometry = false;
				}
				else {
					multi_geometry = true;
				}

				if (geom_iter > 0) {
					mand_consistency = true;
					ConfidenceEvaluation(dense_folder, problems, geom_consistency);
					geom_consistency = true;
				}
				else {
					geom_consistency = true;
					mand_consistency = false;
				}

				for (size_t i = 0; i < num_images; ++i) {
					ProcessProblem(dense_folder, problems, i, geom_consistency, prior_consistency, hierarchy, mand_consistency, max_num_downscale, multi_geometry);
				}
			}
		}
		else {
			for (size_t i = 0; i < num_images; ++i) {
				JointBilateralUpsampling(dense_folder, problems[i], problems[i].cur_image_size);
			}

			hierarchy = true;
			mand_consistency = false;
			geom_consistency = false;
			prior_consistency = false;
			for (size_t i = 0; i < num_images; ++i) {
				ProcessProblem(dense_folder, problems, i, geom_consistency, prior_consistency, hierarchy, mand_consistency, max_num_downscale);
			}
			hierarchy = false;
			prior_consistency = true;
			geom_consistency = false;
			multi_geometry = false;
			for (int hpm_scale = max_hpm_scale; hpm_scale >= max_num_downscale; hpm_scale--) {
				ConfidenceEvaluation(dense_folder, problems, geom_consistency);
				for (size_t i = 0; i < num_images; ++i) {
					std::cout << "HPM Scale: " << hpm_scale << std::endl;
					int hpm_scale_distance = hpm_scale - problems[i].num_downscale - 1;
					ProcessProblem(dense_folder, problems, i, geom_consistency, prior_consistency, hierarchy, mand_consistency, max_num_downscale, multi_geometry, hpm_scale_distance);
				}
			}

			geom_consistency = false;
			prior_consistency = false;
			for (int geom_iter = 0; geom_iter < geom_iterations; ++geom_iter) {
				if (geom_iter == 0) {
					multi_geometry = false;
				}
				else {
					multi_geometry = true;
				}

				if (geom_iter > 0) {
					mand_consistency = true;
					ConfidenceEvaluation(dense_folder, problems, geom_consistency);
					geom_consistency = true;
				}
				else {
					geom_consistency = true;
					mand_consistency = false;
				}

				for (size_t i = 0; i < num_images; ++i) {
					ProcessProblem(dense_folder, problems, i, geom_consistency, prior_consistency, hierarchy, mand_consistency, max_num_downscale, multi_geometry);
				}
			}
		}

		max_num_downscale--;
	}
	geom_consistency = true;
	if (mask_flag) {
		RunFusion_Sky_Strict(dense_folder, problems, geom_consistency);
	}
	else {
		RunFusion(dense_folder, problems, geom_consistency);
	}
	return 0;
}
