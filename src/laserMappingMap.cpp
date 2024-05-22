#include "laserMapping.h"

void LaserMappingNode::map_incremental() {
    // PointVector PointToAdd;                 // 这部分点最终去了哪里?
    PointVector PointNoNeedDownsample;
    // PointToAdd.reserve(feats_down_size);    // 设置大小
    PointNoNeedDownsample.reserve(feats_down_size);

    for (int i = 0; i < feats_down_size; i++) {
        /* No points found within the given threshold of nearest search*/
        if (Nearest_Points[i].empty()) {
            // 当前点无最近邻, 加入"不需要降采样"
            PointNoNeedDownsample.emplace_back(feats_down_world->points[i]);
            continue;
        }
        /* decide if need add to map */
        // 有最近邻
        if (!Nearest_Points[i].empty()) {
            // 获取最近邻引用
            const PointVector& points_near = Nearest_Points[i];
            // bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point;
            // 计算当前点所在下采样网格的中心点
            mid_point.x = floor(feats_down_world->points[i].x / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            /* If the nearest points is definitely outside the downsample box */
            // 如果最近点在盒子之外，加入"不需要降采样" , 即当前格子无最近邻
            if (fabs(points_near[0].x - mid_point.x) > 1.732 * filter_size_map_min || fabs(points_near[0].y - mid_point.y) > 1.732 * filter_size_map_min ||
                fabs(points_near[0].z - mid_point.z) > 1.732 * filter_size_map_min) {
                PointNoNeedDownsample.emplace_back(feats_down_world->points[i]);
                continue;
            }
            // /* Check if there is a point already in the downsample box and closer to the center point */
            // // 计算当前点到中心点 mid_point 的距离 dist
            // float dist  = calc_dist<float>(feats_down_world->points[i],mid_point);
            // // 遍历5次(最近邻最多5个)
            // for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i ++)
            // {
            //     // 如果最近点小于5个, 跳出循环
            //     if (points_near.size() < NUM_MATCH_POINTS) break;
            //     /* Those points which are outside the downsample box should not be considered. */
            //     // 某最近点位于盒子外，判断下一个
            //     if (fabs(points_near[readd_i].x - mid_point.x) > 0.5 * filter_size_map_min || fabs(points_near[readd_i].y - mid_point.y) > 0.5 *
            //     filter_size_map_min || fabs(points_near[readd_i].z - mid_point.z) > 0.5 * filter_size_map_min) {
            //         continue;
            //     }
            //     // 如果找到一个距离中心点更近的点，则该点不需要添加
            //     if (calc_dist<float>(points_near[readd_i], mid_point) < dist)
            //     {
            //         need_add = false;
            //         break;
            //     }
            // }
            // if (need_add) PointToAdd.emplace_back(feats_down_world->points[i]);
        }
        // // 无最近邻,则
        // else
        // {
        //     PointToAdd.emplace_back(feats_down_world->points[i]);
        // }
    }
    // 此处把不需要降采样的点先添加
    ikdtree.Add_Points(PointNoNeedDownsample, false);
}

// 处理局部地图，并根据当前的激光雷达位置，动态调整局部地图的边界
void LaserMappingNode::lasermap_fov_segment() {
    const float MOV_THRESHOLD = 1.5f;
    static bool Localmap_Initialized = false;

    // 对cub_needrm向量进行收缩以适应实际大小
    cub_needrm.shrink_to_fit();
    // 确定当前激光雷达位置pos_LiD
    V3D pos_LiD = kf_output.x_.pos + kf_output.x_.rot * Lidar_T_wrt_IMU;

    // 根据当前激光雷达位置设置局部地图边界
    if (!Localmap_Initialized) {
        for (int i = 0; i < 3; i++) {
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    // 计算激光雷达位置与局部地图边界的距离dist_to_map_edge
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++) {
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
    }
    // 如果激光雷达与任何边界的距离小于阈值,则需要移动局部地图
    if (!need_move) return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD - 1)));
    // 根据激光雷达与边界的距离调整局部地图的边界，并将需要移除的立方体子区域存储在cub_needrm向量中。
    for (int i = 0; i < 3; i++) {
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE) {
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.emplace_back(tmp_boxpoints);
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) {
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.emplace_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    // points_cache_collect(); // debug
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
    // int points_cache_size = points_history.size();
}

void LaserMappingNode::publish_init_kdtree(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFullRes) {
    int size_init_ikdtree = ikdtree.size();
    CloudType::Ptr laserCloudInit(new CloudType(size_init_ikdtree, 1));

    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    PointVector().swap(ikdtree.PCL_Storage);
    ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);

    laserCloudInit->points = ikdtree.PCL_Storage;
    pcl::toROSMsg(*laserCloudInit, laserCloudmsg);

    laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
    laserCloudmsg.header.frame_id = "camera_init";
    pubLaserCloudFullRes->publish(laserCloudmsg);
}
