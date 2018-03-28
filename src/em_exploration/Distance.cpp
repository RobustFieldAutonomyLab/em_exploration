#include "em_exploration/Distance.h"

namespace em_exploration {

double sqDistanceBetweenPoses(const gtsam::Pose2 &pose1, const gtsam::Pose2 &pose2, double angle_weight) {
  double range = pose1.range(pose2);
  double angle = pose1.bearing(pose2).theta();
  return pow(range, 2) + pow(angle * angle_weight, 2);
}

int nearestNeighbor(const std::vector<gtsam::Pose2> &poses, const gtsam::Pose2 &pose, double angle_weight) {
  int n = -1;
  double d = std::numeric_limits<double>::max();
  for (int i = 0; i < poses.size(); ++i) {
    double di = sqDistanceBetweenPoses(poses[i], pose, angle_weight);
    if (di < d) {
      d = di;
      n = i;
    }
  }
  return n;
}

std::vector<int> radiusNeighbors(const std::vector<gtsam::Pose2> &poses,
                                 const gtsam::Pose2 &pose,
                                 double radius,
                                 double angle_weight) {
  std::vector<int> n;
  if (radius < 0)
    return n;

  radius *= radius;
  for (int i = 0; i < poses.size(); ++i) {
    double d = sqDistanceBetweenPoses(poses[i], pose, angle_weight);
    if (d < radius)
      n.push_back(i);
  }
  return n;
}

std::shared_ptr<KDTreeR2> KDTreeR2::clone() const {
  if (tree_ == nullptr)
    return nullptr;

  std::shared_ptr<KDTreeR2Data> data(new KDTreeR2Data(*data_));

  std::shared_ptr<KDTreeR2> kdtree;
  kdtree->data_ = data;

  flann::Matrix<double> dataset(kdtree->data_->data(), kdtree->data_->rows(), 2);
  kdtree->tree_ = std::make_shared<KDTreeR2Index>(dataset, flann::KDTreeSingleIndexParams());
  kdtree->tree_->buildIndex();

  return kdtree;
}

void KDTreeR2::build(const std::vector<gtsam::Point2> &points) {
  data_ = std::make_shared<KDTreeR2Data>(points.size(), 2);
  flann::Matrix<double> dataset(data_->data(), points.size(), 2);
  for (int i = 0; i < points.size(); ++i) {
    *(dataset[i] + 0) = points[i].x();
    *(dataset[i] + 1) = points[i].y();
  }

  tree_ = std::make_shared<KDTreeR2Index>(dataset, flann::KDTreeSingleIndexParams());
  tree_->buildIndex();
}

void KDTreeR2::addPoints(const std::vector<gtsam::Point2> &points) {
  if (!tree_) {
    build(points);
  }

  data_->conservativeResize(points.size() + data_->rows(), Eigen::NoChange);
  flann::Matrix<double> dataset(data_->data() + tree_->size() * 2, points.size(), 2);
  for (int i = 0; i < points.size(); ++i) {
    *(dataset[i] + 0) = points[i].x();
    *(dataset[i] + 1) = points[i].y();
  }

  tree_->addPoints(dataset, REBUILD_THRESHOLD);
}

int KDTreeR2::queryNearestNeighbor(const gtsam::Point2 &point) const {
  assert(tree_ != nullptr);

  flann::Matrix<double> query(new double[2], 1, 2);
  *(query[0] + 0) = point.x();
  *(query[0] + 1) = point.y();

  flann::Matrix<int> indices(new int[1], 1, 1);
  flann::Matrix<double> dists(new double[1], 1, 1);

  flann::SearchParams params(flann::FLANN_CHECKS_UNLIMITED, 1.0, false);
  tree_->knnSearch(query, indices, dists, 1, params);
  int n = *indices[0];

  delete[] query.ptr();
  delete[] indices.ptr();
  delete[] dists.ptr();

  return n;
}

std::vector<int> KDTreeR2::queryRadiusNeighbors(const gtsam::Point2 &point,
                                                double radius,
                                                int max_neighbors) const {
  assert(tree_ != nullptr);

  if (radius < 0)
    return std::vector<int>();

  flann::Matrix<double> query(new double[2], 1, 2);
  *(query[0] + 0) = point.x();
  *(query[0] + 1) = point.y();

  std::vector<std::vector<int>> indices(1);
  std::vector<std::vector<double>> dists(1);

  flann::SearchParams params(flann::FLANN_CHECKS_UNLIMITED, 0, false);
  params.max_neighbors = max_neighbors;
  tree_->radiusSearch(query, indices, dists, (float) (radius * radius), params);

  delete[] query.ptr();

  return indices[0];
}

std::shared_ptr<KDTreeSE2> KDTreeSE2::clone() const {
  if (tree_ == nullptr)
    return nullptr;

  std::shared_ptr<KDTreeSE2Data> data(new KDTreeSE2Data(*data_));

  std::shared_ptr<KDTreeSE2> kdtree;
  kdtree->data_ = data;

  flann::Matrix<double> dataset(kdtree->data_->data(), kdtree->data_->rows(), 4);
  kdtree->tree_ = std::make_shared<KDTreeSE2Index>(dataset, flann::KDTreeSingleIndexParams(),
                                                   L2_SE2<double>(angle_weight_));
  kdtree->tree_->buildIndex();

  return kdtree;
}

void KDTreeSE2::build(const std::vector<gtsam::Pose2> &poses, double angle_weight) {
  angle_weight_ = angle_weight;
  data_ = std::make_shared<KDTreeSE2Data>(poses.size(), 4);
  flann::Matrix<double> dataset(data_->data(), poses.size(), 4);
  for (int i = 0; i < poses.size(); ++i) {
    *(dataset[i] + 0) = poses[i].x();
    *(dataset[i] + 1) = poses[i].y();
    *(dataset[i] + 2) = poses[i].rotation().c();
    *(dataset[i] + 3) = poses[i].rotation().s();
  }

  tree_ = std::make_shared<KDTreeSE2Index>(dataset, flann::KDTreeSingleIndexParams(), L2_SE2<double>(angle_weight));
  tree_->buildIndex();
}

void KDTreeSE2::addPoints(const std::vector<gtsam::Pose2> &poses) {
  assert(tree_);
//    if (!tree_) {
//      build(poses, angle_weight);
//    }

  data_->conservativeResize(poses.size() + data_->rows(), Eigen::NoChange);
  flann::Matrix<double> dataset(data_->data() + tree_->size() * 4, poses.size(), 4);
  for (int i = 0; i < poses.size(); ++i) {
    *(dataset[i] + 0) = poses[i].x();
    *(dataset[i] + 1) = poses[i].y();
    *(dataset[i] + 2) = poses[i].rotation().c();
    *(dataset[i] + 3) = poses[i].rotation().s();
  }
  tree_->addPoints(dataset, REBUILD_THRESHOLD);
}

int KDTreeSE2::queryNearestNeighbor(const gtsam::Pose2 &pose) const {
  assert(tree_ != nullptr);

  flann::Matrix<double> query(new double[4], 1, 4);
  *(query[0] + 0) = pose.x();
  *(query[0] + 1) = pose.y();
  *(query[0] + 2) = pose.rotation().c();
  *(query[0] + 3) = pose.rotation().s();

  flann::Matrix<int> indices(new int[1], 1, 1);
  flann::Matrix<double> dists(new double[1], 1, 1);

  flann::SearchParams params(flann::FLANN_CHECKS_UNLIMITED, 0, false);
  tree_->knnSearch(query, indices, dists, 0, params);
  int n = *indices[0];

  delete[] query.ptr();
  delete[] indices.ptr();
  delete[] dists.ptr();

  return n;
}

std::vector<int> KDTreeSE2::queryRadiusNeighbors(const gtsam::Pose2 &pose, double radius, int max_neighbors) const {
  assert(tree_ != nullptr);

  if (radius < 0)
    return std::vector<int>();

  flann::Matrix<double> query(new double[4], 1, 4);
  *(query[0] + 0) = pose.x();
  *(query[0] + 1) = pose.y();
  *(query[0] + 2) = pose.rotation().c();
  *(query[0] + 3) = pose.rotation().s();

  std::vector<std::vector<int>> indices;
  std::vector<std::vector<double>> dists;

  flann::SearchParams params(flann::FLANN_CHECKS_UNLIMITED, 0, false);
  params.max_neighbors = max_neighbors;
  tree_->radiusSearch(query, indices, dists, (float) (radius * radius), params);

  delete[] query.ptr();

  return indices[0];
}

}

