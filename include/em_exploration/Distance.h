#ifndef EM_EXPLORATION_DISTANCE_H
#define EM_EXPLORATION_DISTANCE_H

#include <vector>

#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Pose2.h>
#include <flann/flann.hpp>

namespace em_exploration {

const float REBUILD_THRESHOLD = 2.0;

/**
 * Distance function for KDTrees in SE2.
 * @tparam T
 */
template<class T>
struct L2_SE2 {
  typedef bool is_kdtree_distance;

  typedef T ElementType;
  typedef typename flann::Accumulator<T>::Type ResultType;

  T angle_weight;

  L2_SE2(T angle_weight) : angle_weight(angle_weight) {}

  template<typename Iterator1, typename Iterator2>
  ResultType operator()(Iterator1 a, Iterator2 b, size_t size, ResultType worst_dist = -1) const {
    ResultType result = ResultType();
    ResultType diff;
    diff = *a++ - *b++;
    result += diff * diff;
    diff = *a++ - *b++;
    result += diff * diff;

    // Angle difference (assuming two angles are bounded in [-pi, pi))
//    diff = (T)fabs(*a++ - *b++);
//    diff = std::min(diff, (T)M_PI * 2 - diff);
//    result += diff * diff * angle_weight * angle_weight;
    diff = *a++ - *b++;
    result += angle_weight * angle_weight * diff * diff;
    diff = *a++ - *b++;
    result += angle_weight * angle_weight * diff * diff;

    return result;
  };

  template<typename U, typename V>
  ResultType accum_dist(const U &a, const V &b, int i) const {
    if (i < 2) {
      return (a - b) * (a - b);
    } else {
      ResultType diff = (T) fabs(a - b);
//      diff = std::min(diff, (T)M_PI * 2 - diff);
      return diff * diff * angle_weight * angle_weight;
    }
  };

};

/**
 * KDTrees in R2 using gtsam::Point2 type.
 */
class KDTreeR2 {
 public:
  typedef flann::Index<flann::L2<double>> KDTreeR2Index;
  typedef std::shared_ptr<KDTreeR2Index> KDTreeR2IndexPtr;
  typedef Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor> KDTreeR2Data;

  KDTreeR2()
      : tree_(nullptr), data_(nullptr) {}

  KDTreeR2(const KDTreeR2 &other)
      : tree_(other.tree_), data_(other.data_) {}

  KDTreeR2 &operator=(const KDTreeR2 &other) {
    tree_ = other.tree_;
    data_ = other.data_;
    return *this;
  }

  ~KDTreeR2() {}

  std::shared_ptr<KDTreeR2> clone() const;

  /// Build the kdtree from points.
  void build(const std::vector<gtsam::Point2> &points);

  /// Add points to the kdtree.
  void addPoints(const std::vector<gtsam::Point2> &points);

  /// Query nearest neighbor and return the index of it.
  int queryNearestNeighbor(const gtsam::Point2 &point) const;

  /// Query neighbors within radius and return all neighbors if max_neighbors is -1.
  std::vector<int> queryRadiusNeighbors(const gtsam::Point2 &point, double radius, int max_neighbors = -1) const;

 private:
  KDTreeR2IndexPtr tree_;
  std::shared_ptr<KDTreeR2Data> data_;
};

/**
 * KDTrees in SE2 with weight on angles using gtsam::Pose2 type.
 */
class KDTreeSE2 {
 public:
  typedef flann::Index<L2_SE2<double>> KDTreeSE2Index;
  typedef std::shared_ptr<KDTreeSE2Index> KDTreeSE2IndexPtr;
  typedef Eigen::Matrix<double, Eigen::Dynamic, 4, Eigen::RowMajor> KDTreeSE2Data;

  KDTreeSE2()
      : tree_(nullptr) {}

  ~KDTreeSE2() {}

  KDTreeSE2(const KDTreeSE2 &other)
      : angle_weight_(other.angle_weight_), tree_(other.tree_), data_(other.data_) {}

  KDTreeSE2 &operator=(const KDTreeSE2 &other) {
    angle_weight_ = other.angle_weight_;
    tree_ = other.tree_;
    data_ = other.data_;
    return *this;
  }

  std::shared_ptr<KDTreeSE2> clone() const;

  void build(const std::vector<gtsam::Pose2> &poses, double angle_weight);

  void addPoints(const std::vector<gtsam::Pose2> &poses);

  int queryNearestNeighbor(const gtsam::Pose2 &pose) const;

  /// Query neighbors within radius and return all neighbors if max_neighbors is -1.
  std::vector<int> queryRadiusNeighbors(const gtsam::Pose2 &pose, double radius, int max_neighbors = -1) const;

 private:
  double angle_weight_;
  KDTreeSE2IndexPtr tree_;
  std::shared_ptr<KDTreeSE2Data> data_;
};

/// Return the squared distance between two poses.
double sqDistanceBetweenPoses(const gtsam::Pose2 &pose1, const gtsam::Pose2 &pose2, double angle_weight = 0.0);

/// Return the nearest neighbor without using KDTrees.
int nearestNeighbor(const std::vector<gtsam::Pose2> &poses, const gtsam::Pose2 &pose, double angle_weight);

/// Return neighbors within radius without using KDTrees.
std::vector<int> radiusNeighbors(const std::vector<gtsam::Pose2> &poses,
                                 const gtsam::Pose2 &pose,
                                 double radius,
                                 double angle_weight);
}

#endif //EM_EXPLORATION_DISTANCE_H
