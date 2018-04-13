#include <queue>
#include "em_exploration/SLAM2D.h"
#include "em_exploration/Utils.h"

using namespace gtsam;

namespace em_exploration {
SLAM2D::SLAM2D(const Map::Parameter &parameter, RNG::SeedType seed)
    : map_(parameter), step_(0), marginals_update_(false), optimized_(false), rng_(seed) {
  ISAM2Params params;
//    params.setFactorization("QR");
  isam_ = std::make_shared<ISAM2>(params);
}

void SLAM2D::printParameters() const {
  std::cout << "----------------------------" << std::endl;
  std::cout << "SLAM2D:" << std::endl;
  map_.getParameter().print();
  std::cout << "----------------------------" << std::endl;
}

void SLAM2D::fromISAM2(std::shared_ptr<gtsam::ISAM2> isam, const Map &map, const gtsam::Values values) {
  isam_ = isam;
  result_ = values;
  optimized_ = true;
  marginals_update_ = false;

  map_ = map;
}

void SLAM2D::addPrior(unsigned int key, const LandmarkBeliefState &landmark_state) {
  optimized_ = false;

  Symbol l0 = getLandmarkSymbol(key);

  noiseModel::Gaussian::shared_ptr noise_model =
      noiseModel::Gaussian::Information(landmark_state.information);
  graph_.add(PriorFactor<Point2>(l0, landmark_state.point, noise_model));

  initial_estimate_.insert(l0, landmark_state.point);
}

void SLAM2D::addPrior(const VehicleBeliefState &vehicle_state) {
  assert(step_ == 0);

  optimized_ = false;

  Symbol x0 = getVehicleSymbol(step_++);

  noiseModel::Gaussian::shared_ptr noise_model =
      noiseModel::Gaussian::Information(vehicle_state.information);
  graph_.add(PriorFactor<Pose2>(x0, vehicle_state.pose, noise_model));

  initial_estimate_.insert(x0, vehicle_state.pose);
}

SLAM2D::OdometryFactor2DPtr SLAM2D::buildOdometryFactor(unsigned int x1, unsigned int x2,
                                                        const SimpleControlModel::ControlState &control_state) {
  Symbol sx1 = getVehicleSymbol(x1);
  Symbol sx2 = getVehicleSymbol(x2);

  noiseModel::Diagonal::shared_ptr noise_model =
      noiseModel::Diagonal::Sigmas(control_state.getSigmas());
  return boost::make_shared<SLAM2D::OdometryFactor2D>(sx1, sx2,
                                                      control_state.getOdom(), noise_model);
}

void SLAM2D::addOdometry(const SimpleControlModel::ControlState &control_state) {
  optimized_ = false;

  OdometryFactor2DPtr factor = SLAM2D::buildOdometryFactor(step_ - 1, step_, control_state);
  graph_.add(*factor);

  Symbol x1 = getVehicleSymbol(step_ - 1);
  Symbol x2 = getVehicleSymbol(step_++);

  Pose2 p1;
  if (result_.exists(x1)) {
    p1 = result_.at<Pose2>(x1);
  } else {
    p1 = initial_estimate_.at<Pose2>(x1);
  }

  Pose2 p2 = p1 * control_state.getOdom();
  initial_estimate_.insert(x2, p2);
}

SLAM2D::MeasurementFactor2DPtr SLAM2D::buildMeasurementFactor(unsigned int x, unsigned int l,
                                                              const BearingRangeSensorModel::Measurement &measurement) {
  Symbol sx = getVehicleSymbol(x);
  Symbol sl = getLandmarkSymbol(l);

  noiseModel::Diagonal::shared_ptr noise_model =
      noiseModel::Diagonal::Sigmas(measurement.getSigmas());
  return boost::make_shared<MeasurementFactor2D>(sx, sl,
                                                 measurement.getBearing(),
                                                 measurement.getRange(), noise_model);
}

void SLAM2D::addMeasurement(unsigned int key, const BearingRangeSensorModel::Measurement &measurement) {
  optimized_ = false;

  SLAM2D::MeasurementFactor2DPtr factor = SLAM2D::buildMeasurementFactor(step_ - 1, key, measurement);
  graph_.add(*factor);

  Symbol x = getVehicleSymbol(step_ - 1);
  Symbol l = getLandmarkSymbol(key);

  if (!initial_estimate_.exists(l) && !result_.exists(l)) {
    Pose2 origin;
    if (result_.exists(x))
      origin = result_.at<Pose2>(x);
    else
      origin = initial_estimate_.at<Pose2>(x);

    Point2 global = measurement.transformFrom(origin);
    initial_estimate_.insert(l, global);

    map_.addLandmark(key, global);
  }
}

void SLAM2D::saveGraph(const std::string &name) const {
  std::ofstream os(name);
  if (!os.is_open()) return;

  isam_->getFactorsUnsafe().saveGraph(os, result_);
  std::cout << "Visualize the graph with `dot -Tpdf ./graph.dot -O`" << std::endl;
  os.close();
}

void SLAM2D::printGraph() const {
  graph_.print("Graph:\n");
  initial_estimate_.print("Initial Estimate:\n");
}

Eigen::MatrixXd SLAM2D::jointMarginalCovarianceLocal(const std::vector<unsigned int> &poses,
                                                     const std::vector<unsigned int> &landmarks) const {
  if (!marginals_update_) {
#ifdef USE_FAST_MARGINAL
    marginals_ = std::make_shared<FastMarginals>(isam_);
#else
    marginals_ = std::make_shared<Marginals>(isam_->getFactorsUnsafe(), result_);
#endif
    marginals_update_ = true;
  }

  std::vector<Key> keys;
  for (unsigned int x : poses)
    keys.emplace_back(getVehicleSymbol(x));
  for (unsigned int l : landmarks)
    keys.emplace_back(getLandmarkSymbol(l));

#ifdef USE_FAST_MARGINAL
  return marginals_->jointMarginalCovariance(keys);
#else
  JointMarginal cov = marginals_->jointMarginalCovariance(keys);

  size_t n = poses.size() * 3 + landmarks.size() * 2;
  Matrix S(n, n);
  int p = 0, q = 0;
  for (int i = 0; i < keys.size(); ++i) {
    for (int j = i; j < keys.size(); ++j) {
      Matrix Sij = cov.at(keys[i], keys[j]);
      S.block(p, q, Sij.rows(), Sij.cols()) = Sij;
      q += Sij.cols();
      if (j == keys.size() - 1) {
        p += Sij.rows();
        q = p;
      }
    }
  }
  S.triangularView<Eigen::Lower>() = S.transpose();
  return S;
#endif
}

Eigen::MatrixXd SLAM2D::jointMarginalCovariance(const std::vector<unsigned int> &poses,
                                                const std::vector<unsigned int> &landmarks) const {
  if (!marginals_update_) {
#ifdef USE_FAST_MARGINAL
    marginals_ = std::make_shared<FastMarginals>(isam_);
#else
    marginals_ = std::make_shared<Marginals>(isam_->getFactorsUnsafe(), result_);
#endif
    marginals_update_ = true;
  }

  std::vector<Key> keys;
  for (unsigned int x : poses)
    keys.emplace_back(getVehicleSymbol(x));
  for (unsigned int l : landmarks)
    keys.emplace_back(getLandmarkSymbol(l));

#ifdef USE_FAST_MARGINAL
  /// TODO
  assert(false);
#else
  JointMarginal cov = marginals_->jointMarginalCovariance(keys);

  size_t n = poses.size() * 3 + landmarks.size() * 2;
  Matrix S(n, n);
  int p = 0, q = 0;
  for (int i = 0; i < keys.size(); ++i) {
    for (int j = i; j < keys.size(); ++j) {
      Matrix Sij = cov.at(keys[i], keys[j]);

      if (i < poses.size()) {
        Pose2 R(result_.at<Pose2>(keys[i]).r(), Point2());
        Sij = R.matrix() * Sij;
      }

      if (j < poses.size()) {
        Pose2 R(result_.at<Pose2>(keys[j]).r(), Point2());
        Sij = Sij * R.matrix().transpose();
      }

      S.block(p, q, Sij.rows(), Sij.cols()) = Sij;
      q += Sij.cols();
      if (j == keys.size() - 1) {
        p += Sij.rows();
        q = p;
      }
    }
  }
  S.triangularView<Eigen::Lower>() = S.transpose();
  return S;
#endif
}

std::shared_ptr<const ISAM2> SLAM2D::getISAM2() const {
  return std::const_pointer_cast<const ISAM2>(isam_);
}

void SLAM2D::optimize(bool update_covariance) {
  if (optimized_)
    return;

  if (graph_.size() == 0)
    return;

  isam_->update(graph_, initial_estimate_);
  result_ = isam_->calculateEstimate();

#ifdef USE_FAST_MARGINAL
  marginals_ = std::make_shared<FastMarginals>(isam_);
#endif

  for (int i = 0; i < step_; ++i) {
    Symbol x = getVehicleSymbol(i);
    Pose2 pose = result_.at<Pose2>(x);
    if (update_covariance) {
#ifdef USE_FAST_MARGINAL
      Eigen::Matrix3d covariance = marginals_->marginalCovariance(x);
#else
      Eigen::Matrix3d covariance = isam_->marginalCovariance(x);
#endif
      Pose2 R(pose.r(), Point2());
      covariance = R.matrix() * covariance * R.matrix().transpose();

      VehicleBeliefState state(pose, inverse(covariance));
      if (i < map_.getTrajectorySize()) {
        state.core_vehicle = map_.getVehicle(i).core_vehicle;
        map_.updateVehicle(i, state);
      } else {
        state.core_vehicle = (i == step_ - 1);
        map_.addVehicle(state);
      }
    } else {
      if (i < map_.getTrajectorySize())
        map_.updateVehicle(i, pose);
      else
        map_.addVehicle(pose);
    }
  }
  for (auto it = map_.beginLandmark(); it != map_.endLandmark(); ++it) {
    Symbol l = getLandmarkSymbol(it->first);
    it->second.point = result_.at<Point2>(l);
    if (update_covariance)
#ifdef USE_FAST_MARGINAL
      it->second.information = marginals_->marginalCovariance(l).inverse();
#else
      it->second.information = isam_->marginalCovariance(l).inverse();
#endif
  }

  graph_.resize(0);
  initial_estimate_.clear();
  optimized_ = true;
  marginals_update_ = false;
}

std::pair<double, std::shared_ptr<Map>> SLAM2D::sample() const {
  FastVector<ISAM2Clique::shared_ptr> roots = isam_->roots();

  double lp = 0.0;
  VectorValues result(isam_->getDelta());
  for (const ISAM2Clique::shared_ptr &root : roots) {
    lp += optimizeInPlacePerturbation(root, result);
  }
  Values values = isam_->getLinearizationPoint().retract(result);

  std::shared_ptr<Map> sampled_map(new Map(map_.getParameter()));
  for (int i = 0; i < step_; ++i)
    sampled_map->addVehicle(values.at<Pose2>(getVehicleSymbol(i)));

  for (auto it = map_.cbeginLandmark(); it != map_.cendLandmark(); ++it)
    sampled_map->addLandmark(it->first, values.at<Point2>(getLandmarkSymbol(it->first)));

  return std::make_pair(lp, sampled_map);
}

double SLAM2D::optimizeInPlacePerturbation(const ISAM2Clique::shared_ptr &clique, VectorValues &result) const {
  GaussianConditional::shared_ptr conditional = clique->conditional();
  const Vector xS = result.vector(FastVector<Key>(conditional->beginParents(), conditional->endParents()));

  Vector rhs = conditional->get_d() - conditional->get_S() * xS;

  double lp = 0.0;
  for (int i = 0; i < rhs.rows(); ++i) {
    double n = rng_.normal01();
    rhs(i) += n;
    lp += -0.5 * n * n;
  }

  const Vector solution = conditional->get_R().triangularView<Eigen::Upper>().solve(rhs);

  if (solution.hasNaN()) {
    throw IndeterminantLinearSystemException(conditional->keys().front());
  }

  DenseIndex vectorPosition = 0;
  for (GaussianConditional::const_iterator frontal = conditional->beginFrontals();
       frontal != conditional->endFrontals(); ++frontal) {
    result.at(*frontal) = solution.segment(vectorPosition, conditional->getDim(frontal));
    vectorPosition += conditional->getDim(frontal);
  }

  for(const ISAM2Clique::shared_ptr &child: clique->children)
    lp += optimizeInPlacePerturbation(child, result);

  return lp;
}

}
