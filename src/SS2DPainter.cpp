#include "em_exploration/SLAM2D.h"
#include "em_exploration/VirtualMap.h"

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/eigen.h"

using namespace em_exploration;
namespace py = pybind11;

typedef Eigen::MatrixXd Matrix;

void buildCanvas(Matrix &canvas, const Map::Parameter &map_parameter, double res) {
  int cols = static_cast<int>(ceil((map_parameter.getMaxX() - map_parameter.getMinX()) / res));
  int rows = static_cast<int>(ceil((map_parameter.getMaxY() - map_parameter.getMinY()) / res));
  canvas = Matrix::Zero(rows, cols);
}

Matrix paintMap(const Map &map, double res, double chi_value) {
  Matrix canvas;
  const Map::Parameter &map_parameter = map.getParameter();
  buildCanvas(canvas, map_parameter, res);

  int rows = static_cast<int>(canvas.rows());
  int cols = static_cast<int>(canvas.cols());

  for (auto it = map.cbeginLandmark(); it != map.cendLandmark(); ++it) {
    int origin_row = static_cast<int>(floor((it->second.point.y() - map_parameter.getMinY()) / res));
    int origin_col = static_cast<int>(floor((it->second.point.x() - map_parameter.getMinX()) / res));

    double max_eig = 1.0 / minEigenvalue(it->second.information);
    int max_offset = static_cast<int>(ceil(sqrt(max_eig * chi_value) / res));

    int min_row = std::min(std::max(0, origin_row - max_offset), rows - 1);
    int max_row = std::min(std::max(0, origin_row + max_offset), rows - 1);
    int min_col = std::min(std::max(0, origin_col - max_offset), cols - 1);
    int max_col = std::min(std::max(0, origin_col + max_offset), cols - 1);

    const Eigen::Vector2d &u = it->second.point.vector();
    const Eigen::Matrix2d &I = it->second.information;
//    double c = sqrt(I.determinant()) / (2 * M_PI);
    double c = 1.0;

    for (int row = min_row; row <= max_row; ++row) {
      for (int col = min_col; col <= max_col; ++col) {
        Eigen::Vector2d x;
        x << map_parameter.getMinX() + res * col,
            map_parameter.getMinY() + res * row;

        double cv = ((x - u).transpose() * I * (x - u))(0, 0);
        if (cv < chi_value) {
          double p = c * exp(-0.5 * cv);
          if (p > canvas(row, col))
            canvas(row, col) = p;
        }
      }
    }
  }

  return canvas;
}

Matrix paintTrajectory(const Map &map, double res, double chi_value) {
  Matrix canvas;
  const Map::Parameter &map_parameter = map.getParameter();
  buildCanvas(canvas, map_parameter, res);

  int rows = static_cast<int>(canvas.rows());
  int cols = static_cast<int>(canvas.cols());

  for (auto it = map.cbeginTrajectory(); it != map.cendTrajectory(); ++it) {
    int origin_row = static_cast<int>(floor((it->pose.y() - map_parameter.getMinY()) / res));
    int origin_col = static_cast<int>(floor((it->pose.x() - map_parameter.getMinX()) / res));

    const Eigen::Matrix2d &I = it->information.block(0, 0, 2, 2);
    double max_eig = 1.0 / minEigenvalue(I);
    int max_offset = static_cast<int>(ceil(sqrt(max_eig * chi_value) / res));

    int min_row = std::min(std::max(0, origin_row - max_offset), rows - 1);
    int max_row = std::min(std::max(0, origin_row + max_offset), rows - 1);
    int min_col = std::min(std::max(0, origin_col - max_offset), cols - 1);
    int max_col = std::min(std::max(0, origin_col + max_offset), cols - 1);

    const Eigen::Vector2d &u = it->pose.translation().vector();
//    double c = sqrt(I.determinant()) / (2 * M_PI);
    double c = 1.0;

    for (int row = min_row; row <= max_row; ++row) {
      for (int col = min_col; col <= max_col; ++col) {
        Eigen::Vector2d x;
        x << map_parameter.getMinX() + res * col,
            map_parameter.getMinY() + res * row;

        double cv = ((x - u).transpose() * I * (x - u))(0, 0);
        if (cv < chi_value) {
          double p = c * exp(-0.5 * cv);
          if (p > canvas(row, col))
            canvas(row, col) = p;
        }
      }
    }
  }

  return canvas;
}

Matrix paintVehicle(const Map &map, double res, const BearingRangeSensorModel &sensor_model) {
  Matrix canvas;
  const Map::Parameter &map_parameter = map.getParameter();
  buildCanvas(canvas, map_parameter, res);

  int rows = static_cast<int>(canvas.rows());
  int cols = static_cast<int>(canvas.cols());

  const Pose2 &pose = map.getCurrentVehicle().pose;
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      Point2 point(map_parameter.getMinX() + res * col, map_parameter.getMinY() + res * row);
      BearingRangeSensorModel::Measurement m = sensor_model.measure(pose, point, false, false);
      if (sensor_model.check(m))
        canvas(row, col) = 1;
    }
  }
  return canvas;
}

Matrix paintVirtualMap(const VirtualMap &map, double res, double chi_value) {
  Matrix canvas;
  const Map::Parameter &map_parameter = map.getParameter();
  buildCanvas(canvas, map_parameter, res);

  int rows = static_cast<int>(canvas.rows());
  int cols = static_cast<int>(canvas.cols());

  for (auto it = map.cbeginVirtualLandmark(); it != map.cendVirtualLandnmark(); ++it) {
    int origin_row = static_cast<int>(floor((it->point.y() - map_parameter.getMinY()) / res));
    int origin_col = static_cast<int>(floor((it->point.x() - map_parameter.getMinX()) / res));

    double max_eig = 1.0 / minEigenvalue(it->information);
    int max_offset = static_cast<int>(ceil(sqrt(max_eig * chi_value) / res));

    int min_row = std::min(std::max(0, origin_row - max_offset), rows - 1);
    int max_row = std::min(std::max(0, origin_row + max_offset), rows - 1);
    int min_col = std::min(std::max(0, origin_col - max_offset), cols - 1);
    int max_col = std::min(std::max(0, origin_col + max_offset), cols - 1);

    const Eigen::Vector2d &u = it->point.vector();
//    double c = sqrt(I.determinant()) / (2 * M_PI);
    double c = 1.0;

    for (int row = min_row; row <= max_row; ++row) {
      for (int col = min_col; col <= max_col; ++col) {
        Eigen::Vector2d x;
        x << map_parameter.getMinX() + res * col,
            map_parameter.getMinY() + res * row;

        double cv = ((x - u).transpose() * it->information * (x - u))(0, 0);
        if (cv < chi_value) {
          double p = c * exp(-0.5 * cv);
          if (p > canvas(row, col))
            canvas(row, col) = p;
        }
      }
    }
  }

  return canvas;
}

PYBIND11_PLUGIN(ss2dpainter) {
  py::module m("ss2dpainter", "2D simulation and SLAM mudule");

  m.def("paint_map", &paintMap);
  m.def("paint_trajectory", &paintTrajectory);
  m.def("paint_virtual_map", &paintVirtualMap);
  m.def("paint_vehicle", &paintVehicle);

  return m.ptr();
}
