#pragma once

#include <array>
#include <nanobind/ndarray.h>

// Input types
using GridType = nanobind::ndarray<nanobind::pytorch, float, nanobind::ndim<3>,
                                   nanobind::device::cuda, nanobind::c_contig>;
using CellType =
    nanobind::ndarray<nanobind::pytorch, float, nanobind::shape<-1, 2, 2, 3>,
                      nanobind::device::cuda, nanobind::c_contig>;
using AABBType = std::array<float, 6>;

// Output types
using VerticesType =
    nanobind::ndarray<nanobind::pytorch, float, nanobind::shape<-1, 3>,
                      nanobind::device::cuda, nanobind::c_contig>;
using FacesType =
    nanobind::ndarray<nanobind::pytorch, int, nanobind::shape<-1, 3>,
                      nanobind::device::cuda, nanobind::c_contig>;
