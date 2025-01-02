#pragma once

#include "math.cuh"
#include <memory>
#include <string>
#include <thrust/device_vector.h>
#include <unordered_map>

namespace mc {

// Base class for marching cubes variants
class MCBase {
  public:
    virtual ~MCBase() = default;

    virtual void run(float3 *v, const uint num_cells, const uint8_t *cases,
                     const uint *cell_indices, const float *values,
                     const float3 *points, const uint *cells,
                     const float level) = 0;

    virtual size_t get_max_triangles() const = 0;

    // Static factory and registration methods
    static std::unique_ptr<MCBase> create(const std::string &method);
    static void
    register_variant(const std::string &name,
                     std::function<std::unique_ptr<MCBase>()> creator);

  private:
    static std::unordered_map<std::string,
                              std::function<std::unique_ptr<MCBase>()>> &
    get_registry();
};

// Helper class for automatic registration
template <typename T> class MCRegistrar {
  public:
    MCRegistrar(const std::string &name) {
        MCBase::register_variant(name, []() { return std::make_unique<T>(); });
    }
};

}   // namespace mc