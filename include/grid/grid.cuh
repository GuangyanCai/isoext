#pragma once

class Grid {
  public:
    Grid() = default;

    virtual ~Grid() = default;

    virtual NDArray<float3> get_points() const = 0;

    virtual NDArray<float> get_values() const = 0;

    virtual void set_values(const NDArray<float> &new_values) = 0;
};
