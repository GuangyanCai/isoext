#include "mc_base.cuh"

namespace mc {

std::unordered_map<std::string, std::function<std::unique_ptr<MCBase>()>> &
MCBase::get_registry() {
    static std::unordered_map<std::string,
                              std::function<std::unique_ptr<MCBase>()>>
        registry;
    return registry;
}

void
MCBase::register_variant(const std::string &name,
                         std::function<std::unique_ptr<MCBase>()> creator) {
    get_registry()[name] = creator;
}

std::unique_ptr<MCBase>
MCBase::create(const std::string &method) {
    auto &registry = get_registry();
    auto it = registry.find(method);
    if (it == registry.end()) {
        throw std::runtime_error("Unknown method: " + method);
    }
    return it->second();
}

}   // namespace mc