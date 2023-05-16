#include "justlm.hpp"
#include "dlhandle.hpp"

#include <string>
#include <vector>
#include <fstream>
#include <filesystem>



static
Dlhandle get_implementation(uint32_t magic) {
    Dlhandle matching;
    Dlhandle fallback;
    // Iterate over all libraries
    for (const auto& f : std::filesystem::directory_iterator(".")) {
        // Get path
        const auto& p = f.path();
        // Check extension
        if (p.extension() != LIB_FILE_EXT) continue;
        // Load library
        try {
            Dlhandle dl(p);
            // Get implementation info getter
            auto implementation_getter = dl.get<const LM::Implementation *()>("get_justlm_implementation");
            if (!implementation_getter) continue;
            // Get implementation info
            const auto *implementation_info = implementation_getter();
            // Set if fallback
            if (implementation_info->is_fallback) {
                fallback = std::move(dl);
                continue;
            }
            // Set if matching magic
            auto magic_match = dl.get<bool(uint32_t)>("magic_match");
            if (magic_match && magic_match(magic)) {
                matching = std::move(dl);
                continue;
            }
        } catch (...) {}
    }
    // Return matching if any, fallback otherwise
    if (matching) return matching;
    return fallback;
}

LM::Inference *LM::Inference::construct(const std::string &weights_path, const Params &p) {
    static std::vector<Dlhandle> dls;
    // Read magic
    std::ifstream f(weights_path, std::ios::binary);
    uint32_t magic;
    if (!f.read(reinterpret_cast<char*>(&magic), sizeof(magic))) {
        throw Exception("Failed to open weights file for reading at "+weights_path);
    }
    f.seekg(0);
    // Get correct implementation
    auto impl = get_implementation(magic);
    if (!impl) return nullptr;
    // Get inference constructor
    auto constructor = impl.get<LM::Inference *(const std::string &, std::ifstream&, const LM::Inference::Params &)>("construct");
    if (!constructor) return nullptr;
    // Back up Dlhandle
    dls.push_back(std::move(impl));
    // Construct inference
    return constructor(weights_path, f, p);
}
