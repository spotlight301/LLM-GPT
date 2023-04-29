#include "justlm_pool.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>



bool LM::InferencePool::store_slot(Slot &slot) {
    auto& inference = slot.get_inference();
    // Open output file
    std::ofstream f(get_slot_filename(slot.get_id()), std::ios::binary);
    // Write weights path
    auto weights_path = slot.get_weights_path();
    uint32_t weights_path_len = weights_path.size();
    f.write(reinterpret_cast<const char*>(&weights_path_len), sizeof(weights_path_len));
    f.write(weights_path.data(), weights_path.size());
    // Write params
    if (!f.write(reinterpret_cast<const char*>(&inference.params), sizeof(inference.params))) {
        return false;
    }
    // Serialize instance
    try {
        inference.serialize(f);
    } catch (...) {
        return false;
    }
    // Return success
    return true;
}

LM::InferencePool::Slot *LM::InferencePool::load_slot(size_t id, Slot *suggested_slot) {
    // Open input file
    std::ifstream f(get_slot_filename(id), std::ios::binary);
    if (!f) {
        // Does not exist
        return nullptr;
    }
    // Read weights path
    std::string weights_path;
    uint32_t weights_path_len;
    if (!f.read(reinterpret_cast<char*>(&weights_path_len), sizeof(weights_path_len))) {
        return nullptr;
    }
    weights_path.resize(weights_path_len);
    if (!f.read(weights_path.data(), weights_path.size())) {
        return nullptr;
    }
    // Read params
    LM::Inference::Params p;
    if (!f.read(reinterpret_cast<char*>(&p), sizeof(p))) {
        return nullptr;
    }
    // Create instance
    auto& slot = suggested_slot?*suggested_slot:get_free_slot();
    auto& inference = slot.create_inference(id, weights_path, p);
    // Deserialize instance
    try {
        inference.deserialize(f);
    } catch (...) {
        slot.reset();
        return nullptr;
    }
    // Return final slot
    return &slot;
}

LM::InferencePool::Slot &LM::InferencePool::get_free_slot() {
    // Attempt to find free slot while finding oldest one
    Slot *oldest = nullptr;
    for (auto& slot : slots) {
        // Take free slot
        if (slot.is_free()) {
            return slot;
        }
        // Update oldest
        if (oldest == nullptr || slot.get_last_access() < oldest->get_last_access()) {
            oldest = &slot;
        }
    }
    // Free up oldest slot and take that one
    // Note: Since there has to be at least 1 slot, oldest is never going to be a nullptr
    store_and_reset_slot(*oldest);
    return *oldest;
}

LM::InferencePool::Slot *LM::InferencePool::find_slot_by_id(size_t id, bool deserialize) {
    // Attempt to find given slot while finding oldest one
    Slot *oldest = nullptr;
    for (auto& slot : slots) {
        // Take slot with ID
        if (slot.get_id() == id) {
            return &slot;
        }
        // Update oldest
        if (oldest == nullptr || slot.get_last_access() < oldest->get_last_access()) {
            oldest = &slot;
        }
    }
    // Slot not found, attempt to load it
    if (deserialize) {
        if (!oldest->is_free()) store_slot(*oldest);
        if (!load_slot(id, oldest)) {
            // In case slot loading failed, still reset slot for later use
            //TODO: Make this configurable
            oldest->reset();
        } else {
            return oldest;
        }
    }
    // Slot not found
    return nullptr;
}

std::optional<std::reference_wrapper<LM::Inference> > LM::InferencePool::get_inference(size_t id) {
    auto slot = find_slot_by_id(id);
    if (slot) {
        return slot->get_inference(true);
    }
    return {};
}

LM::Inference &LM::InferencePool::get_or_create_inference(size_t id, const std::string &weights_path, const Inference::Params &p) {
    auto slot = find_slot_by_id(id);
    if (slot) {
        return slot->get_inference(true);
    }
    slot = &get_free_slot();
    return slot->create_inference(id, weights_path, p);
}

void LM::InferencePool::delete_inference(size_t id) {
    auto slot = find_slot_by_id(id, false);
    // Reset slot
    if (slot) {
        slot->reset();
    }
    // Delete file
    std::error_code ec;
    std::filesystem::remove(get_slot_filename(id), ec);
}

void LM::InferencePool::store_all() {
    for (auto& slot : slots) {
        if (slot.is_free()) continue;
        store_slot(slot);
    }
}

std::vector<size_t> LM::InferencePool::get_active_slot_ids() const {
    std::vector<size_t> fres;
    for (const auto& slot : slots) {
        fres.push_back(slot.get_id());
    }
    return fres;
}

void LM::InferencePool::cleanup() {
    // Collect files
    const auto prefix = get_slot_filename_prefix();
    for (auto& file : std::filesystem::directory_iterator(".")) {
        if (file.path().filename().string().find(prefix) != 0) continue;
        std::error_code ec;
        std::filesystem::remove(file, ec);
    }
}

template<typename TP>
std::time_t to_time_t(TP tp) {
    using namespace std::chrono;
    auto sctp = time_point_cast<system_clock::duration>(tp - TP::clock::now()
              + system_clock::now());
    return system_clock::to_time_t(sctp);
}

void LM::InferencePool::cleanup(time_t max_age) {
    const auto current_time = to_time_t(std::chrono::system_clock::now());
    // Collect files
    const auto prefix = get_slot_filename_prefix();
    for (auto& file : std::filesystem::directory_iterator(".")) {
        if (file.path().filename().string().find(prefix) != 0) continue;
        // Delete files older than max age
        if (current_time - to_time_t(file.last_write_time()) > max_age) {
            std::error_code ec;
            std::filesystem::remove(file, ec);
        }
    }
}
